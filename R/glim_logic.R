#' @useDynLib GLIM, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom progress progress_bar
#' @importFrom RhpcBLASctl blas_set_num_threads
#' @importFrom parallel detectCores
NULL

# Original code written by Joe Harrison (jrharr25@ncsu.edu), translated to cpp by Gemini
# pkgbuild::compile_dll() validates the package directory differently. Rcpp might implement it's directory check differently
# After making changes, restart R, get in the package directory
# devtools::document()
# devtools::install() or devtools::load_all()

#' Fits GLIM (Raw Implementation)
#'
#' Evaluates possibility for beta/dispersion values.
#'
#' @param X Input predictor matrix. If a data frame is provided, it will be coerced to a model matrix.
#' @param y Response vector.
#' @param family A string indicating the error distribution. Options include `"gaussian"`, `"poisson"`, `"gamma"`, `"binomial"`.
#' @param betas A grid of beta values to evaluate, where each column is a new beta vector. Not required if `approx = TRUE`.
#' @param mle_coefs Maximum likelihood estimates of the coefficients.
#' @param mle_val The log-likelihood value at the maximum likelihood estimates.
#' @param m Number of evaluations per beta (number of samples).
#' @param parallel Logical indicating whether to run in parallel (primarily for debugging).
#' @param approx Logical indicating whether to use the elliptical approximation.
#' @return A matrix of evaluated possibility outputs.
#' @export
glim_raw <- function(X, y, family = "gaussian", betas, mle_coefs, mle_val, m, parallel, approx) {
  if (is.data.frame(X)) {
    X <- model.matrix(X)
  }
  output <- matrix()

  if (parallel) {
    num_omp_threads <- max(1, parallel::detectCores() - 1)

    # We change the number of blas threads down to 1 so that the parallelization doesn't
    # request even more threads.
    if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
      original_blas_threads <- RhpcBLASctl::blas_get_num_procs()
      RhpcBLASctl::blas_set_num_threads(1)
    }
  } else {
    num_omp_threads <- 1
  }

  output <- fit_glm_omp_cpp(
    X = X,
    y = y,
    mle_coefs = mle_coefs,
    betas = betas,
    family = family, # Pass the string straight down
    num_threads = num_omp_threads,
    m = m,
    parallel = parallel,
    approx = approx
  )

  # If we changed the amount of threads that blas uses, change this back. Globally for the R session.
  if (parallel && requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    RhpcBLASctl::blas_set_num_threads(original_blas_threads)
  }

  return(output)
}

#' Generate Elliptical Approximation Samples (Inner Probability)
#'
#' Internal function called to generate samples when the elliptical approximation is used.
#'
#' @param X Input predictor matrix.
#' @param y Response vector.
#' @param family A string indicating the error distribution. Default is `"gaussian"`.
#' @param mle_coefs Maximum likelihood estimates of the coefficients.
#' @param mle_val The log-likelihood value at the maximum likelihood estimates.
#' @param m Number of samples to generate.
#' @param parallel Logical indicating whether to use parallel processing.
#' @return A matrix of generated samples based on the elliptical approximation.
#' @export
glim_inner_prob_approx_samples <- function(
  X,
  y,
  family = "gaussian",
  mle_coefs,
  mle_val,
  m,
  parallel
) {
  print("glim_inner_prob")
  B <- 100
  AA <- seq(0.001, 0.999, length = B)
  if (family == "gaussian" || family == "normal") {
    res <- lm(y ~ X - 1)
    J <- crossprod(X, X)
    dispersion <- 1
  } else if (family == "binomial") {
    res <- glm(y ~ X - 1, family = "binomial")
    p_i <- res$fitted.values
    J <- crossprod(X, X * as.vector((p_i * (1 - p_i))))
    dispersion <- 1
  } else if (family == "gamma") {
    res <- glm(y ~ X - 1, family = Gamma(link = "log"))
    J <- crossprod(X, X)
    mle_coefs <- res$coefficients
    dispersion <- mle_estimate_dispersion_gamma(y, exp(X %*% mle_coefs), length(mle_coefs))
  } else if (family == "inverse.gaussian") {
    res <- glm(y ~ X - 1, family = inverse.gaussian(link = "1/mu^2"))
    eta <- X %*% res$coefficients
    mu_i <- as.vector(sqrt(1 / eta))
    J <- crossprod(X, X * (mu_i^3)) # TODO #7 check on this calculation
    dispersion <- mle_estimate_dispersion_inv_gauss(y, mean(y))
  } else if (family == "poisson") {
    res <- glm(y ~ X - 1, family = poisson(link = "log"))
    eta <- X %*% coef(res)
    lambda_i <- exp(eta)
    J <- crossprod(X, X * as.vector(lambda_i))
    dispersion <- 1
  }
  J <- (J + t(J)) / 2 # symmetrize to try to kill some rounding asymmetries
  eJ <- eigen(J)

  eJ$values[eJ$values < 1e-4] <- .000001
  mle_coefs <- res$coefficients

  pl <- function(z) {
    betas_matrix <- if (is.matrix(z)) z else matrix(z, nrow = 1)
    glim_raw(X, y, family, betas_matrix, mle_coefs, mle_val = mle_val, m, parallel, approx = TRUE)
  }

  i <- 0
  xi <- list()
  prev_xi <- rep(1, length(mle_coefs))
  pb <- progress::progress_bar$new(
    total = length(AA),
    format = "[:bar] :percent eta :eta",
    show_after = 0,
    force = TRUE
  )
  parallel <- FALSE
  for (a in AA) {
    pb$tick()
    i <- i + 1
    xi[[i]] <- imvar(
      X,
      y,
      prev_xi,
      family,
      a,
      mle = mle_coefs,
      mle_val,
      as.matrix(eJ$vectors),
      as.vector(eJ$values),
      dispersion,
      tol = .01,
      a = 2,
      b = 1,
      max_it = 25,
      parallel = FALSE
    )
    prev_xi <- xi[[i]]
  }

  print("We've gotten the xi's")
  U <- runif(m)
  lerped_xi <- -1
  samples <- matrix(nrow = length(mle_coefs), ncol = m)
  i <- 0
  for (u in U) {
    i <- i + 1
    if (u < min(AA)) {
      lerped_xi <- xi[[1]]
    } else if (u > max(AA)) {
      lerped_xi <- xi[[B]]
    } else {
      r <- sum(AA < u)
      w <- (u - AA[r]) / (AA[r + 1] - AA[r])
      lerped_xi <- (1 - w) * xi[[r]] + w * xi[[r + 1]]
    }

    rand_dir <- generate_unit_matrix(1, length(mle_coefs))
    spatial_dir <- eJ$vectors %*% (1 / sqrt(eJ$values) * rand_dir)

    samples[, i] <- mle_coefs +
      as.vector(sqrt(qchisq(1 - u, length(mle_coefs))) * sqrt(lerped_xi) * spatial_dir * dispersion)
  }
  return(samples)
}

#' Generalized Linear Inferential Models (GLIM) Main Function
#'
#' Main wrapper function to fit a GLIM model.
#'
#' @param X Matrix of predictors.
#' @param y Vector of response variables.
#' @param family String denoting the exponential family. Choices are `"gaussian"`, `"binomial"`, `"gamma"`, `"poisson"`, `"inverse-gaussian"`.
#' @param betas A matrix (or column vector) of different beta values to evaluate the possibility over.
#' @param m Number of samples/evaluations to perform (default `1000`).
#' @param approx Logical indicating whether to use the elliptical approximation (default `FALSE`).
#' @param parallel Logical indicating whether to process in parallel.
#' @return A matrix of outputs or samples depending on whether the approximation is used.
#' @export
glim <- function(X, y, family = "gaussian", betas, m = 1000, approx = FALSE, parallel) {
  print("glim_called")
  if (family == "binomial" || family == "logistic") {
    ll_mle_original_data <- as.numeric(logLik(glm(y ~ X - 1, family = "binomial")))
    mle_coefs <- glm(y ~ X - 1, family = "binomial")$coefficients
  } else if (family == "gamma") {
    mle_coefs <- glm(y ~ X - 1, family = Gamma(link = "log"))$coefficients
    eta <- X %*% mle_coefs
    ratio <- y / exp(eta)
    n <- length(y)
    ll_mle_original_data <- compute_gamma_ll_r(
      y,
      eta,
      1 / mle_estimate_dispersion_gamma(y, exp(eta), length(mle_coefs))
    )
  } else if (family == "poisson") {
    ll_mle_original_data <- as.numeric(logLik(glm(y ~ X - 1, family = poisson(link = "log"))))
    mle_coefs <- glm(y ~ X - 1, family = poisson(link = "log"))$coefficients
  } else if (family == "inverse-gaussian") {
    ll_mle_original_data <- as.numeric(logLik(glm(
      y ~ X - 1,
      family = inverse.gaussian(link = "1/mu^2")
    )))
    mle_coefs <- glm(y ~ X - 1, family = inverse.gaussian(link = "1/mu^2"))$coefficients
  } else if (family == "normal" || family == "gaussian") {
    ll_mle_original_data <- as.numeric(logLik(lm(y ~ X - 1)))
    mle_coefs <- lm(y ~ X - 1)$coefficients
  } else {
    stop("Family not supported")
  }
  if (approx == FALSE) {
    return(glim_raw(
      X,
      y,
      family = family,
      as.matrix(betas),
      mle_coefs,
      mle_val = ll_mle_original_data,
      m = m,
      parallel = parallel,
      approx = approx
    ))
  }

  if (approx == TRUE) {
    return(glim_inner_prob_approx_samples(
      X,
      y,
      family = family,
      mle_coefs,
      mle_val = ll_mle_original_data,
      m = m,
      parallel
    ))
  }
}

#' Probability to Possibility Mapping for Logistic Regression
#'
#' @param X Predictor matrix.
#' @param y Response vector.
#' @param samples Matrix of simulated sample coefficients.
#' @param the_compared_theta The theta values to compare against.
#' @return A vector of mapped possibility values.
#' @export
prob2poss_logis <- function(X, y, samples, the_compared_theta) {
  eta <- X %*% the_compared_theta
  log_term <- pmax(eta, 0) + log1p(exp(-abs(eta)))
  ll_val <- y %*% eta - colSums(log_term)

  eta_samps <- X %*% samples
  log_term_samps <- pmax(eta_samps, 0) + log1p(exp(-abs(eta_samps)))
  ll_val_samps <- as.vector(y %*% eta_samps) - colSums(log_term_samps)

  return(sapply(ll_val, function(x) sum(ll_val_samps < x)) / length(ll_val_samps))
}

#' Probability to Possibility Mapping for Gamma Regression
#'
#' Will throw an error if \code{the_compared_theta} is a scalar.
#'
#' @param X Predictor matrix.
#' @param y Response vector.
#' @param samples Matrix of simulated sample coefficients.
#' @param the_compared_theta The theta values to compare against.
#' @return A vector of mapped possibility values.
#' @export
prob2poss_gamma <- function(X, y, samples, the_compared_theta) {
  eta <- X %*% samples
  initial_coefs <- coef(lm(log(y) ~ X - 1))
  mle_coefs <- fit_gamma_log_cpp(X, t(X) %*% X, y, initial_coefs, FALSE)
  est_shape <- 1 / mle_estimate_dispersion_gamma(y, exp(X %*% mle_coefs), length(mle_coefs))

  ll_val_samps <- as.vector(compute_gamma_ll_r(y, eta, shape = est_shape))
  ll_val <- as.vector(compute_gamma_ll_r(y, X %*% the_compared_theta, shape = est_shape))
  message("Is ll_val sorted? ", !is.unsorted(ll_val))
  sapply(ll_val, function(x) sum(ll_val_samps < x) / length(ll_val_samps))
}

#' Compute Gamma Log-Likelihood
#'
#' @param y Response vector.
#' @param eta Linear predictor (can be a vector or a matrix).
#' @param shape The shape parameter for the gamma distribution.
#' @return The calculated log-likelihood.
#' @export
compute_gamma_ll_r <- function(y, eta, shape) {
  if (is.vector(eta)) {
    return(compute_gamma_ll(y, eta, shape))
  } else {
    return(compute_gamma_ll_mat(y, eta, shape))
  }
}

#' Compute Poisson Log-Likelihood
#'
#' @param y Response vector.
#' @param eta Linear predictor (can be a vector or a matrix).
#' @return The calculated log-likelihood.
#' @export
compute_poisson_ll_r <- function(y, eta) {
  if (is.vector(eta)) {
    return(compute_poisson_ll(eta, y))
  } else {
    return(compute_poisson_ll_mat(eta, y))
  }
}

#' Faster Elliptical Approximation Samples (Alternative Implementation)
#'
#' Function called if doing elliptical approx, designed for better performance.
#'
#' @param X Predictor matrix.
#' @param y Response vector.
#' @param family A string indicating the error distribution. Default is `"gaussian"`.
#' @param mle_val The log-likelihood value at the maximum likelihood estimates.
#' @param m Number of samples to generate.
#' @param parallel Logical indicating whether to use parallel processing.
#' @param a Hyperparameter `a` for the approximation tuning.
#' @param b Hyperparameter `b` for the approximation tuning.
#' @param max_it Maximum number of iterations allowed for the algorithm.
#' @param tol Tolerance criteria for convergence.
#' @return A matrix of generated samples.
#' @export
glim_inner_prob_approx_samples_2 <- function(
  X,
  y,
  family = "gaussian",
  mle_val,
  m,
  parallel,
  a,
  b,
  max_it,
  tol
) {
  print("glim_inner_prob")
  B <- 100
  AA <- seq(0.001, 0.999, length = B)
  if (family == "gaussian" || family == "normal") {
    res <- lm(y ~ X - 1)
    J <- crossprod(X, X)
    dispersion <- 1
  } else if (family == "binomial") {
    res <- glm(y ~ X - 1, family = "binomial")
    p_i <- res$fitted.values
    J <- crossprod(X, X * (p_i * (1 - p_i)))
    dispersion <- 1
  } else if (family == "gamma") {
    res <- glm(y ~ X - 1, family = Gamma(link = "log"))
    J <- crossprod(X, X)
    mle_coefs <- res$coefficients
    dispersion <- mle_estimate_dispersion_gamma(y, exp(X %*% mle_coefs), length(mle_coefs))
  } else if (family == "inverse.gaussian") {
    res <- glm(y ~ X - 1, family = inverse.gaussian(link = "1/mu^2"))
    eta <- X %*% res$coefficients
    mu_i <- as.vector(sqrt(1 / eta))
    J <- crossprod(X, X * (mu_i^3)) # TODO #7 check on this calculation
    dispersion <- mle_estimate_dispersion_inv_gauss(y, mean(y))
  } else if (family == "poisson") {
    res <- glm(y ~ X - 1, family = poisson(link = "log"))
    lambda_i <- res$fitted.values
    J <- crossprod(X, X * (lambda_i))
    dispersion <- 1
  }
  J <- (J + t(J)) / 2
  eJ <- eigen(J)

  eJ$values[eJ$values < 1e-4] <- .000001
  mle_coefs <- res$coefficients
  eJ_vectors <- eJ$vectors
  ej_values <- diag(eJ$values)

  matrix_of_xis <- get_xi(
    AA,
    mle_coefs,
    family,
    eJ_vectors,
    eJ_values,
    dispersion,
    mle_val,
    a,
    b,
    max_it,
    tol
  )

  u <- runif(m)
  lerped_xi <- c()
  samples <- matrix(nrow = length(mle_coefs), ncol = m)
  for (i in 1:m) {
    if (u[i] < 1 / length(AA)) {
      lerped_xi <- matrix_of_xis[, 1]
    } else if (u[i] > 1 - 1 / length(AA)) {
      lerped_xi <- matrix_of_xis[, length(AA)]
    } else {
      where_located <- findInterval(u[i], AA)
      w <- u[i] - AA[where_located]
      lerped_xi <- w * matrix_of_xis[, where_located] + (1 - w) * matrix_of_xis[, where_located + 1]
    }
    if (is.na(lerped_xi)) {
      print("You dun messed up")
    }

    rand_dir <- generate_unit_matrix(1, length(mle_coefs))
    spatial_dir <- eJ$vectors %*% (1 / sqrt(eJ$values) * rand_dir)

    samples[, i] <- mle_coefs +
      as.vector(sqrt(qchisq(1 - u, length(mle_coefs))) * lerped_xi * spatial_dir)
  }
  return(samples)
}

#' Appendix C++ Bridge Function
#'
#' Helper function acting as a bridge to underlying C++ routines for sample generation.
#'
#' @param eJ Eigen decomposition object.
#' @param num_samps Number of samples to generate.
#' @param d Dimensionality parameter.
#' @param X Predictor matrix.
#' @param y Response vector.
#' @param mle_coefs Maximum likelihood estimates for coefficients.
#' @param family String denoting the exponential family.
#' @param dispersion The dispersion parameter.
#' @param m Parameter `m` defining scaling or sampling limits.
#' @param tol Tolerance level for convergence criteria.
#' @param max_it Maximum number of iterations.
#' @param a Hyperparameter `a` for the underlying routine.
#' @param b Hyperparameter `b` for the underlying routine.
#' @return A matrix of output samples evaluated by the C++ backend.
#' @export
appendix <- function(eJ, num_samps, d, X, y, mle_coefs, family, dispersion, m, tol, max_it, a, b) {
  eig_vecs <- eJ$vectors
  eig_vals <- eJ$values
  output_samples <- lets_go_to_cpp(
    eig_vecs,
    eig_vals,
    num_samps,
    d,
    X,
    y,
    mle_coefs,
    family,
    dispersion,
    m,
    tol,
    max_it,
    a,
    b
  )
  return(output_samples)
}

#' Probability to Possibility Mapping for Poisson Regression
#'
#' @param X Predictor matrix.
#' @param y Response vector.
#' @param samples Matrix of simulated sample coefficients.
#' @param the_compared_theta The theta values to compare against.
#' @return A vector of mapped possibility values.
#' @export
prob2poss_poisson <- function(X, y, samples, the_compared_theta) {
  eta <- X %*% samples
  mle_coefs <- fit_poisson_log_cpp(X, y)
  ll_val_samps <- as.vector(compute_poisson_ll_r(y, eta))
  ll_val <- as.vector(compute_poisson_ll_r(y, X %*% the_compared_theta))
  message("Is ll_val sorted? ", !is.unsorted(ll_val))
  sapply(ll_val, function(x) sum(ll_val_samps < x) / length(ll_val_samps))
}

#' Fit GLM using OpenMP via Rcpp
#'
#' A wrapper to the underlying `fit_glm_omp_cpp` C++ implementation.
#'
#' @param X Predictor matrix.
#' @param y Response vector.
#' @param mle_coefs Maximum likelihood estimates of the coefficients.
#' @param betas Grid of beta values to evaluate.
#' @param family String indicating the error distribution.
#' @param num_threads The number of OpenMP threads to utilize.
#' @param m Number of evaluations per beta.
#' @param parallel Logical indicating whether to run in parallel.
#' @param approx Logical indicating whether to use elliptical approximation.
#' @return Resulting output from the C++ GLM routine.
#' @export
fit_glm_omp_r <- function(X, y, mle_coefs, betas, family, num_threads, m, parallel, approx) {
  return(fit_glm_omp_cpp(X, y, mle_coefs, betas, family, num_threads, m, parallel, approx))
}

#' Poisson Possibility Evaluation via Rcpp
#'
#' Evaluates profile likelihood configurations for Poisson models leveraging C++.
#'
#' @param X Predictor matrix.
#' @param y Response vector.
#' @param mle_coefs Maximum likelihood estimates of the coefficients.
#' @param beta_vals Matrix of beta configurations.
#' @param m Number of samples/evaluations to perform.
#' @param approx Logical indicating whether to use approximation logic.
#' @return A vector or matrix of profile likelihood results from the C++ backend.
#' @export
pois_pos <- function(X, y, mle_coefs, beta_vals, m, approx) {
  return(glm_poisson_pl_cpp(X, y, mle_coefs, beta_vals, m, approx))
}
