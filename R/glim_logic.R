#' @useDynLib GLIM, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom progress progress_bar
#' @importFrom parallel detectCores
#' @importFrom stats logLik glm Gamma poisson inverse.gaussian lm coef runif qchisq model.matrix rnorm
#' @importFrom utils head
NULL


# TODO Warnings on whether p > n

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
#' @noRd
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
      # If we get a function crash, reset the threads back to what it originally was
      on.exit(RhpcBLASctl::blas_set_num_threads(original_blas_threads), add = TRUE)
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
#' @noRd
glim_inner_prob_approx_samples <- function(
  X,
  y,
  family = "gaussian",
  mle_coefs,
  mle_val,
  m,
  parallel
) {
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
    J <- crossprod(X, X * (mu_i^3) / 4)
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
#' @param m Number of samples/evaluations to perform (default `10000`).
#' @param approx Logical indicating whether to use the elliptical approximation (default `FALSE`).
#' @param parallel Logical indicating whether to process in parallel.
#' @param intercept Logical indicating whether to add an intercept term
#' @return A matrix of outputs or samples depending on whether the approximation is used.
#' @export
glim <- function(
  X,
  y,
  family = "gaussian",
  betas,
  m = 10000,
  approx = FALSE,
  parallel = TRUE,
  intercept = TRUE
) {
  if (is.data.frame(X)) {
    X <- as.matrix(X)
  }
  if (intercept == TRUE) {
    # Generates a matrix of TRUEs and FALSEs. Then takes colSums to see if any column consists of just 1s. Fails for the c(2, 2, 2, ..) case, but why on earth would you do that???
    has_intercept <- any(colSums(X == 1) == nrow(X))
    if (!has_intercept) {
      X <- cbind("(Intercept)" = 1, X)
    }
  }
  if (family == "binomial" || family == "logistic") {
    if (is.vector(y) && all(y == 0 | y == 1)) {
      if (length(y) != nrow(X)) {
        print("y and X lengths differ")
        stop()
      }
    } else if (is.matrix(y) && ncol(y) == 2) {
      # Two-column integer matrix (Successes / Failures)
      successes <- y[, 1]
      failures <- y[, 2]

      idx_success <- rep(1:nrow(X), times = successes)
      idx_fail <- rep(1:nrow(X), times = failures)

      # I did not know this, but X[c(2, 2), ] will repeat the second index twice
      # drop = FALSE means that it stays as a matrix, rather than going to a vector
      X <- rbind(X[idx_success, , drop = FALSE], X[idx_fail, , drop = FALSE])
      y <- c(rep(1, sum(successes)), rep(0, sum(failures)))
    } else if (is.factor(y)) {
      # Success is any level that is NOT the first level

      is_success <- as.integer(y != levels(y)[1])
      successes <- is_success * rep(1, length(y))
      failures <- (1 - is_success) * rep(1, length(y))

      idx_success <- rep(1:nrow(X), times = successes)
      idx_fail <- rep(1:nrow(X), times = failures)

      # I did not know this, but X[c(2, 2), ] will repeat the second index twice
      # drop = FALSE means that it stays as a matrix, rather than going to a vector
      y <- c(rep(1, sum(successes)), rep(0, sum(failures)))
      X <- rbind(X[idx_success, , drop = FALSE], X[idx_fail, , drop = FALSE])
    }
    # We are using (-1) so that R knows the X matrix we are using, we don't want to append any extra intercepts
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
    mle_coefs <- fit_gaussian_cpp(X, y)
  } else {
    stop("Family not supported")
  }
  if (is.vector(betas)) {
    betas <- matrix(betas, nrow = 1)
  }
  if (approx == FALSE) {
    return(glim_raw(
      X,
      y,
      family = family,
      betas,
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
#' @param intercept Whether or not an intercept should be added to the design matrix
#' @return A vector of mapped possibility values.
#' @export
prob2poss_logis <- function(X, y, samples, the_compared_theta, intercept = TRUE) {
  if (is.data.frame(X)) {
    X <- as.matrix(X)
  }
  if (intercept == TRUE) {
    # Generates a matrix of TRUEs and FALSEs. Then takes colSums to see if any column consists of just 1s. Fails for the c(2, 2, 2, ..) case, but why on earth would you do that???
    has_intercept <- any(colSums(X == 1) == nrow(X))
    if (!has_intercept) {
      X <- cbind(rep(1, length(y)), X)
    }
  }
  if (is.vector(y) && all(y == 0 | y == 1)) {
    if (length(y) != nrow(X)) {
      print("y and X lengths differ")
      stop()
    }
  } else if (is.matrix(y) && ncol(y) == 2) {
    # Two-column integer matrix (Successes / Failures)
    successes <- y[, 1]
    failures <- y[, 2]

    idx_success <- rep(1:nrow(X), times = successes)
    idx_fail <- rep(1:nrow(X), times = failures)

    # I did not know this, but X[c(2, 2), ] will repeat the second index twice
    # drop = FALSE means that it stays as a matrix, rather than going to a vector
    X <- rbind(X[idx_success, , drop = FALSE], X[idx_fail, , drop = FALSE])
    y <- c(rep(1, sum(successes)), rep(0, sum(failures)))
  } else if (is.factor(y)) {
    # Success is any level that is NOT the first level

    is_success <- as.integer(y != levels(y)[1])
    successes <- is_success * rep(1, length(y))
    failures <- (1 - is_success) * rep(1, length(y))

    idx_success <- rep(1:nrow(X), times = successes)
    idx_fail <- rep(1:nrow(X), times = failures)

    # I did not know this, but X[c(2, 2), ] will repeat the second index twice
    # drop = FALSE means that it stays as a matrix, rather than going to a vector
    y <- c(rep(1, sum(successes)), rep(0, sum(failures)))
    X <- rbind(X[idx_success, , drop = FALSE], X[idx_fail, , drop = FALSE])
    X <- cbind(rep(1, length(y)), X)
  }
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
#' @param intercept Whether or not an intercept should be added to the design matrix
#' @return A vector of mapped possibility values.
#' @export
prob2poss_gamma <- function(X, y, samples, the_compared_theta, intercept = TRUE) {
  if (is.data.frame(X)) {
    X <- as.matrix(X)
  }
  if (intercept == TRUE) {
    # Generates a matrix of TRUEs and FALSEs. Then takes colSums to see if any column consists of just 1s. Fails for the c(2, 2, 2, ..) case, but why on earth would you do that???
    has_intercept <- any(colSums(X == 1) == nrow(X))
    if (!has_intercept) {
      X <- cbind(rep(1, length(y)), X)
    }
  }
  eta <- X %*% samples
  initial_coefs <- coef(lm(log(y) ~ X - 1))
  mle_coefs <- fit_gamma_log_cpp(X, t(X) %*% X, y, initial_coefs, FALSE)
  est_shape <- 1 / mle_estimate_dispersion_gamma(y, exp(X %*% mle_coefs), length(mle_coefs))

  ll_val_samps <- as.vector(compute_gamma_ll_r(y, eta, shape = est_shape))
  ll_val <- as.vector(compute_gamma_ll_r(y, X %*% the_compared_theta, shape = est_shape))
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

#' Compute Gaussian Log-Likelihood
#' @param y Response vector.
#' @param mu Mean (can be a vector or a matrix).
#' @param sigma Estimated standard deviation
#' @return The calculated log-likelihood.
#' @export
compute_gaussian_ll_r <- function(y, mu, sigma) {
  if (is.vector(mu)) {
    return(compute_gaussian_ll(y, mu, sigma))
  } else {
    return(compute_gaussian_ll_mat(y, mu, sigma))
  }
}
#' Compute inverse gaussian Log-Likelihood
#'
#' @param y Response vector.
#' @param mu Mean (can be a vector or a matrix).
#' @param gamma_val Estimated dispersion
#' @return The calculated log-likelihood.
#' @export
compute_invgauss_ll_r <- function(y, mu, gamma_val) {
  if (is.vector(mu)) {
    return(compute_invgauss_ll(y, mu, gamma_val))
  } else {
    return(compute_invgauss_ll_mat(y, mu, gamma_val))
  }
}


#' Probability to Possibility Mapping for Gaussian Regression
#'
#' @param X Predictor matrix.
#' @param y Response vector.
#' @param samples Matrix of simulated sample coefficients.
#' @param the_compared_theta The theta values to compare against.
#' @param intercept Whether or not an intercept should be added to the design matrix
#' @return A vector of mapped possibility values.
#' @export
prob2poss_gaussian <- function(X, y, samples, the_compared_theta, intercept = TRUE) {
  if (is.data.frame(X)) {
    X <- as.matrix(X)
  }
  if (intercept == TRUE) {
    # Generates a matrix of TRUEs and FALSEs. Then takes colSums to see if any column consists of just 1s. Fails for the c(2, 2, 2, ..) case, but why on earth would you do that???
    has_intercept <- any(colSums(X == 1) == nrow(X))
    if (!has_intercept) {
      X <- cbind(rep(1, length(y)), X)
    }
  }

  eta <- X %*% samples

  # Assuming identity link for Gaussian by default, or adjust to fit_gaussian_identity_cpp if that matches your backend
  mle_coefs <- fit_gaussian_cpp(X, y)
  est_sd <- sqrt(est_dispersion(y, X %*% mle_coefs, length(mle_coefs)))

  ll_val_samps <- as.vector(compute_gaussian_ll_r(y, eta, sigma = est_sd))
  ll_val <- as.vector(compute_gaussian_ll_r(y, X %*% the_compared_theta, sigma = est_sd))

  sapply(ll_val, function(x) sum(ll_val_samps < x) / length(ll_val_samps))
}


#' Probability to Possibility Mapping for Inverse Gaussian Regression
#'
#' Will throw an error if \code{the_compared_theta} is a scalar.
#'
#' @param X Predictor matrix.
#' @param y Response vector.
#' @param samples Matrix of simulated sample coefficients.
#' @param the_compared_theta The theta values to compare against.
#' @param intercept Whether or not an intercept should be added to the design matrix
#' @return A vector of mapped possibility values.
#' @export
prob2poss_invgauss <- function(X, y, samples, the_compared_theta, intercept = TRUE) {
  if (is.data.frame(X)) {
    X <- as.matrix(X)
  }
  if (intercept == TRUE) {
    # Generates a matrix of TRUEs and FALSEs. Then takes colSums to see if any column consists of just 1s. Fails for the c(2, 2, 2, ..) case, but why on earth would you do that???
    has_intercept <- any(colSums(X == 1) == nrow(X))
    if (!has_intercept) {
      X <- cbind(rep(1, length(y)), X)
    }
  }
  eta <- X %*% samples

  # Providing initial coefficients using a rough approximation
  initial_coefs <- coef(lm(1 / (y^2) ~ X - 1))

  mle_coefs <- fit_invgauss_cpp(X, y, initial_coefs, FALSE)

  est_dispersion <- mle_estimate_dispersion_inv_gauss(y, mean(y))

  # Check whether mapping from eta to mu is correct
  ll_val_samps <- as.vector(compute_invgauss_ll_r(y, sqrt(1 / eta), gamma_val = est_dispersion))
  ll_val <- as.vector(compute_invgauss_ll_r(
    y,
    sqrt(1 / X %*% the_compared_theta), # should do it element-wise
    gamma_val = est_dispersion
  ))

  sapply(ll_val, function(x) sum(ll_val_samps < x) / length(ll_val_samps))
}


# #' Faster Elliptical Approximation Samples (Alternative Implementation)
# #'
# #' Function called if doing elliptical approx, designed for better performance.
# #'
# #' @param X Predictor matrix.
# #' @param y Response vector.
# #' @param family A string indicating the error distribution. Default is `"gaussian"`.
# #' @param mle_val The log-likelihood value at the maximum likelihood estimates.
# #' @param m Number of samples to generate.
# #' @param parallel Logical indicating whether to use parallel processing.
# #' @param a Hyperparameter `a` for the approximation tuning.
# #' @param b Hyperparameter `b` for the approximation tuning.
# #' @param max_it Maximum number of iterations allowed for the algorithm.
# #' @param tol Tolerance criteria for convergence.
# #' @return A matrix of generated samples.
# #' @export
# glim_inner_prob_approx_samples_2 <- function(
#   X,
#   y,
#   family = "gaussian",
#   mle_val,
#   m,
#   parallel,
#   a,
#   b,
#   max_it,
#   tol
# ) {
#   print("glim_inner_prob")
#   B <- 1000
#   AA <- seq(0.001, 0.999, length = B)
#   if (family == "gaussian" || family == "normal") {
#     res <- lm(y ~ X - 1)
#     J <- crossprod(X, X)
#     dispersion <- 1
#   } else if (family == "binomial") {
#     res <- glm(y ~ X - 1, family = "binomial")
#     p_i <- res$fitted.values
#     J <- crossprod(X, X * (p_i * (1 - p_i)))
#     dispersion <- 1
#   } else if (family == "gamma") {
#     res <- glm(y ~ X - 1, family = Gamma(link = "log"))
#     J <- crossprod(X, X)
#     mle_coefs <- res$coefficients
#     dispersion <- mle_estimate_dispersion_gamma(y, exp(X %*% mle_coefs), length(mle_coefs))
#   } else if (family == "inverse.gaussian") {
#     res <- glm(y ~ X - 1, family = inverse.gaussian(link = "1/mu^2"))
#     eta <- X %*% res$coefficients
#     mu_i <- as.vector(sqrt(1 / eta))
#     J <- crossprod(X, X * (mu_i^3)) # TODO #7 check on this calculation
#     dispersion <- mle_estimate_dispersion_inv_gauss(y, mean(y))
#   } else if (family == "poisson") {
#     res <- glm(y ~ X - 1, family = poisson(link = "log"))
#     lambda_i <- res$fitted.values
#     J <- crossprod(X, X * (lambda_i))
#     dispersion <- 1
#   }
#   J <- (J + t(J)) / 2
#   eJ <- eigen(J)

#   eJ$values[eJ$values < 1e-4] <- .000001
#   mle_coefs <- res$coefficients
#   eJ_vectors <- eJ$vectors
#   ej_values <- diag(eJ$values)

#   matrix_of_xis <- get_xi(
#     AA,
#     mle_coefs,
#     family,
#     eJ_vectors,
#     eJ_values,
#     dispersion,
#     mle_val,
#     a,
#     b,
#     max_it,
#     tol
#   )

#   u <- runif(m)
#   lerped_xi <- c()
#   samples <- matrix(nrow = length(mle_coefs), ncol = m)
#   for (i in 1:m) {
#     if (u[i] < 1 / length(AA)) {
#       lerped_xi <- matrix_of_xis[, 1]
#     } else if (u[i] > 1 - 1 / length(AA)) {
#       lerped_xi <- matrix_of_xis[, length(AA)]
#     } else {
#       where_located <- findInterval(u[i], AA)
#       w <- u[i] - AA[where_located]
#       lerped_xi <- w * matrix_of_xis[, where_located] + (1 - w) * matrix_of_xis[, where_located + 1]
#     }
#     if (is.na(lerped_xi)) {
#       print("You dun messed up")
#     }

#     rand_dir <- generate_unit_matrix(1, length(mle_coefs))
#     spatial_dir <- eJ$vectors %*% (1 / sqrt(eJ$values) * rand_dir)

#     samples[, i] <- mle_coefs +
#       as.vector(sqrt(qchisq(1 - u, length(mle_coefs))) * lerped_xi * spatial_dir)
#   }
#   return(samples)
# }

# #' Appendix C++ Bridge Function
# #'
# #' Helper function acting as a bridge to underlying C++ routines for sample generation.
# #'
# #' @param eJ Eigen decomposition object.
# #' @param num_samps Number of samples to generate.
# #' @param d Dimensionality parameter.
# #' @param X Predictor matrix.
# #' @param y Response vector.
# #' @param mle_coefs Maximum likelihood estimates for coefficients.
# #' @param family String denoting the exponential family.
# #' @param dispersion The dispersion parameter.
# #' @param m Parameter `m` defining scaling or sampling limits.
# #' @param tol Tolerance level for convergence criteria.
# #' @param max_it Maximum number of iterations.
# #' @param a Hyperparameter `a` for the underlying routine.
# #' @param b Hyperparameter `b` for the underlying routine.
# #' @return A matrix of output samples evaluated by the C++ backend.
# #' @export
# appendix <- function(eJ, num_samps, d, X, y, mle_coefs, family, dispersion, m, tol, max_it, a, b) {
#   eig_vecs <- eJ$vectors
#   eig_vals <- eJ$values
#   output_samples <- lets_go_to_cpp(
#     eig_vecs,
#     eig_vals,
#     num_samps,
#     d,
#     X,
#     y,
#     mle_coefs,
#     family,
#     dispersion,
#     m,
#     tol,
#     max_it,
#     a,
#     b
#   )
#   return(output_samples)
# }

#' Probability to Possibility Mapping for Poisson Regression
#'
#' @param X Predictor matrix.
#' @param y Response vector.
#' @param samples Matrix of simulated sample coefficients.
#' @param the_compared_theta The theta values to compare against.
#' @param intercept Whether or not an intercept should be added to the design matrix
#' @return A vector of mapped possibility values.
#' @export
prob2poss_poisson <- function(X, y, samples, the_compared_theta, intercept = TRUE) {
  if (is.data.frame(X)) {
    X <- as.matrix(X)
  }
  if (intercept == TRUE) {
    # Generates a matrix of TRUEs and FALSEs. Then takes colSums to see if any column consists of just 1s. Fails for the c(2, 2, 2, ..) case, but why on earth would you do that???
    has_intercept <- any(colSums(X == 1) == nrow(X))
    if (!has_intercept) {
      X <- cbind(rep(1, length(y)), X)
    }
  }
  eta <- X %*% samples
  mle_coefs <- fit_poisson_log_cpp(X, y)
  ll_val_samps <- as.vector(compute_poisson_ll_r(y, eta))
  ll_val <- as.vector(compute_poisson_ll_r(y, X %*% the_compared_theta))
  sapply(ll_val, function(x) sum(ll_val_samps < x) / length(ll_val_samps))
}


#' Returns a confidence interval
#'
#' Note that a 95% confidence interval corresponds to alpha = .95
#'
#' @param alpha Note that a 95% confidence interval corresponds to alpha = .95
#' @param betas The grid of beta values which we are evaluating on
#' @param possibilities The vector of possibilities
#' @return A matrix of compatable beta values from the grid.
#' @export
get_CI <- function(alpha, betas, possibilities) {
  return(betas[possibilities > alpha, ])
}
