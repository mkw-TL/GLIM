#' @useDynLib GLIM, .registration = TRUE
#' @importFrom Rcpp sourceCpp
NULL
#
# Original code written by Joe Harrison (jrharr25@ncsu.edu), translated to cpp by Gemini
# pkgbuild::compile_dll() validates the package directory differently. Rcpp might implement it's directory check differently
# After making changes, restart R, get in the package directory
#devtools::document()
#devtools::install()
# Pass everything to our single unified C++ dispatcher function

# Evaluates possibility for beta/dispersion values
# m is the number of samples for each parameter value you would like to have
#' @export
glim_raw <- function(X, y, family = "gaussian", betas, mle_coefs, mle_val, m, parallel) {
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
    betas = betas,
    family_str = family, # Pass the string straight down
    num_threads = num_omp_threads,
    m = m
  )

  # If we changed the amount of threads that blas uses, change this back. Globally for the R session.
  if (parallel && requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    RhpcBLASctl::blas_set_num_threads(original_blas_threads)
  }

  return(output)
}

#' Generates all random numbers at once, so doesn't need to use the slow apply
#' Doesn't work if you pass in more than one column
#' @export
generate_unit_matrix <- function(n, d) {
  m <- matrix(rnorm(n * d), nrow = d, ncol = n)
  norms <- sqrt(colSums(m^2))
  return(m / norms)
}

#' @export
imvar <- function(xi, alpha, pl, mle, J, parallel, tol = 1e-2, a = 5, b = 1, max.it = 25) {
  D <- length(mle)
  maxpl <- function(v) {
    max(c(pl(as.vector(mle) + v), pl(as.vector(mle) - v)))
  }
  w <- function(s) a / (1 + s)**b # our weighting function, dampens over time
  for (d in 1:D) {
    # print(d)
    # log(xi) because we are getting the exponentiated version (so that it is for sure positive)
    xi <- log(xi[d])
    # TODO: #4 Check if removing the case where D = 1 has any issues
    posts <- J$vectors[, d] * (sqrt(qchisq(1 - alpha, D) / abs(J$values[d]))) # Our current best Q, and Cholesky decomp (R^1/2) (although without the scaling xi)
    # These are the directions to go
    # Don't I need Qsigma^-1/2 Qt? I am very confused at why we have t(J$vectors). Shouldn't we have just J$vectors as our Q matrix?
    # Think this was a mistake in the original code, although the outputs are practically identical(?)

    # Define our updating function. Cannot just do a newton rhapson to update our xi.
    # TODO: #5 Provide reference of stochastic algorithm
    it <- 1
    repeat {
      # print(it)
      posts.xi <- as.vector(posts * exp(xi / 2)) # Xi scales singular values (scalar for each directions). Again, we are going to evaluate this direction * scaling to see how far off.
      # exp parameterization lets us avoid negative xi (so when we do sqrt(xi) we don't get imaginary)
      # removed an if else that dealt with if D == 1
      g.xi <- maxpl(posts.xi) - alpha
      # print(g.xi)
      if (all(abs(g.xi) <= tol) || (it >= max.it)) {
        break
      } else {
        xi <- xi + w(it) * g.xi
        it <- it + 1
      }
    }
    xi[d] <- exp(xi)
  }
  # Return the exponential version
  return(xi)
}

# Function that is called if doing the elliptical approximation
#' @export
glim_inner_prob_approx_samples <- function(X, y, family = "gaussian", mle_val, m, parallel) {
  print("glim_inner_prob")
  B <- 100
  AA <- seq(0.001, 0.999, length = B)
  if (family == "gaussian" || family == "normal") {
    res <- lm(y ~ X - 1)
    # This is the observed variability for this link function
    J <- crossprod(X, X)
  } else if (family == "binomial") {
    res <- glm(y ~ X - 1, family = "binomial")
    p_i <- res$fitted.values
    J <- crossprod(X, diag(p_i * (1 - p_i)) %*% X)
  } else if (family == "gamma") {
    # canonical link for gamma family (1/mu) is incredibly numerically unstable. If Xb is ever close to zero during the process, we get infinities. Additionally, if xb is ever negative, then we are saying that the mean of a gamma is negative.
    # Note that the weights are one here.
    res <- glm(y ~ X - 1, family = Gamma(link = "log"))
    J <- crossprod(X, X)
  } else if (family == "inverse.gaussian") {
    res <- glm(y ~ X - 1, family = inverse.gaussian(link = "1/mu^2"))
    # default link for the inverse gaussian in glm is not the canonical parameter (-1/2mu^2), but rather 1/mu. Additionally, note that the link we are using here is not a canonical link. The constant gets absorbed in a lot of places, and what ends up changing is the scaling factor outside our gradient update.
    eta <- X %*% res$coefficients
    mu_i <- as.vector(sqrt(1 / eta))
    J <- crossprod(X, diag(mu_i^3) %*% X) # TODO #7 check on this calculation
  } else if (family == "poisson") {
    res <- glm(y ~ X - 1, family = poisson(link = "log"))
    lambda_i <- res$fitted.values
    J <- crossprod(X, diag(lambda_i) %*% X)
  }
  J <- (J + t(J)) / 2 # symmetrize to try to kill some rounding asymmetries -- Gemini's idea
  eJ <- eigen(J)
  # James' solution to needing to scale along a direction didn't work in my case
  # can't have zeros in the eigvalues because will not be invertible
  eJ$values[eJ$values < 1e-4] <- .000001
  mle_coefs <- res$coefficients

  pl <- function(z) {
    # If z is a vector, matrix(z, nrow = 1) makes it a 1 x p row matrix
    betas_matrix <- if (is.matrix(z)) z else matrix(z, nrow = 1)

    glim_raw(X, y, family, betas_matrix, mle_coefs, mle_val = mle_val, m, parallel)
  }

  i <- 0
  xi <- list()
  prev_xi <- rep(1, length(mle_coefs))
  # finding xi
  parallel <- FALSE
  for (a in AA) {
    print(a)
    i <- i + 1
    parallel <- FALSE
    # xi is our scaling
    xi[[i]] <- imvar(
      prev_xi,
      a,
      pl,
      mle = mle_coefs,
      J = eJ,
      parallel,
      tol = 1e-2,
      a = 5,
      b = 1,
      max.it = 20
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

    # Sample randomly on the boundary TODO #8 explain code
    rand_dir <- generate_unit_matrix(1, length(mle_coefs))
    spatial_dir <- eJ$vectors %*% (sqrt(1 / eJ$values) * rand_dir)

    samples[, i] <- mle_coefs +
      as.vector(sqrt(qchisq(1 - u, length(mle_coefs))) * lerped_xi * spatial_dir)
  }
  return(samples)
}

#' @export
glim <- function(X, y, family = "gaussian", betas, m = 1000, parallel = TRUE, approx = FALSE) {
  print("glim_called")
  if (family == "binomial" || family == "logistic") {
    ll_mle_original_data <- as.numeric(logLik(glm(y ~ X - 1, family = "binomial")))
    mle_coefs <- glm(y ~ X - 1, family = "binomial")
  } else if (family == "gamma") {
    # Don't want to use R's logLik() as it uses the pearson estimate of phi.
    # Note that the IRLS to maximize the log lik of beta doesn't rely on phi.
    mle_coefs <- glm(y ~ X - 1, family = Gamma(link = "log"))$coefficients
    eta <- X %*% mle_coefs
    ratio <- y / exp(eta)
    n <- length(y)
    ll_mle_original_data <- compute_gamma_ll(
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
      m,
      parallel = parallel
    ))
  }
  # Need to get dispersion into this bottom function
  if (approx == TRUE) {
    return(glim_inner_prob_approx_samples(
      X,
      y,
      family = family,
      mle_val = ll_mle_original_data,
      m = m,
      parallel = parallel
    ))
  }
}

# TODO #9 Dispersion in prob2poss
#' @export
prob2poss_logis <- function(X, y, samples, the_compared_theta) {
  # p <- 1/(1 + exp(-eta))
  eta <- X %*% the_compared_theta
  log_term <- log1p(exp(eta))
  ll_val <- y %*% eta - colSums(log_term)
  print(ll_val)

  eta_samps <- X %*% samples
  # p <- 1/(1 + exp(-eta_samps))
  log_term_samps <- log1p(exp(eta_samps))
  ll_val_samps <- y %*% eta_samps - colSums(log_term_samps)
  print(ll_val_samps)

  return(sapply(ll_val, function(x) sum(ll_val_samps < x)) / length(ll_val_samps))
}


#' @export
compute_gamma_ll_r <- function(y, eta, shape) {
  return(compute_gamma_ll(y, eta, shape))
}
