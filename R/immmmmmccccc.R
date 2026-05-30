#' @useDynLib IMMC, .registration = TRUE
#' @importFrom Rcpp sourceCpp
NULL
#
# Original code written by Joe Harrison (jrharr25@ncsu.edu), optimized in cpp by Gemini
# pkgbuild::compile_dll() validates the package directory differently. Rcpp might implement it's directory check differently
# After making changes, restart R, get in the package directory
#devtools::document()
#devtools::install()

#' @export
glim_raw <- function(
  X,
  y,
  family = "gaussian",
  betas,
  dispersions,
  mle_coefs,
  mle_val,
  m,
  parallel
) {
  if (is.data.frame(X)) {
    X <- model.matrix(X)
  }
  if (family == "binomial" || family == "logistic") {
    cpp_fit_glm <- glm_logis_pl_cpp
    dispersion <- 1 # Not needed, only doing this to
  } else if (family == "gamma") {
    cpp_fit_glm <- glm_gamma_pl_cpp
    dispersion <- pearson_estimate_dispersion_gamma(y, exp(X * mle_coefs), length(mle_coefs))
  } else if (family == "poisson") {
    cpp_fit_glm <- glm_poisson_pl_cpp
    dispersion <- 1
  } else if (family == "inverse-gaussian" || family == "inverse.gaussian") {
    cpp_fit_glm <- glm_invgauss_pl_cpp
    dispersion <- 1
  } else if (family == "normal" || family == "gaussian") {
    cpp_fit_glm <- glm_gaussian_pl_cpp
    dispersion <- 1
  } else {
    stop("Family not supported")
  }
  if (parallel == TRUE) {
    # if we don't have multiple beta that we want to evaluate it over, then we don't
    # want to allocate clusters
    if (is.vector(betas)) {
      result_vector <- as.matrix(cpp_fit_glm(
        X = X,
        y = y,
        beta_vals = betas,
        dispersion,
        mle_coefs,
        mle_val,
        m = m
      ))
    } else {
      num_cores <- parallel::detectCores() - 1
      cl <- parallel::makeCluster(num_cores)

      parallel::clusterCall(cl, function() library(IMMC))
      # Export the DATA to the workers
      parallel::clusterExport(
        cl,
        varlist = c("X", "y", "ll_mle_original_data", "m"),
        envir = environment()
      )

      result_vector <- pbapply::pbapply(
        betas,
        1,
        function(b) {
          cpp_fit_glm(
            X = X,
            y = y,
            beta_vals = as.numeric(b),
            dispersion,
            mle_coefs,
            mle_val,
            m = m
          )
        },
        cl = cl
      )
    }
    parallel::stopCluster(cl)
  } else {
    if (is.vector(betas)) {
      result_vector <- as.matrix(cpp_fit_glm(
        X = X,
        y = y,
        beta_vals = betas,
        dispersion,
        mle_coefs,
        mle_val = ll_mle_original_data,
        m = m
      ))
    } else {
      result_vector <- pbapply::pbapply(betas, 1, function(b) {
        cpp_fit_glm(
          X = X,
          y = y,
          beta_vals = as.numeric(b),
          dispersion,
          mle_coefs,
          mle_val = ll_mle_original_data,
          m = m
        )
      })
    }
  }
  return(result_vector)
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
imvar <- function(xi, alpha, pl, mle, J, parallel, tol = 1e-2, a = 1, b = 1, max.it = 25) {
  # tol <- 1e-2
  # a <- 5
  # b <- 1
  # max.it <- 20
  # xi <- -.425
  # mle <- mle_coefs
  # alpha <- .45
  # J <- eJ
  xi <- log(xi)
  D <- length(mle)
  # removed a case where D = 1
  posts <- as.vector(J$vectors %*% sqrt(qchisq(1 - alpha, D) / abs(J$values))) # Our current best Q, and Cholesky decomp (R^1/2) (although without the scaling xi)
  # These are the directions to go
  # Don't I need Qsigma^-1/2 Qt? I am very confused at why we have t(J$vectors). Shouldn't we have just J$vectors as our Q matrix?
  maxpl <- function(v, lambda) {
    max(c(pl(as.vector(mle) + v, lambda), pl(as.vector(mle) - v, lambda)))
  }
  w <- function(s) a / (1 + s)**b # our weighting function, dampens over time
  it <- 1
  repeat {
    posts.xi <- as.vector(posts * exp(xi / 2)) # Xi scales singular values (scalar for each directions). Again, we are going to evaluate this direction * scaling to see how far off.
    # exp parameterization lets us avoid negative xi (so when we do sqrt(xi) we don't get imaginary)
    # removed an if else that dealt with if D == 1
    g.xi <- maxpl(posts.xi, phi) - alpha
    if (all(abs(g.xi) <= tol) || (it >= max.it)) {
      break
    } else {
      xi <- xi + w(it) * g.xi
      it <- it + 1
    }
  }
  # Return the exponential version
  return(exp(xi))
}

#' @export
glim_inner_prob_approx_samples <- function(
  X,
  y,
  family = "gaussian",
  ll_mle_original_data,
  m,
  parallel
) {
  # X <- X_binary
  # y <- y_binary
  m <- 100
  pl <- function(z) {
    glim_raw(X, y, family, z, dispersions, mle_coefs, mle_val = ll_mle_original_data, m, parallel)
  }
  B <- 100
  AA <- seq(0.001, 0.999, length = B)
  # Need to find the fisher information for each parameterization TODO(?)
  if (family == "gaussian" || family == "normal") {
    res <- lm(y ~ X - 1)
    J <- crossprod(X, X)
  } else if (family == "binomial") {
    res <- glm(y ~ X - 1, family = "binomial")
    p_i <- res$fitted.values
    J <- crossprod(X, diag(p_i * (1 - p_i)) %*% X)
  } else if (family == "gamma") {
    # canonical link for gamma family is incredibly numerically unstable. If Xb is ever close to zero during the process, we get infinities. Additionally, if xb is ever negative, then we are saying that the mean of a gamma is negative.
    # Note that the weights are one here.
    res <- glm(y ~ X - 1, family = Gamma(link = "log"))
    J <- crossprod(X, X)
  } else if (family == "inverse.gaussian") {
    res <- glm(y ~ X - 1, family = inverse.gaussian(link = "1/mu^2"))
    # default link for the inverse gaussian in glm is not the canonical parameter (-1/2mu^2), but rather 1/mu. Additionally, note that the link we are using here is not a canonical link. The constant gets absorbed in a lot of places, and what ends up changing is the scaling factor outside our gradient update.
    eta <- X %*% res$coefficients
    mu_i <- as.vector(sqrt(1 / eta))
    J <- crossprod(X, diag(mu_i^3) %*% X) # check on this
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

  i <- 0
  xi <- rep(1, B)
  prev_xi <- 1
  # finding xi
  parallel <- FALSE
  for (a in AA) {
    i <- i + 1
    m <- 100
    parallel <- FALSE
    # imvar(log(1), .05, pl, mle_coefs, eJ, parallel, 1e-2, 5, 1, 20)
    xi[i] <- imvar(
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
    prev_xi <- xi[i]
  }
  U <- runif(m)
  lerped_xi <- -1
  samples <- matrix(nrow = length(mle_coefs), ncol = m)
  i <- 0
  for (u in U) {
    i <- i + 1
    if (u < min(AA)) {
      lerped_xi <- xi[1]
    } else if (u > max(AA)) {
      lerped_xi <- xi[B]
    } else {
      r <- sum(AA < u)
      w <- (u - AA[r]) / (AA[r + 1] - AA[r])
      lerped_xi <- (1 - w) * xi[r] + w * xi[r + 1]
    }

    rand_dir <- generate_unit_matrix(1, length(mle_coefs))
    spatial_dir <- eJ$vectors %*% (sqrt(1 / eJ$values) * rand_dir)

    samples[, i] <- mle_coefs +
      as.vector(sqrt(qchisq(1 - u, length(mle_coefs))) * lerped_xi * spatial_dir)
  }
  return(samples)
}

#' @export
glim <- function(X, y, family = "gaussian", betas, m = 1000, parallel = TRUE, approx = FALSE) {
  if (family == "binomial" || family == "logistic") {
    ll_mle_original_data <- as.numeric(logLik(glm(y ~ X - 1, family = "binomial")))
    mle_coefs <- glm(y ~ X - 1, family = "binomial")
  } else if (family == "gamma") {
    # Don't want to use R's logLik as it uses the pearson estimate of phi.
    # However, the IRLS used by R to maximize the log lik doesn't rely on phi.
    mle_coefs <- glm(y ~ X - 1, family = Gamma(link = "log"))$coefficients
    eta <- X %*% mle_coefs
    ratio <- y / exp(eta)
    n <- length(y)
    ll_mle_original_data <- compute_gamma_ll(
      y,
      eta,
      pearson_estimate_dispersion_gamma(y, exp(eta), length(mle_coefs)),
      n
    )

    # ll_mle_original_data <- as.numeric(logLik(glm(y ~ X - 1, family = Gamma(link = "log"))))
  } else if (family == "poisson") {
    ll_mle_original_data <- as.numeric(logLik(glm(y ~ X - 1, family = poisson(link = "log"))))
    mle_coefs <- glm(y ~ X - 1, family = poisson(link = "log"))$coefficients
  } else if (family == "inverse-gaussian") {
    ll_mle_original_data <- as.numeric(logLik(glm(
      y ~ X - 1,
      family = inverse.gaussian(link = "1/mu^2")
    )))
    mle_coefs <- glm(y ~ X - 1, family = inverse.gaussian(link = "1/mu^2"))$coefficients
  } else if (family == "normal" || "gaussian") {
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
      betas,
      mle_coefs,
      mle_val = ll_mle_original_data,
      m,
      parallel = parallel
    ))
  }
  if (approx == TRUE) {
    return(glim_inner_prob_approx_samples(
      X,
      y,
      family = family,
      mle_coefs,
      mle_val = ll_mle_original_data,
      m,
      parallel = parallel
    ))
  }
}

#' @export
prob2poss_logis <- function(X, y, samples, the_compared_theta) {
  # p <- 1/(1 + exp(-eta))
  eta <- X %*% the_compared_theta
  log_term <- log1p(exp(eta))
  ll_val <- y %*% eta - colSums(log_term)

  eta_samps <- X %*% samples
  # p <- 1/(1 + exp(-eta_samps))
  log_term_samps <- log1p(exp(eta_samps))
  ll_val_samps <- y %*% eta_samps - colSums(log_term_samps)

  return(sapply(ll_val, function(x) sum(ll_val_samps < x)) / length(ll_val_samps))
}
