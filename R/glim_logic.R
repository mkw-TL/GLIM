#' @useDynLib GLIM, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom progress progress_bar
#' @importFrom parallel detectCores
#' @importFrom stats logLik glm Gamma poisson inverse.gaussian lm coef runif qchisq model.matrix rnorm model.offset model.response binomial nobs vcov
#' @importFrom utils head
#' @importFrom graphics par plot.default abline grid mtext
#' @importFrom methods is
#' @import RcppProgress
NULL


# Original code written by Joe Harrison (joe.harrison.va@gmail.com), translated to C++ by Gemini
#
# Linear Model formula parsing is copied from stats::lm in R.
# Copyright (C) The R Core Team
#
# Distributed under the terms of the GNU General Public License,
# version 3.

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
  if (parallel & requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    RhpcBLASctl::blas_set_num_threads(original_blas_threads)
  }
  return(output)
}


#' Generates a grid of parameter values (cut by imvar)
#'
#' Aligned in parameter grid, rather than eigen-vector space for easier marginalization
#' @noRd
generate_grid <- function(
  X,
  y,
  family,
  mle_coefs,
  eigen_vecs,
  eigen_vals,
  dispersion,
  ll_mle_original_data,
  n_grid_evals = 25,
  m = 500
) {
  initial_xi <- rep(1, length(mle_coefs))

  alpha_target <- 0.001

  imvar_xi <- imvar(
    X,
    y,
    initial_xi,
    family,
    alpha_target,
    mle_coefs,
    ll_mle_original_data,
    eigen_vecs,
    eigen_vals,
    dispersion,
    tol = .001, # the tolerance value
    a_val = .5, # found through a grid search
    b_val = 1, # found through a grid search
    max_it = 25,
    parallel = FALSE,
    m = m # This is our m value
  )
  q_val <- qchisq(1 - alpha_target, df = length(mle_coefs)) # This is the first guess at our quantile distance, which we need to modify slightly by xi.
  base_scale <- sqrt(dispersion * q_val * (1 / eigen_vals))
  semi_axes <- base_scale * sqrt(imvar_xi)
  transformed_mat <- eigen_vecs %*% diag(as.vector(semi_axes)) # Takes the principle axes (through diag) and scales them (by semi_axes), then rotate to eigen_vector span

  # To make the beta coordinate the largest, we have our transformation matrix (times u, where u is a unit sphere).
  # Then, for a particular coordinate, we have b1 = row_1 * u, where this is maximized if u is in the direction of row_1.
  H <- sqrt(rowSums(transformed_mat^2))

  # Generate the Axis-Aligned Grid using the bounding box widths
  beta <- list()
  n_left <- floor((n_grid_evals - 1) / 2)
  n_right <- n_grid_evals - 1 - n_left # if n_grid_evals is even, right side gets one more data point
  for (i in 1:length(mle_coefs)) {
    beta[[i]] <- c(
      # Allocate half the points to the left, half to the right
      seq(mle_coefs[i] - H[i], mle_coefs[i], length.out = n_left + 1)[-(n_left + 1)], # Left side excluding MLE. Need to do length.out + 1 because we're going to get rid of one
      mle_coefs[i], # Exact MLE
      seq(mle_coefs[i], mle_coefs[i] + H[i], length.out = n_right + 1)[-1] # Right side excluding MLE
    )
  }

  betas <- as.matrix(expand.grid(beta))
  colnames(betas) <- colnames(X)
  # Written by Gemini
  shifted_points <- sweep(betas, 2, mle_coefs, FUN = "-") # 2 means that it applies along the columns. Betas - mle_coefs for each column
  rotated_points <- shifted_points %*% eigen_vecs
  scaled_sq_points <- sweep(rotated_points^2, 2, semi_axes^2, FUN = "/")
  inside_indices <- rowSums(scaled_sq_points) <= 1
  # Check if they are inside the unit sphere (i.e. inside the ellipse)
  if (!any(inside_indices)) {
    stop(
      "GLIM error: no grid points fell inside the confidence ellipse. ",
      "Try increasing 'n_grid_evals' or check that the model fit ",
      "(mle_coefs / eigen decomposition) is not degenerate.",
      call. = FALSE
    )
  }
  # Check if they are inside the unit sphere (i.e. inside the ellipse)
  # From betas, select all the columns (proposed betas) that are inside
  points_inside <- betas[inside_indices, ]
  return(points_inside)
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
  parallel,
  eJ,
  dispersion,
  a_val,
  b_val,
  max_it
) {
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
  B <- 100
  AA <- seq(0.001, 0.999, length.out = B)

  i <- 0
  xi <- list()
  prev_xi <- rep(1, length(mle_coefs))
  pb <- progress::progress_bar$new(
    total = length(AA),
    format = "[:bar] :percent eta :eta",
    show_after = 0,
    force = TRUE
  )
  for (alpha in AA) {
    pb$tick()
    i <- i + 1
    xi[[i]] <- imvar(
      X,
      y,
      prev_xi,
      family,
      alpha,
      mle = mle_coefs,
      mle_val,
      as.matrix(eJ$vectors),
      as.vector(eJ$values),
      dispersion,
      tol = .01,
      a_val = a_val,
      b_val = b_val,
      max_it = max_it,
      parallel = TRUE
    )
    prev_xi <- xi[[i]]
  }

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
      as.vector(
        sqrt(qchisq(1 - u, length(mle_coefs))) * sqrt(lerped_xi) * spatial_dir * sqrt(dispersion)
      )
  }
  return(samples)
}

#' Generalized Linear Inferential Models (GLIM) Main Function
#'
#' Main wrapper function to fit a GLIM model. Users can pass in their own grid of beta values to evaluate by setting betas = MATRIX.
#' A typical use would be MATRIX = as.matrix(expand.grid(beta_0_seq, beta_1_seq))
#'
#' If using the elliptical approximation, can tweak the Robbins-Monroe algorithm by changing a_val, b_val, and max_it.
#'
#' @param formula Formula object to be interpreted
#' @param data Dataframe (if using)
#' @param radial Bool indicating whether the radial method is to be used.
#' @param tol Double indicating the level of precision the approximation should get to
#' @param family String denoting the exponential family. Choices are `"gaussian"`, `"binomial"`, `"gamma"`, `"poisson"`, `"inverse.gaussian"`.
#' @param m Number of samples/evaluations to perform (default `1000`).
#' @param approx Logical indicating whether to use the elliptical approximation (default `FALSE`).
#' @param ... Other arguments include a_val, b_val, max_it, betas, n_grid_evals, tol, and num_samps.
#' @return If using an elliptical approximation, returns samples of parameter values. Else, returns a list containing the matrix of betas upon which it was evaluated, and a vector of the corresponding possibilities.
#' @examples
#' \dontrun{
#' set.seed(1)
#' n <- 100
#' x <- rnorm(n)
#' y <- 2 + 1.5 * x + rnorm(n)
#' dat <- data.frame(x = x, y = y)
#'
#' # Grid-based evaluation (default)
#' fit <- glim(y ~ x, data = dat, family = "gaussian", m = 500)
#' get_CI(fit, alpha = .05)
#' plot(fit)
#' print(fit)
#'
#' # Elliptical approximation instead of a grid search
#' fit_approx <- glim(y ~ x, data = dat, family = "gaussian", approx = TRUE, m = 500)
#'
#' # Passing a custom grid of betas to evaluate via '...'
#' beta0_seq <- seq(0, 4, length.out = 10)
#' beta1_seq <- seq(0, 3, length.out = 10)
#' custom_betas <- as.matrix(expand.grid(beta0_seq, beta1_seq))
#' fit_custom <- glim(y ~ x, data = dat, family = "gaussian", betas = custom_betas)
#' }
#' @export
glim <- function(
  formula,
  data = NULL,
  family = "gaussian",
  m = 1000,
  approx = FALSE,
  radial = FALSE,
  tol = 1e-2,
  ...
) {
  mf <- match.call(expand.dots = FALSE) # Captures the whole input
  match <- match(
    # filters out any non-used arguments
    c("formula", "data"),
    names(mf),
    0L
  )
  mf <- mf[c(1L, match)] # from match.call, take elements 1 (function name), 2 (match with formula), 3 (match with data)
  mf$drop.unused.levels <- TRUE
  mf[[1L]] <- quote(stats::model.frame)
  env <- environment(formula)
  if (is.null(env)) {
    env <- parent.frame()
  }
  mf <- eval(mf, env)
  mt <- attr(mf, "terms") # figure out the variables to expand out in the model.matrix. Contains further attributes of factors, etc
  offset <- model.offset(mf) # would need to put into glim()
  X <- model.matrix(mt, mf, NULL) # model.matrix() creates the dummy columns. NULL is the contrasts (which could be set as default setting within a useR's R session, but I don't want to incorporate this)
  if (any(is.na(X))) {
    stop("NA values detected in the X matrix.")
  }
  # If more than one column is all 1s, drop the automatic R intercept
  # Gets rid of auto-prepended intercept if there already exists one. X == 1 is a boolean vector, so colsums is a good way of doing this
  if (sum(colSums(X == 1) == nrow(X)) > 1 && "(Intercept)" %in% colnames(X)) {
    X <- X[, colnames(X) != "(Intercept)", drop = FALSE]
  }
  y <- model.response(mf, "any")
  if (any(is.na(y))) {
    stop("NA values detected in the response.")
  }
  args <- list(...)
  if ("a_val" %in% names(args)) {
    if (is.numeric(args$a_val) & length(args$a_val) == 1 & args$a_val > 0) {
      a_val <- args$a_val
    } else {
      stop("Input Error: a_val must be a positive number")
    }
  } else {
    a_val <- 1.5
  }
  if ("b_val" %in% names(args)) {
    if (is.numeric(args$b_val) & length(args$b_val) == 1 & args$b_val > 0) {
      b_val <- args$b_val
    } else {
      stop("Input Error: b_val must be a positive number")
    }
    b_val <- args$b_val
  } else {
    b_val <- .65
  }
  parallel <- TRUE
  if ("parallel" %in% names(args)) {
    parallel <- args$parallel
  }
  betas <- NULL
  if (approx == FALSE) {
    if ("betas" %in% names(args)) {
      if (is.vector(args$betas)) {
        betas <- matrix(args$betas, nrow = 1)
      } else if (!(is.matrix(args$betas) & typeof(args$betas) == "double")) {
        stop("Grid of betas not in the correct form")
      } else {
        betas <- args$betas
      }
    }
  }
  if ("n_grid_evals" %in% names(args)) {
    if (
      is.numeric(args$n_grid_evals) &
        (args$n_grid_evals %% 1 == 0) &
        length(as.integer(args$n_grid_evals)) == 1 &
        as.integer(args$n_grid_evals > 1)
    ) {
      n_grid_evals <- as.integer(args$n_grid_evals)
    } else {
      stop("Input Error: n_grid_evals is not a positive integer")
    }
  } else {
    n_grid_evals <- 25 # Is this a good number?
  }
  if (is.numeric(as.numeric(tol)) & length(as.numeric(tol)) == 1 & as.numeric(tol) > 0) {
    tol <- tol
  } else {
    stop("Input Error: tol is not currently valid")
  }

  if ("max_it" %in% names(args)) {
    if (
      is.numeric(args$max_it) & args$max_it %% 1 == 0 & length(args$max_it) == 1 & args$max_it > 0
    ) {
      max_it <- args$max_it
    } else {
      stop("Input Error: max_it must be a positive integer")
    }
  } else {
    max_it <- 25
  }
  if ("num_samps" %in% names(args)) {
    if (is.integer(as.integer(args$num_samps)) & length(args$num_samps) == 1 & args$num_samps > 0) {
      num_samps <- args$num_samps
    } else {
      stop("Input Error: num_samps must be a positive integer")
    }
  } else {
    num_samps <- 2000
  }
  if (!is.null(names(args))) {
    if (
      !(all(
        names(args) %in% c("a_val", "b_val", "max_it", "betas", "n_grid_evals", "tol", "num_samps")
      ))
    ) {
      stop("Incorrect names of additional arguments passed")
    }
  }

  if (is.vector(y)) {
    if (nrow(X) != length(y)) {
      stop(
        "Input Error: The number of rows in the design matrix X must equal the length of the response vector y."
      )
    }
  } else {
    if (nrow(X) != nrow(y)) {
      stop(
        "Input Error: The number of rows in the design matrix X must equal the length of the response vector y."
      )
    }
  }

  if (ncol(X) > nrow(X)) {
    warning(
      "Input Warning: The number of predictors (p) is greater than the number of observations (n)."
    )
  }

  # This allows users to type family = "pois" and it will auto-match to "poisson"
  family <- match.arg(
    family,
    choices = c("gaussian", "poisson", "gamma", "binomial", "inverse.gaussian")
  )

  if (!is.logical(parallel) || length(parallel) != 1 || is.na(parallel)) {
    stop("Input Error: 'parallel' must be either TRUE or FALSE.")
  }

  if (!is.logical(approx) || length(approx) != 1 || is.na(approx)) {
    stop("Input Error: 'approx' must be either TRUE or FALSE.")
  }

  # Inside glim_raw or the main exported wrapper
  if (family == "poisson") {
    if (any(y < 0) || !all(y == floor(y))) {
      stop("Input Error: For Poisson family, 'y' must contain only non-negative integers.")
    }
  } else if (family %in% c("gamma", "inverse.gaussian")) {
    if (any(y <= 0)) {
      stop(sprintf(
        "Input Error: For %s family, all values in 'y' must be strictly positive (y > 0).",
        family
      ))
    }
  }

  if (!is.numeric(m) || length(m) != 1 || m <= 0 || m != floor(m)) {
    stop("Input Error: 'm' (number of samples) must be a single positive integer.")
  }

  if ((!is.numeric(tol) || length(tol) != 1 || tol <= 0)) {
    stop("Input Error: 'tol' must be a strictly positive numeric scalar.")
  }

  J <- NULL
  if (is.data.frame(X)) {
    X <- as.matrix(X)
  }
  # Got rid of intercept logic, since covered.
  ## Binomial setup
  if (family == "binomial" || family == "logistic") {
    if (is.data.frame(X)) {
      X <- as.matrix(X)
    }
    if (is.vector(y) && all(y == 0 | y == 1)) {
      if (length(y) != nrow(X)) {
        stop("y and X lengths differ")
      }
    } else if (is.matrix(y) && ncol(y) == 2) {
      # Two-column integer matrix (Successes / Failures)
      successes <- y[, 1]
      failures <- y[, 2]

      if (ncol(X) == 1) {
        idx_success <- rep(1:nrow(X), times = successes)
        idx_fail <- rep(1:nrow(X), times = failures)
      } else {
        idx_success <- rep(1:nrow(X), times = successes)
        idx_fail <- rep(1:nrow(X), times = failures)
      }
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
    # We are using (-1) so that R knows the X matrix we are using; we don't want to append any extra intercepts
    fit <- glm(y ~ X - 1, family = binomial)
    eps <- 10 * .Machine$double.eps
    if (any(fit$fitted.values > (1 - eps)) || any(fit$fitted.values < eps)) {
      stop("Data is completely seperable. MLE does not exist")
    }
    mle_coefs <- fit$coefficients
    vcov <- vcov(fit)
    p_i <- 1 / (1 + exp(-X %*% mle_coefs))
    dispersion <- 1
    if (is.null(betas)) {
      J <- crossprod(X, X * as.vector((p_i * (1 - p_i))))
    }
    p2p_function <- prob2poss_logis

    ll_mle_original_data <- compute_logistic_ll(X, y, mle_coefs)

    ## Gamma setup
  } else if (family == "gamma") {
    fit <- glm(y ~ X - 1, family = Gamma(link = "log"))
    mle_coefs <- coef(fit)
    vcov <- vcov(fit)
    eta <- X %*% mle_coefs
    dispersion <- mle_estimate_dispersion_gamma(y, exp(X %*% mle_coefs), length(mle_coefs))
    ll_mle_original_data <- compute_gamma_ll_r(y, eta, 1 / dispersion)
    if (is.null(betas)) {
      J <- crossprod(X, X)
    }
    p2p_function <- prob2poss_gamma

    ## Poisson setup
  } else if (family == "poisson") {
    pois_fit <- glm(y ~ X - 1, family = poisson(link = "log"))
    if (!pois_fit$converged) {
      stop("Poisson GLM fails to converge")
    }
    vcov <- vcov(pois_fit)
    ll_mle_original_data <- as.numeric(logLik(pois_fit))
    mle_coefs <- coef(pois_fit)
    eta <- X %*% mle_coefs
    lambda_i <- exp(eta)
    dispersion <- 1
    if (is.null(betas)) {
      J <- crossprod(X, X * as.vector(lambda_i))
    }
    p2p_function <- prob2poss_poisson

    ## Inverse Gaussian setup
  } else if (family == "inverse.gaussian") {
    inv_gaus_fit <- glm(y ~ X - 1, family = inverse.gaussian(link = "1/mu^2"))
    vcov <- vcov(inv_gaus_fit)
    ll_mle_original_data <- as.numeric(logLik(inv_gaus_fit))
    mle_coefs <- coef(inv_gaus_fit)
    eta <- X %*% mle_coefs
    mu_i <- as.vector(sqrt(1 / eta))
    dispersion <- mle_estimate_dispersion_inv_gauss(y, mean(y))
    if (is.null(betas)) {
      J <- crossprod(X, X * (mu_i^3) / 4)
    }
    p2p_function <- prob2poss_invgauss

    ## Gaussian Setup
  } else if (family == "normal" || family == "gaussian") {
    fit <- lm(y ~ X - 1)
    ll_mle_original_data <- as.numeric(logLik(fit))
    mle_coefs <- coef(fit)
    vcov <- vcov(fit)
    dispersion <- est_dispersion_normal(y, X %*% mle_coefs, length(mle_coefs))
    if (is.null(betas)) {
      J <- crossprod(X, X)
    }
    p2p_function <- prob2poss_gaussian
  } else {
    stop("Family not supported")
  }
  if (!is.null(J)) {
    J <- (J + t(J)) / 2 # symmetrize to try to kill some rounding asymmetries
    eJ <- eigen(J)

    eJ$values[eJ$values < 1e-4] <- .000001
  }
  if (!is.null(mle_coefs)) {
    if (length(mle_coefs) != ncol(X)) {
      stop(
        "Input Error: The length of 'mle_coefs' must exactly match the number of columns (predictors) in X."
      )
    }
  }
  if (!missing(dispersion) && (!is.numeric(dispersion) || dispersion <= 0)) {
    stop("Internal error: 'dispersion' must be a strictly positive numeric value.")
  }

  if (approx == FALSE & radial == FALSE) {
    if (is.null(betas)) {
      message("generating_grid_of_betas")
      betas <- generate_grid(
        X,
        y,
        family,
        mle_coefs,
        eigen_vecs = eJ$vectors,
        eigen_vals = eJ$values,
        dispersion,
        ll_mle_original_data,
        n_grid_evals = n_grid_evals,
        m = m
      )
      colnames(betas) <- colnames(X)
    }
    message(cat("Our MLE is: ", mle_coefs))
    obj_to_return <- list(
      possibilities = glim_raw(
        X,
        y,
        family = family,
        betas,
        mle_coefs,
        mle_val = ll_mle_original_data,
        m = m,
        parallel = parallel,
        approx = approx
      ),
      samples = NULL,
      betas_evaluated = betas,
      family = family,
      X = X,
      y = y,
      mle_coefs = mle_coefs,
      logLik = ll_mle_original_data,
      vcov = vcov
    )
    class(obj_to_return) <- "glim_object"
    return(obj_to_return)
  }

  if ((approx == TRUE) & (radial == FALSE)) {
    samples <- glim_inner_prob_approx_samples(
      X = X,
      y = y,
      family = family,
      mle_coefs = mle_coefs,
      mle_val = ll_mle_original_data,
      m = m,
      parallel = parallel,
      eJ = eJ,
      dispersion = dispersion,
      a_val = a_val,
      b_val = b_val,
      max_it = max_it
    )
    betas <- generate_grid(
      X,
      y,
      family,
      mle_coefs,
      eJ$vectors,
      eJ$values,
      dispersion,
      ll_mle_original_data,
      n_grid_evals,
      m
    )
    poss <- p2p_function(X, y, samples, t(betas))
    colnames(betas) <- colnames(X)
    obj_to_return <- list(
      possibilities = poss,
      samples = samples,
      betas_evaluated = betas,
      family = family,
      X = X,
      y = y,
      mle_coefs = mle_coefs,
      logLik = ll_mle_original_data,
      vcov = vcov
    )
    class(obj_to_return) <- "glim_object"
    return(obj_to_return)
  }

  if (radial == TRUE & approx == TRUE) {
    stop("Radial and Elliptical approximation methods cannot both be used")
  }
  if (radial == TRUE & approx == FALSE) {
    samples <- radial(
      num_samps,
      X,
      y,
      eJ,
      mle_coefs,
      ll_mle_original_data,
      family,
      dispersion,
      m,
      tol,
      max_it,
      a_val,
      b_val,
      n_grid_evals = n_grid_evals
    )
    beta_seq_list <- list()
    for (i in 1:nrow(samples)) {
      beta_seq_list[[i]] <- seq(min(samples[i, ]), max(samples[i, ]), length.out = n_grid_evals)
    }
    beta_radial_grid <- as.matrix(expand.grid(beta_seq_list))
    poss <- p2p_function(X, y, samples, t(beta_radial_grid))
    colnames(beta_radial_grid) <- colnames(X)
    obj_to_return <- list(
      possibilities = poss,
      samples = samples,
      betas = beta_radial_grid,
      family = family,
      X = X,
      y = y,
      mle_coefs = mle_coefs,
      logLik = ll_mle_original_data,
      vcov = vcov
    )
    class(obj_to_return) <- "glim_object"
    return(obj_to_return)
  }
}

#' Probability to Possibility Mapping for Logistic Regression
#'
#' @param X Predictor matrix.
#' @param y Response vector.
#' @param samples Matrix of simulated sample coefficients.
#' @param the_compared_theta The theta values to compare against.
#' @return A vector of mapped possibility values.
#' @examples
#' \dontrun{
#' set.seed(1)
#' n <- 100
#' X <- cbind(intercept = 1, x = rnorm(n))
#' y <- rbinom(n, 1, plogis(X %*% c(0, 1.5)))
#'
#' # Simulated candidate coefficient vectors (one per column)
#' samples <- cbind(c(0, 1.2), c(0.1, 1.6), c(-0.1, 1.4))
#'
#' # Theta value(s) to evaluate against the simulated samples
#' the_compared_theta <- c(0, 1.5)
#'
#' prob2poss_logis(X, y, samples, the_compared_theta)
#' }
#' @export
prob2poss_logis <- function(X, y, samples, the_compared_theta) {
  if (is.data.frame(X)) {
    X <- as.matrix(X)
  }
  if (is.vector(y) && all(y == 0 | y == 1)) {
    if (length(y) != nrow(X)) {
      stop("y and X lengths differ")
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
  eta <- X %*% the_compared_theta
  log_term <- pmax(eta, 0) + log1p(exp(-abs(eta)))
  ll_val <- y %*% eta - colSums(log_term)

  eta_samps <- X %*% samples
  log_term_samps <- pmax(eta_samps, 0) + log1p(exp(-abs(eta_samps)))
  ll_val_samps <- as.vector(y %*% eta_samps) - colSums(log_term_samps)

  return(sapply(ll_val, function(x) sum(ll_val_samps <= x)) / length(ll_val_samps))
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
#' @examples
#' \dontrun{
#' set.seed(1)
#' n <- 100
#' X <- cbind(intercept = 1, x = rnorm(n, sd = .2))
#' y <- rgamma(n, shape = 5, rate = 5 / exp(X %*% c(1, .5)))
#'
#' # Simulated candidate coefficient vectors (one per column)
#' samples <- cbind(c(0.9, 0.4), c(1.1, 0.6), c(1.0, 0.5))
#'
#' # Theta values to evaluate against the simulated samples (must not be a scalar)
#' the_compared_theta <- cbind(c(1, .5))
#'
#' prob2poss_gamma(X, y, samples, the_compared_theta)
#' }
#' @export
prob2poss_gamma <- function(X, y, samples, the_compared_theta) {
  if (is.data.frame(X)) {
    X <- as.matrix(X)
  }
  eta <- X %*% samples
  initial_coefs <- coef(lm(log(y) ~ X - 1))
  mle_coefs <- fit_gamma_log_cpp(X, t(X) %*% X, y, initial_coefs, FALSE)
  est_shape <- 1 / mle_estimate_dispersion_gamma(y, exp(X %*% mle_coefs), length(mle_coefs))

  ll_val_samps <- as.vector(compute_gamma_ll_r(y, eta, shape = est_shape))
  ll_val <- as.vector(compute_gamma_ll_r(y, X %*% the_compared_theta, shape = est_shape))
  sapply(ll_val, function(x) sum(ll_val_samps <= x) / length(ll_val_samps))
}

#' Compute Gamma Log-Likelihood
#' @noRd
compute_gamma_ll_r <- function(y, eta, shape) {
  if (is.vector(eta)) {
    return(compute_gamma_ll(y, eta, shape))
  } else {
    return(compute_gamma_ll_mat(y, eta, shape))
  }
}

#' Compute Poisson Log-Likelihood
#' @noRd
compute_poisson_ll_r <- function(y, eta) {
  if (is.vector(eta)) {
    return(compute_poisson_ll(eta, y))
  } else {
    return(compute_poisson_ll_mat(eta, y))
  }
}

#' Compute Gaussian Log-Likelihood
#' @noRd
compute_gaussian_ll_r <- function(y, mu, sigma) {
  if (is.vector(mu)) {
    return(compute_gaussian_ll(y, mu, sigma))
  } else {
    return(compute_gaussian_ll_mat(y, mu, sigma))
  }
}

#' Compute inverse gaussian Log-Likelihood
#' @noRd
compute_invgauss_ll_r <- function(y, mu, gamma_val) {
  if (is.vector(mu)) {
    return(compute_invgauss_ll(y, mu, gamma_val))
  } else {
    return(compute_invgauss_ll_mat(y, mu, gamma_val))
  }
}

#' Compute logistic Log-Likelihood
#' @noRd
compute_logistic_ll_r <- function(X, y, beta_vals) {
  if (is.vector(beta_vals)) {
    return(compute_logistic_ll(X, y, beta_vals))
  } else {
    return(compute_logistic_ll_mat(X, y, beta_vals))
  }
}


#' Probability to Possibility Mapping for Gaussian Regression
#'
#' @param X Predictor matrix.
#' @param y Response vector.
#' @param samples Matrix of simulated sample coefficients.
#' @param the_compared_theta The theta values to compare against.
#' @return A vector of mapped possibility values.
#' @examples
#' \dontrun{
#' set.seed(1)
#' n <- 100
#' X <- cbind(intercept = 1, x = rnorm(n))
#' y <- 2 + 1.5 * X[, "x"] + rnorm(n)
#'
#' # Simulated candidate coefficient vectors (one per column)
#' samples <- cbind(c(1.8, 1.4), c(2.1, 1.6), c(2.0, 1.5))
#'
#' # Theta value(s) to evaluate against the simulated samples
#' the_compared_theta <- c(2, 1.5)
#'
#' prob2poss_gaussian(X, y, samples, the_compared_theta)
#' }
#' @export
prob2poss_gaussian <- function(X, y, samples, the_compared_theta) {
  if (is.data.frame(X) || is.vector(X)) {
    X <- as.matrix(X)
  }

  eta <- X %*% samples

  # Assuming identity link for Gaussian by default
  mle_coefs <- fit_gaussian_cpp(X, y)
  est_sd <- est_dispersion_normal(y, X %*% mle_coefs, ncol(X)) # mle
  ll_val_samps <- as.vector(compute_gaussian_ll_r(y, eta, sigma = est_sd))
  ll_val <- as.vector(compute_gaussian_ll_r(y, X %*% the_compared_theta, sigma = est_sd))

  sapply(ll_val, function(x) sum(ll_val_samps <= x) / length(ll_val_samps))
}


#' Probability to Possibility Mapping for Inverse Gaussian Regression
#'
#' Will throw an error if \code{the_compared_theta} is a scalar.
#'
#' @param X Predictor matrix.
#' @param y Response vector.
#' @param samples Matrix of simulated sample coefficients.
#' @param the_compared_theta The theta values to compare against.
#' @return A vector of mapped possibility values.
#' @examples
#' \dontrun{
#' set.seed(1)
#' n <- 100
#' X <- cbind(intercept = 1, x = rnorm(n, sd = .2))
#' y <- statmod::rinvgauss(n, mean = exp(X %*% c(0.7, 0.3)), dispersion = 1)
#'
#' # Simulated candidate coefficient vectors (one per column)
#' samples <- cbind(c(0.6, 0.2), c(0.8, 0.4), c(0.7, 0.3))
#'
#' # Theta values to evaluate against the simulated samples (must not be a scalar)
#' the_compared_theta <- cbind(c(.7, .3))
#'
#' prob2poss_invgauss(X, y, samples, the_compared_theta)
#' }
#' @export
prob2poss_invgauss <- function(X, y, samples, the_compared_theta) {
  if (is.data.frame(X)) {
    X <- as.matrix(X)
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

  sapply(ll_val, function(x) sum(ll_val_samps <= x) / length(ll_val_samps))
}

#' radial C++ Bridge Function
#'
#' Helper function acting as a bridge to underlying C++ routines for sample generation.
#'
#' @param num_samps Number of samples to generate.
#' @param eJ Eigen decomposition object.
#' @param X Predictor matrix.
#' @param y Response vector.
#' @param mle_coefs Maximum likelihood estimates for coefficients.
#' @param family String denoting the exponential family.
#' @param dispersion The dispersion parameter.
#' @param ll_mle_original_data Needed for imvar
#' @param m Parameter `m` defining scaling or sampling limits.
#' @param tol Tolerance level for convergence criteria.
#' @param max_it Maximum number of iterations the stochastic algorithm runs for for each grid value.
#' @param a_val Hyperparameter `a`, the step size. Small values (1-2) are typically what this is set to
#' @param b_val Hyperparameter `b`, the step-size decay. Should be between [.5 and 1]
#' @param n_grid_evals Resolution of grid
#' @return A matrix of output samples evaluated by the C++ backend.
#' @noRd
radial <- function(
  num_samps,
  X,
  y,
  eJ,
  mle_coefs,
  ll_mle_original_data,
  family,
  dispersion,
  m,
  tol,
  max_it,
  a_val,
  b_val,
  n_grid_evals
) {
  output_samples <- radial_code(
    num_samps,
    X,
    y,
    mle_coefs,
    eJ$vectors,
    eJ$values,
    family,
    dispersion,
    m,
    tol,
    max_it,
    a_val,
    b_val
  )
  grid <- generate_grid(
    X = X,
    y = y,
    family = family,
    mle_coefs = mle_coefs,
    eigen_vecs = eJ$vectors,
    eigen_vals = eJ$values,
    dispersion = dispersion,
    ll_mle_original_data = ll_mle_original_data,
    n_grid_evals = n_grid_evals,
    m = m
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
#' @examples
#' \dontrun{
#' set.seed(1)
#' n <- 100
#' X <- cbind(intercept = 1, x = rnorm(n, sd = .2))
#' y <- rpois(n, lambda = exp(X %*% c(1, .5)))
#'
#' # Simulated candidate coefficient vectors (one per column)
#' samples <- cbind(c(0.9, 0.4), c(1.1, 0.6), c(1.0, 0.5))
#'
#' # Theta value(s) to evaluate against the simulated samples
#' the_compared_theta <- c(1, .5)
#'
#' prob2poss_poisson(X, y, samples, the_compared_theta)
#' }
#' @export
prob2poss_poisson <- function(X, y, samples, the_compared_theta) {
  if (is.data.frame(X)) {
    X <- as.matrix(X)
  }
  eta <- X %*% samples
  mle_coefs <- coef(glm(y ~ X - 1, family = poisson("log")))
  ll_val_samps <- as.vector(compute_poisson_ll_r(y, eta))
  ll_val <- as.vector(compute_poisson_ll_r(y, X %*% the_compared_theta))
  sapply(ll_val, function(x) sum(ll_val_samps <= x) / length(ll_val_samps))
}


#' Returns a confidence interval
#'
#' Requires a grid of betas. Note that a 95% confidence interval corresponds to alpha = .05
#'
#' @param alpha Note that a 95% confidence interval corresponds to alpha = .05
#' @param betas The grid of beta values which we are evaluating on
#' @param possibilities The vector of possibilities
#' @return A matrix of compatable beta values from the grid.
#' @examples
#' betas <- cbind(intercept = c(1.8, 1.9, 2.0, 2.1, 2.2), x = c(1.3, 1.4, 1.5, 1.6, 1.7))
#' possibilities <- c(0.02, 0.20, 1.00, 0.30, 0.03)
#'
#' # 95% confidence region (alpha = .05)
#' get_CI_manual(alpha = .05, betas = betas, possibilities = possibilities)
#' @export
get_CI_manual <- function(alpha, betas, possibilities) {
  return(betas[possibilities > alpha, ])
}

#' Returns a confidence interval
#'
#' Note that a 95% confidence interval corresponds to alpha = .05
#'
#' @param glim_object An object returned by glim()
#' @param ... Used to pass in the alpha level. Ex: 'alpha = .10'
#' @return A matrix of compatable beta values from the grid.
#' @examples
#' \dontrun{
#' fit <- glim(y ~ x, data = dat, family = "gaussian")
#'
#' # 95% confidence region (default)
#' get_CI(fit)
#'
#' # 90% confidence region
#' get_CI(fit, alpha = .10)
#' }
#' @export
get_CI <- function(glim_object, ...) {
  args <- list(...)
  alpha <- .05
  if ("alpha" %in% names(args)) {
    if (
      (is.numeric(args$alpha)) & (length(args$alpha == 1)) & (args$alpha <= 1) & (args$alpha >= 0)
    ) {
      alpha <- args$alpha
    } else {
      stop("Alpha cutoff invalid")
    }
  }

  if (!is(glim_object, "glim_object")) {
    stop("Object which was passed is not a direct result from glim()")
  }
  return(glim_object$betas[glim_object$possibilities > alpha, ])
}

#' Plot
#'
#' Plotting for glim objects
#'
#' @param x A 'glim_object' from 'glim()'
#' @param ... Used to pass in an alpha cut. Ex: 'alpha = .10'
#' @return Marginal plots for betas
#' @examples
#' \dontrun{
#' fit <- glim(y ~ x, data = dat, family = "gaussian")
#'
#' # Plot marginal profiled plausibility for each coefficient
#' plot(fit)
#'
#' # Overlay a reference line at a chosen alpha cutoff
#' plot(fit, alpha = .05)
#' }
#' @export
plot.glim_object <- function(x, ...) {
  betas <- x$betas_evaluated
  poss <- x$possibilities
  family <- x$family
  args <- list(...)
  alpha <- -1
  if ("alpha" %in% names(args)) {
    if (
      (is.numeric(args$alpha)) & (length(args$alpha == 1)) & (args$alpha <= 1) & (args$alpha >= 0)
    ) {
      alpha <- args$alpha
    } else {
      stop("Alpha cutoff invalid")
    }
  }

  num_predictors <- ncol(betas)
  grid_cols <- ceiling(sqrt(num_predictors))
  grid_rows <- ceiling(num_predictors / grid_cols)

  old_par <- par(no.readonly = TRUE)
  # Add 'oma' (outer margin area) to par(). c(bottom, left, top, right)
  # We add 3 lines of space to the top
  par(mfrow = c(grid_rows, grid_cols), mar = c(4, 4, 3, 1), oma = c(0, 0, 3, 0))

  on.exit(par(old_par)) # if any crashes, don't have the user's state altered

  # Where the marginalization happens
  for (col in 1:ncol(betas)) {
    max_plaus <- tapply(poss, betas[, col], max, na.rm = TRUE)

    # Filter out -Inf values from empty grid slices
    beta_vals <- as.numeric(names(max_plaus))
    valid <- is.finite(beta_vals) & is.finite(max_plaus)

    if (any(valid)) {
      plot.default(
        beta_vals[valid],
        max_plaus[valid],
        type = 'l',
        xlab = colnames(betas)[col],
        ylab = "Profiled Plausibility",
        ylim = c(0, 1)
      )
      if (alpha != -1) {
        abline(h = alpha, col = "red", lty = 3)
      }
      grid(nx = NULL, ny = NULL, col = "lightgrey", lty = "dashed")
    } else {
      # Draw an empty box if no data falls in this slice
      plot.default(1, type = 'n', axes = FALSE, xlab = "", ylab = "", main = "No Data")
    }
  }
  # Margin text. Ensuring family is correctly uppercased
  mtext(
    text = paste0(
      toupper(substr(family, 1, 1)),
      substr(family, 2, nchar(family)),
      " Possibility contours (marginalized)"
    ),
    side = 3, # 3 means top
    outer = TRUE, # Put it in the outer margin space
    line = -.3, # Distance from the edge of the plots
    cex = 1.3, # Font size multiplier (makes it larger)
    font = 2 # 2 means Bold
  )
}


# Note: Initially written by the author, Gemini helped with formatting of output

#' Print
#'
#' Printing for glim objects
#'
#' @param x A 'glim_object' from 'glim()'
#' @param ... Supports differing alpha levels. EX: alpha = .05
#' @return Print output
#' @examples
#' \dontrun{
#' fit <- glim(y ~ x, data = dat, family = "gaussian")
#'
#' # Print the 95% confidence and credible region (default)
#' print(fit)
#'
#' # Print at a different alpha level
#' print(fit, alpha = .10)
#' }
#' @export
print.glim_object <- function(x, ...) {
  args <- list(...)
  alpha <- .05
  if (length(args) == 1 & is.numeric(args$alpha)) {
    alpha <- args$alpha
  }
  betas <- x$betas_evaluated
  poss <- x$possibilities
  family <- x$family
  mle_coefs <- as.numeric(x$mle_coefs)

  cat(sprintf("%.0f%% (marginal) confidence and credible region:\n", (1 - alpha) * 100))
  cat("------------------------------------------------------------------------\n")

  # Initialize vectors to collect data for the table
  row_names <- colnames(betas)
  mle_vals <- numeric(ncol(betas))
  lower_bound <- numeric(ncol(betas))
  upper_bound <- numeric(ncol(betas))

  for (col in 1:ncol(betas)) {
    max_plaus <- tapply(poss, betas[, col], max, na.rm = TRUE)

    beta_vals <- as.numeric(names(max_plaus))
    valid <- is.finite(beta_vals) & is.finite(max_plaus)
    alpha_cut <- valid & (max_plaus > alpha)
    valid_betas <- beta_vals[alpha_cut]

    # Save results into our vectors
    mle_vals[col] <- mle_coefs[col]
    lower_bound[col] <- if (length(valid_betas) > 0) min(valid_betas) else NA
    upper_bound[col] <- if (length(valid_betas) > 0) max(valid_betas) else NA
  }

  # Build the structured data frame
  summary_table <- data.frame(
    Estimate = sprintf("%.4f", mle_vals),
    Lower = sprintf("%.4f", lower_bound),
    Upper = sprintf("%.4f", upper_bound),
    row.names = row_names
  )

  # Rename columns to show the interval width dynamically
  colnames(summary_table) <- c(
    "MLE",
    paste0("Lower ", (1 - alpha) * 100, "%"),
    paste0("Upper ", (1 - alpha) * 100, "%")
  )

  # Print the formatted table cleanly
  print(summary_table, quote = FALSE)
  cat("------------------------------------------------------------------------\n")
  invisible(x)
}


#' Extract the log likelihood at the MLE
#'
#' Extracts the logLik at the MLE value. Will be very similar to glm's implementation, but will be minor differences based on the implementation to arrive at the MLE. Most notably, since the gamma and inverse gaussian case use a MLE estimator (instead of glm's default Pearson MOM estimator) for the dispersion parameter, this will provide different results in this case.
#'
#' @param object A 'glim_object' from 'glim()'
#' @param ... Additional arguments
#' @return A logLik class object
#' @export
logLik.glim_object <- function(object, ...) {
  val <- object$logLik
  attr(val, "df") <- length(object$mle_coefs)
  attr(val, "nobs") <- length(object$y)
  class(val) <- "logLik"
  return(val)
}

#' Extract the log likelihood at the MLE
#'
#' Extracts the logLik at the MLE value. Will be very similar to glm's implementation, but will be minor differences based on the implementation to arrive at the MLE. Most notably, since the gamma and inverse gaussian case use a MLE estimator (instead of glm's default Pearson MOM estimator) for the dispersion parameter, this will provide different results in this case.
#'
#' @param object A 'glim_object' from 'glim()'
#' @param ... Additional arguments
#' @return A logLik class object
#' @export
coef.glim_object <- function(object, ...) {
  val <- object$logLik
  attr(val, "df") <- length(object$mle_coefs)
  attr(val, "nobs") <- length(object$y)
  class(val) <- "logLik"
  return(val)
}

#' Extract the log likelihood at the MLE
#'
#' Extracts the logLik at the MLE value. Will be very similar to glm's implementation, but will be minor differences based on the implementation to arrive at the MLE. Most notably, since the gamma and inverse gaussian case use a MLE estimator (instead of glm's default Pearson MOM estimator) for the dispersion parameter, this will provide different results in this case.
#'
#' @param object A 'glim_object' from 'glim()'
#' @param ... Additional arguments
#' @return A matrix containing the confidence/credible interval
#' @export
confint.glim_object <- function(object, ...) {
  alpha <- .05
  args <- list(...)
  if ("alpha" %in% names(args)) {
    alpha <- args$alpha
  }
  row_names <- colnames(object$X)
  lower_bound <- numeric(ncol(object$X))
  upper_bound <- numeric(ncol(object$X))

  for (col in 1:ncol(object$X)) {
    max_plaus <- tapply(object$possibilities, object$betas[, col], max, na.rm = TRUE)

    beta_vals <- as.numeric(names(max_plaus))
    valid <- is.finite(beta_vals) & is.finite(max_plaus)
    alpha_cut <- valid & (max_plaus > alpha)
    valid_betas <- beta_vals[alpha_cut]

    # Save results into our vectors
    lower_bound[col] <- if (length(valid_betas) > 0) min(valid_betas) else NA
    upper_bound[col] <- if (length(valid_betas) > 0) max(valid_betas) else NA
  }
  out <- matrix(c(lower_bound, upper_bound), ncol = 2)
  lower_a <- (alpha / 2) * 100
  upper_a <- (1 - (alpha / 2)) * 100
  colnames(out) <- c(sprintf("%.1f %%", lower_a), sprintf("%.1f %%", upper_a))
  rownames(out) <- row_names
  return(out)
}

#' Returns the variance-covariance at the MLE
#'
#' Extracts the variance-covariance matrix from the glim object.
#'
#' @param object A 'glim_object' from 'glim()'
#' @param ... Additional arguments
#' @return A logLik class object
#' @export
vcov.glim_object <- function(object, ...) {
  return(object$vcov)
}

#' nobs
#'
#' Returns the number of observations
#'
#' @param object A 'glim_object' from 'glim()'
#' @param ... Additional arguments
#' @return nobs
#' @export
nobs.glim_object <- function(object, ...) {
  return(length(object$y))
}

#' Evaluate log-likelihoods
#'
#' Helper function to evaluate log-likelihoods.
#'
#' @param family "gaussian", "binomial", "gamma", "inverse.gaussian", "poisson"
#' @param y Vector of y values. Required to have more than 1 observation
#' @param X Design matrix.
#' @param betas Either a vector of the beta values, or a matrix where each row corresponds to the beta vector to evaluate. Will be transposed so that it conforms to the dimensions of X
#' @return A vector/scalar of Log-Likelihood value(s)
#' @examples
#' \dontrun{
#' y <- rgamma(10, 3, 2)
#' X <- cbind(rep(1, 10), runif(10))
#' beta_eval_1 <- c(2.3, 12.4)
#' beta_eval_2 <- c(5, 18.9)
#' beta_eval_3 <- c(3.14, 22.2)
#' betas <- matrix(c(beta_eval_1, beta_eval_2, beta_eval_3), nrow = 3, byrow = TRUE)
#' compute_ll("gamma", y, X, betas)
#' }
#' @export
compute_ll <- function(family, y, X, betas) {
  if (is.matrix(betas)) {
    betas <- t(betas)
  }
  ll <- NULL
  if (family == "gaussian") {
    ll <- compute_gaussian_ll_r(
      y,
      X %*% betas,
      sqrt(est_dispersion_normal(y, X %*% fit_gaussian_cpp(X, y), ncol(X)))
    )
  } else if (family == "gamma") {
    ll <- compute_gamma_ll_r(
      y = y,
      exp(X %*% betas),
      shape = 1 /
        mle_estimate_dispersion_gamma(
          y,
          exp(X %*% fit_gamma_log_cpp(X = X, XtX = t(X) %*% X, y = y, rep(1, ncol(X)), FALSE)),
          ncol(X)
        )
    )
  } else if (family == "poisson") {
    ll <- compute_poisson_ll_r(y, exp(X %*% betas))
  } else if (family == "binomial") {
    ll <- compute_logistic_ll_r(X, y, betas)
  } else if (family == "inverse.gaussian") {
    # Again, note that 1/eta will be element-wise
    eta <- X %*% betas
    ll <- compute_invgauss_ll_r(
      y,
      sqrt(1 / eta),
      gamma_val = mle_estimate_dispersion_inv_gauss(y = y, mean(y))
    )
  } else {
    stop("family not defined")
  }
  return(ll)
}
