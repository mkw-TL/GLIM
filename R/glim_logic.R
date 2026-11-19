#' @useDynLib GLIM, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom progress progress_bar
#' @importFrom parallel detectCores
#' @importFrom stats logLik glm Gamma poisson inverse.gaussian lm coef runif qchisq model.matrix rnorm model.offset model.response binomial nobs vcov ave step
#' @importFrom utils head
#' @importFrom graphics par plot.default abline grid mtext
#' @importFrom methods is
#' @importFrom graphics contour
#' @import RcppProgress
NULL


# Original code written by Joe Harrison (joe.harrison.va@gmail.com), translated to C++ by Gemini
#
# Linear Model formula parsing is copied from stats::lm in R.
# Copyright (C) The R Core Team
#
# Distributed under the terms of the GNU General Public License,
# version 3.

gradient <- function(family, X, y, beta, dispersion = 1) {
  # Calculate the linear predictor vector for all observations
  eta <- as.vector(X %*% beta)

  if (family == "gaussian") {
    mu <- eta
    return(as.vector(crossprod(X, y - mu)) / dispersion)
  } else if (family == "poisson") {
    mu <- exp(eta)
    return(as.vector(crossprod(X, y - mu)))
  } else if (family == "binomial") {
    mu <- 1 / (1 + exp(-eta))
    return(as.vector(crossprod(X, y - mu)))
  } else if (family == "gamma") {
    mu <- exp(eta)
    return(as.vector(crossprod(X, (y / mu) - 1)) / dispersion)
  } else if (family == "inverse.gaussian") {
    mu <- eta^(-1 / 2)
    return(as.vector(crossprod(X, mu - y)) / (2 * dispersion))
  } else {
    stop("Family provided to the gradient not recognized")
  }
}

partial <- function(family, X, y, beta, dispersion, j) {
  # Calculate the linear predictor vector for all n observations once
  eta <- as.vector(X %*% beta)

  if (family == "gaussian") {
    mu <- eta
    return(sum((y - mu) * X[, j]) / dispersion)
  } else if (family == "poisson") {
    mu <- exp(eta)
    return(sum((y - mu) * X[, j]))
  } else if (family == "binomial") {
    # Note the added negative sign for the logit inverse
    mu <- 1 / (1 + exp(-eta))
    return(sum((y - mu) * X[, j]))
  } else if (family == "gamma") {
    mu <- exp(eta)
    return(sum((y / mu - 1) * X[, j]) / dispersion)
  } else if (family == "inverse.gaussian") {
    mu <- eta^(-1 / 2)
    return(sum((mu - y) * X[, j]) / (2 * dispersion))
  } else {
    stop("Family provided to the gradient not recognized")
  }
}

generate_marginal_object <- function(
  X,
  y,
  family,
  mle_coefs,
  ll_mle_original_data,
  vcov = NULL,
  n_grid_evals = 25,
  dispersion = 1,
  alpha = 0.05,
  step = NULL,
  m = 50,
  parallel = TRUE,
  max_iter = 500
) {
  margins <- list()
  p <- ncol(X)

  for (col in 1:p) {
    message(sprintf("Marginalizing col %d out of %d", col, p))
    step_col <- step[col]
    off_idx <- setdiff(1:p, col)

    max_est <- 2 * n_grid_evals + 100
    beta_list <- vector("list", max_est)
    plaus_list <- vector("list", max_est)

    beta_list[[1]] <- mle_coefs
    plaus_list[[1]] <- 1
    count <- 1

    for (direction in c(-1, 1)) {
      beta <- mle_coefs
      current_plaus <- 1
      iter <- 0

      while (current_plaus > (alpha - 0.01) && iter < max_iter) {
        iter <- iter + 1

        tang <- profile_tangent(family, X, beta, col, dispersion)
        if (is.null(tang) | any(!is.finite(tang))) {
          warning(sprintf(
            "Column %d: profile tangent failed or non-finite; extension stopped.",
            col
          ))
          break
        }

        cand <- beta
        cand[col] <- cand[col] + direction * step_col
        cand[off_idx] <- cand[off_idx] + direction * step_col * tang

        if (!all(is.finite(cand)) | !is.finite(compute_ll(family, y, X, cand, mle_coefs))) {
          warning(sprintf(
            "Column %d: step left the valid parameter domain; extension stopped.",
            col
          ))
          break
        }

        beta <- correct_profile_point(family, X, y, cand, col, dispersion, mle_coefs)
        if (is.null(beta) | any(!is.finite(beta))) {
          warning(sprintf(
            "Column %d: profile correction returned non-finite values; extension stopped.",
            col
          ))
          break
        }

        raw_plaus <- glim_raw(
          X,
          y,
          family = family,
          betas = matrix(beta, nrow = 1),
          mle_coefs = mle_coefs,
          ll_mle_original_data = ll_mle_original_data,
          m = m,
          parallel = TRUE,
          approx = FALSE,
          show_progress = FALSE
        )

        current_plaus <- as.numeric(raw_plaus)

        if (length(current_plaus) == 0 | !is.finite(current_plaus)) {
          warning(sprintf(
            "Column %d: plausibility returned non-finite value; extension stopped.",
            col
          ))
          break
        }

        count <- count + 1

        if (count > length(beta_list)) {
          length(beta_list) <- length(beta_list) * 2
          length(plaus_list) <- length(plaus_list) * 2
        }

        beta_list[[count]] <- beta
        plaus_list[[count]] <- current_plaus
      }
    }

    beta_for_margin <- do.call(rbind, beta_list[1:count])
    plaus_for_margin <- unlist(plaus_list[1:count], use.names = FALSE)

    margins[[col]] <- list(beta_for_margin = beta_for_margin, plaus_for_margin = plaus_for_margin)
  }

  obj_to_return <- list(
    margins = margins,
    extra_info = list(
      X = X,
      y = y,
      family = family,
      mle_coefs = mle_coefs,
      ll_mle_original_data = ll_mle_original_data,
      n_grid_evals = n_grid_evals,
      vcov = vcov,
      dispersion = dispersion,
      alpha = alpha,
      step = step,
      m = m,
      parallel = parallel
    )
  )
  class(obj_to_return) <- "glim_object"
  return(obj_to_return)
}


#' Generate Elliptical Approximation Samples (Inner Probability)
#'
#' Internal function called to generate samples when the elliptical approximation is used.
#'
#' @param X Input predictor matrix.
#' @param y Response vector.
#' @param family A string indicating the error distribution. Default is `"gaussian"`.
#' @param mle_coefs Maximum likelihood estimates of the coefficients.
#' @param ll_mle_original_data The log-likelihood value at the maximum likelihood estimates.
#' @param m Number of samples to generate.
#' @param parallel Logical indicating whether to use parallel processing.
#' @return A matrix of generated samples based on the elliptical approximation.
#' @noRd
glim_inner_prob_approx_samples <- function(
  X,
  y,
  family = "gaussian",
  mle_coefs,
  ll_mle_original_data,
  m,
  parallel,
  eJ,
  dispersion,
  a_val,
  b_val,
  max_it,
  tol
) {
  n_threads <- 1
  if (parallel) {
    cores <- parallel::detectCores(logical = FALSE)
    if (is.na(cores)) {
      cores <- parallel::detectCores()
    }
    n_threads <- max(1L, cores - 1L)
  }
  if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    old <- RhpcBLASctl::blas_get_num_procs()
    RhpcBLASctl::blas_set_num_threads(1)
    on.exit(RhpcBLASctl::blas_set_num_threads(old), add = TRUE)
  }
  B <- 99
  AA <- seq(0.01, 0.99, length.out = B)

  i <- 0
  xi <- list()
  coordinate_marginal_bounds <- list()
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
      ll_mle_original_data,
      as.matrix(eJ$vectors),
      as.vector(eJ$values),
      dispersion,
      generate_grid = FALSE, # unused argument to be deleted
      tol = tol,
      a_val = a_val,
      b_val = b_val,
      max_it = max_it,
      parallel = TRUE,
      m = m,
      base_seed = 31,
      num_threads = n_threads
    )
    prev_xi <- xi[[i]]
    q_val <- qchisq(1 - alpha, df = length(mle_coefs)) # This is the first guess at our quantile distance, which we need to modify slightly by xi.
    base_scale <- sqrt(dispersion * q_val * (1 / eJ$values))
    semi_axes <- base_scale * sqrt(prev_xi)
    transformed_mat <- eJ$vectors %*% diag(as.vector(semi_axes)) # Takes the principle axes (through diag) and scales them (by semi_axes), then rotate to eigen_vector span

    # To make the beta coordinate the largest, we have our transformation matrix (times u, where u is a unit sphere).
    # Then, for a particular coordinate, we have b1 = row_1 * u, where this is maximized if u is in the direction of row_1.
    # Is component-wise squaring
    H <- sqrt(rowSums(transformed_mat^2))
    coordinate_marginal_bounds[[i]] <- H
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

    z <- rnorm(mle_coefs, 0, 1)
    rand_dir <- z / norm(z, "2")
    spatial_dir <- eJ$vectors %*% (1 / sqrt(eJ$values) * rand_dir) # map it out to the corresponding elliptical distance

    samples[, i] <- mle_coefs +
      as.vector(
        sqrt(qchisq(1 - u, length(mle_coefs))) * sqrt(lerped_xi) * spatial_dir * sqrt(dispersion)
      )
  }
  return(list(
    samples = samples,
    xi = xi,
    alphas = AA,
    coordinate_marginal_bounds = coordinate_marginal_bounds
  ))
}

glm_weights <- function(family, eta, dispersion = 1) {
  if (family %in% c("gaussian", "normal")) {
    rep(1 / dispersion, length(eta))
  } else if (family == "poisson") {
    exp(eta)
  } else if (family == "binomial") {
    mu <- 1 / (1 + exp(-eta))
    mu * (1 - mu)
  } else if (family == "gamma") {
    rep(1 / dispersion, length(eta))
  } else if (family == "inverse.gaussian") {
    mu <- eta^(-1 / 2) # NaN if eta < 0; callers check is.finite()
    (mu^3) / (4 * dispersion)
  } else {
    stop("Unsupported family for weight calculation")
  }
}

# db_0/db_c is is what we are calculating. RETURNS A P-1 vector. Does not append the value of 1 for the partial deriv of the column wrt to itself
profile_tangent <- function(family, X, beta, col, dispersion = 1) {
  off_idx <- setdiff(1:length(beta), col)
  if (length(off_idx) == 0) {
    return(numeric(0))
  }

  w <- glm_weights(family, as.vector(X %*% beta), dispersion)
  if (!all(is.finite(w))) {
    return(NULL)
  }

  Info <- crossprod(X, w * X)
  I_oo <- Info[off_idx, off_idx, drop = FALSE] # select the submatrix of other beta vals
  I_oc <- Info[off_idx, col]
  ridge <- max(1e-8 * mean(diag(I_oo)), 1e-12)

  # -I_oo^-1 * I_oc is db_0/db_c.
  tryCatch(as.vector(-solve(I_oo + diag(ridge, length(off_idx)), I_oc)), error = function(e) NULL)
}

# Does NOT handle the logic of the step-size. Only operates on the p-1 other columns
correct_profile_point <- function(
  family,
  X,
  y,
  beta,
  col,
  dispersion = 1,
  mle_coefs = NULL,
  max_iter = 50,
  tol = 1e-6
) {
  beta <- as.numeric(beta)
  p <- length(beta)
  off_idx <- setdiff(seq_len(p), col)
  if (length(off_idx) == 0L) {
    return(beta) # nothing to profile out
  }
  X_others <- X[, off_idx, drop = FALSE]

  for (iter in seq_len(max_iter)) {
    grad_off <- gradient(family = family, X = X, y = y, beta = beta, dispersion = dispersion)[
      off_idx
    ]

    if (!all(is.finite(grad_off))) {
      break
    }
    if (max(abs(grad_off)) < tol) {
      break
    }

    eta <- as.vector(X %*% beta)
    w <- as.numeric(glm_weights(family, eta, dispersion))
    if (!all(is.finite(w))) {
      break
    }

    Info <- crossprod(X_others, w * X_others)
    ridge <- max(1e-8 * mean(diag(Info)), 1e-12)
    Info_reg <- Info + diag(ridge, length(off_idx))

    delta <- tryCatch(solve(Info_reg, grad_off), error = function(e) NULL)
    if (is.null(delta) || !all(is.finite(delta))) {
      break
    }

    # Root-finding line search: accept if gradient norm decreases (bypasses deviance/loglik sign mismatch)
    step_multiplier <- 1
    accepted <- FALSE
    proposed_beta <- beta

    while (step_multiplier > 1e-10) {
      proposed_beta[off_idx] <- beta[off_idx] + step_multiplier * delta
      grad_new <- gradient(family, X, y, proposed_beta, dispersion)[off_idx]

      if (all(is.finite(grad_new)) && max(abs(grad_new)) <= max(abs(grad_off)) + 1e-6) {
        accepted <- TRUE
        break
      }
      step_multiplier <- step_multiplier / 2
    }

    if (!accepted) {
      break
    }
    beta <- proposed_beta
  }

  return(beta)
}

# ---- extend_profile ----------------------------------------------------------
# Geometric step growth until plausibility drops below alpha, then bisection
# on the bracket. Stepping uses a smaller m_ext and a fresh seed per step; all
# accepted points are then re-evaluated once at full m. Signature is backward
# compatible (new arguments are at the end).
extend_profile <- function(
  direction,
  needs_extension,
  betas,
  col,
  possibilities,
  alpha,
  tolerance,
  step,
  family,
  X,
  y,
  dispersion,
  mle_coefs,
  ll_mle_original_data,
  m,
  parallel,
  approx,
  max_steps = 40,
  n_bisect = 6,
  growth = 1.5,
  m_ext = min(m, 300),
  seed0 = 7919L
) {
  p <- length(mle_coefs)
  off_idx <- setdiff(seq_len(p), col)
  empty <- list(betas = matrix(nrow = 0, ncol = p), plaus = numeric(0))
  if (!needs_extension) {
    return(empty)
  }

  edge_val <- if (direction > 0) max(betas[, col]) else min(betas[, col])
  edge_rows <- which(betas[, col] == edge_val)
  seed_row <- edge_rows[which.max(possibilities[edge_rows])]
  prev_beta <- betas[seed_row, ]

  eval_plaus <- function(b, tag, mm) {
    as.numeric(glim_raw(
      X,
      y,
      family = family,
      betas = matrix(b, nrow = 1),
      mle_coefs = mle_coefs,
      ll_mle_original_data = ll_mle_original_data,
      m = mm,
      parallel = parallel,
      approx = approx,
      show_progress = FALSE,
      base_seed = as.integer(seed0 + 1009L * tag)
    ))
  }

  out_betas <- list()
  out_plaus <- numeric(0)
  tag <- 0L
  h <- step
  bracket <- NULL
  stop_reason <- NULL

  # ---- Phase 1: grow until plausibility falls below alpha --------------------
  for (k in seq_len(max_steps)) {
    tang <- profile_tangent(family, X, prev_beta, col, dispersion)
    if (is.null(tang)) {
      stop_reason <- "profile tangent failed"
      break
    }
    cand <- prev_beta
    cand[col] <- cand[col] + direction * h
    cand[off_idx] <- cand[off_idx] + direction * h * tang
    if (!is.finite(compute_ll(family, y, X, cand, mle_coefs))) {
      stop_reason <- "step left the valid parameter domain"
      break
    }
    new_beta <- correct_profile_point(family, X, y, cand, col, dispersion, mle_coefs)
    tag <- tag + 1L
    pl <- eval_plaus(new_beta, tag, m_ext)
    if (!is.finite(pl)) {
      stop_reason <- "non-finite plausibility"
      break
    }
    out_betas[[length(out_betas) + 1L]] <- unname(new_beta)
    out_plaus <- c(out_plaus, pl)

    if (pl < alpha) {
      bracket <- list(lo = prev_beta, hi = new_beta) # lo above alpha, hi below
      break
    }
    prev_beta <- new_beta
    h <- h * growth
  }

  if (is.null(bracket) && is.null(stop_reason)) {
    warning(sprintf("Column %d: no crossing of alpha within %d steps.", col, max_steps))
  } else if (!is.null(stop_reason)) {
    warning(sprintf("Column %d: %s; extension stopped.", col, stop_reason))
  }

  # ---- Phase 2: bisect the bracket ------------------------------------------
  if (!is.null(bracket)) {
    lo <- bracket$lo
    hi <- bracket$hi
    for (b in seq_len(n_bisect)) {
      mid <- (lo + hi) / 2
      mid <- correct_profile_point(family, X, y, mid, col, dispersion, mle_coefs)
      tag <- tag + 1L
      pl <- eval_plaus(mid, tag, m_ext)
      if (!is.finite(pl)) {
        break
      }
      out_betas[[length(out_betas) + 1L]] <- unname(mid)
      out_plaus <- c(out_plaus, pl)
      if (abs(pl - alpha) <= tolerance) {
        break
      }
      if (pl >= alpha) lo <- mid else hi <- mid
    }
  }

  if (length(out_betas) == 0L) {
    return(empty)
  }

  out_betas <- do.call(rbind, out_betas)

  # ---- Re-evaluate everything accepted at full m (one batched call) ----------
  if (m_ext < m) {
    out_plaus <- as.numeric(glim_raw(
      X,
      y,
      family = family,
      betas = out_betas,
      mle_coefs = mle_coefs,
      ll_mle_original_data = ll_mle_original_data,
      m = m,
      parallel = parallel,
      approx = approx,
      show_progress = FALSE,
      base_seed = as.integer(seed0 + 104729L)
    ))
  }

  ord <- order(direction * out_betas[, col])
  list(betas = out_betas[ord, , drop = FALSE], plaus = out_plaus[ord])
}

.is_flag <- function(x) is.logical(x) && length(x) == 1 && !is.na(x)
.is_whole <- function(x) {
  is.numeric(x) && length(x) == 1 && is.finite(x) && x %% 1 == 0
}
.is_num1 <- function(x) is.numeric(x) && length(x) == 1 && is.finite(x)


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
  family = "gaussian",
  data = NULL,
  m = 1000,
  approx = FALSE,
  radial = FALSE,
  tol = .005,
  ...
) {
  # Allows family = "pois" to auto-match to "poisson"
  family <- match.arg(
    family,
    choices = c("gaussian", "poisson", "gamma", "binomial", "inverse.gaussian")
  )

  if (!.is_flag(approx)) {
    stop("Input Error: 'approx' must be either TRUE or FALSE.")
  }
  if (!.is_flag(radial)) {
    stop("Input Error: 'radial' must be either TRUE or FALSE.")
  }
  if (radial && approx) {
    stop("Input Error: radial and approx cannot both be TRUE.")
  }

  if (!(.is_whole(m) && m > 0)) {
    stop("Input Error: m must be a positive integer")
  }
  if (!(.is_num1(tol) && tol > 0 && tol < 1)) {
    stop("Input Error: tol must be a number strictly between 0 and 1")
  }

  args <- list(...)

  # Unknown / unnamed extra arguments
  if (length(args) > 0) {
    allowed <- c(
      "a_val",
      "b_val",
      "max_it",
      "betas",
      "n_grid_evals",
      "num_samps",
      "alpha",
      "directly_calc_marginals",
      "trim",
      "parallel",
      "base_seed"
    )
    nms <- names(args)
    if (is.null(nms) || any(nms == "") || !all(nms %in% allowed)) {
      stop("Incorrect names of additional arguments passed")
    }
  }

  if ("a_val" %in% names(args)) {
    if (.is_num1(args$a_val) && args$a_val > 0) {
      a_val <- args$a_val
    } else {
      stop("Input Error: a_val must be a positive number")
    }
  } else {
    a_val <- 1.5
  }

  if ("b_val" %in% names(args)) {
    if (.is_num1(args$b_val) && args$b_val > 0) {
      b_val <- args$b_val
    } else {
      stop("Input Error: b_val must be a positive number")
    }
  } else {
    b_val <- .65
  }

  if ("parallel" %in% names(args)) {
    if (.is_flag(args$parallel)) {
      parallel <- args$parallel
    } else {
      stop("Input Error: 'parallel' must be either TRUE or FALSE.")
    }
  } else {
    parallel <- TRUE
  }

  if ("n_grid_evals" %in% names(args)) {
    if (.is_whole(args$n_grid_evals) && args$n_grid_evals > 1) {
      n_grid_evals <- as.integer(args$n_grid_evals)
    } else {
      stop("Input Error: n_grid_evals must be an integer greater than 1")
    }
  } else {
    n_grid_evals <- 17 # Is this a good number?
  }

  if ("max_it" %in% names(args)) {
    if (.is_whole(args$max_it) && args$max_it > 0) {
      max_it <- args$max_it
    } else {
      stop("Input Error: max_it must be a positive integer")
    }
  } else {
    max_it <- 25
  }

  if ("num_samps" %in% names(args)) {
    # Accept 1000 as well as 1000L
    if (.is_whole(args$num_samps) && args$num_samps > 0) {
      num_samps <- as.integer(args$num_samps)
    } else {
      stop("Input Error: num_samps must be a positive integer")
    }
  } else {
    num_samps <- 2000
  }

  if ("alpha" %in% names(args)) {
    if (.is_num1(args$alpha) && args$alpha > 0 && args$alpha < 1) {
      alpha <- args$alpha
    } else {
      stop("Input Error: alpha must be a number strictly between 0 and 1")
    }
  } else {
    alpha <- .05
  }

  directly_calc_marginals <- FALSE
  if ("directly_calc_marginals" %in% names(args)) {
    if (.is_flag(args$directly_calc_marginals)) {
      directly_calc_marginals <- args$directly_calc_marginals
    } else {
      stop("Input Error: directly_calc_marginals must be TRUE or FALSE")
    }
  }

  trim <- TRUE
  if ("trim" %in% names(args)) {
    if (.is_flag(args$trim)) {
      trim <- args$trim
    } else {
      stop("Input Error: trim must be TRUE or FALSE")
    }
  }

  ## ---- 2. Model frame (drops incomplete cases) ----
  mf <- match.call(expand.dots = FALSE)
  match_idx <- match(
    c("formula", "data", "subset", "weights", "na.action", "offset"),
    names(mf),
    0L
  )
  mf <- mf[c(1L, match_idx)]
  mf$drop.unused.levels <- TRUE

  # Force na.omit so NAs never reach C++
  if (is.null(mf$na.action)) {
    mf$na.action <- quote(stats::na.omit)
  }
  mf[[1L]] <- quote(stats::model.frame)
  env <- environment(formula)
  if (is.null(env)) {
    env <- parent.frame()
  }
  mf <- eval(mf, env)
  n_dropped <- length(attr(mf, "na.action"))
  if (n_dropped > 0L) {
    message(sprintf(
      ngettext(
        n_dropped,
        "%d row has been dropped from the analysis due to missing values.",
        "%d rows have been dropped from the analysis due to missing values."
      ),
      n_dropped
    ))
  }
  mt <- attr(mf, "terms")

  # Check if any predictor (including columns inside a matrix predictor) is all 1s
  if (attr(mt, "intercept") == 1L && ncol(mf) > 1L) {
    has_explicit_intercept <- FALSE

    # Loop over predictor variables (skipping column 1, which is the response)
    for (j in 2:ncol(mf)) {
      col_data <- mf[[j]]
      if (is.matrix(col_data)) {
        # Check each column of a matrix predictor (e.g., X_poisson)
        if (any(apply(col_data, 2L, function(x) is.numeric(x) && all(x == 1)))) {
          has_explicit_intercept <- TRUE
          break
        }
      } else if (is.numeric(col_data) && all(col_data == 1)) {
        # Also catches a standalone vector of 1s (e.g., y ~ x1 + x2)
        has_explicit_intercept <- TRUE
        break
      }
    }

    # Drop the automatic intercept from the terms object if an explicit one exists
    if (has_explicit_intercept) {
      attr(mt, "intercept") <- 0L
    }
  }

  X <- model.matrix(mt, mf, NULL)
  y <- model.response(mf, "any")
  if (any(is.na(y))) {
    stop("NA values detected in the response.")
  }

  ## ---- 3. Validation that needs X / y ----
  betas <- NULL
  if (!approx && "betas" %in% names(args)) {
    b_mat <- args$betas

    # Automatically convert data.frames to matrices
    if (is.data.frame(b_mat)) {
      b_mat <- as.matrix(b_mat)
    }

    # Strictly require a double matrix (rejects "a" and integer matrices)
    if (!is.matrix(b_mat) || typeof(b_mat) != "double") {
      stop("Input Error: Grid of betas must be a numeric matrix.")
    }

    if (ncol(b_mat) != ncol(X)) {
      stop("Input Error: Grid of betas columns must match the number of predictors.")
    }

    betas <- b_mat # Save the coerced, validated matrix back
  }

  if (!("betas" %in% names(args))) {
    if (ncol(X) > 5) {
      directly_calc_marginals <- TRUE
    }
  }

  if (NROW(X) != NROW(y)) {
    stop(
      "Input Error: The number of rows in the design matrix X must equal the length of the response vector y."
    )
  }

  if (NCOL(X) > NROW(X)) {
    warning(
      "Input Warning: The number of predictors (p) is greater than the number of observations (n)."
    )
  }

  # Response domain checks per family
  if (family == "poisson") {
    if (!is.numeric(y) || any(y < 0 | y != floor(y))) {
      stop("Input Error: For Poisson family, 'y' must contain only non-negative integers.")
    }
  } else if (family %in% c("gamma", "inverse.gaussian")) {
    if (!is.numeric(y) || any(y <= 0)) {
      stop(sprintf(
        "Input Error: For %s family, all values in 'y' must be strictly positive (y > 0).",
        family
      ))
    }
  }

  J <- NULL
  if (is.data.frame(X)) {
    X <- as.matrix(X)
  }

  ## ---- 4. Family-specific MLE setup ----
  ## Binomial setup
  if (family == "binomial") {
    if (is.factor(y)) {
      if (nlevels(y) != 2) {
        stop("Factor response must have exactly 2 levels")
      }
      y <- as.numeric(y == levels(y)[2]) # levels(y)[2] is treated as success
    }

    if (is.null(dim(y))) {
      if (length(y) != nrow(X)) {
        stop("y and X lengths differ")
      }
      if (!all(y %in% c(0, 1))) {
        stop("Binomial vector response must contain only 0s and 1s")
      }
    } else if (is.matrix(y) && ncol(y) == 2) {
      if (nrow(y) != nrow(X)) {
        stop("y and X row counts differ")
      }
      successes <- y[, 1]
      failures <- y[, 2]

      idx_success <- rep(seq_len(nrow(X)), times = successes)
      idx_fail <- rep(seq_len(nrow(X)), times = failures)

      X <- rbind(X[idx_success, , drop = FALSE], X[idx_fail, , drop = FALSE])
      y <- c(rep(1, sum(successes)), rep(0, sum(failures)))
    } else {
      stop("Invalid response format for binomial family")
    }
    # -1 because X already contains the intercept column (if any)
    # suppressWarnings: glm warns about fitted probabilities of 0/1, but we
    # detect that case ourselves right below and give a clearer error.
    fit <- suppressWarnings(glm(y ~ X - 1, family = binomial))
    eps <- 10 * .Machine$double.eps
    if (any(fit$fitted.values > (1 - eps)) || any(fit$fitted.values < eps)) {
      stop("Data is completely separable. MLE does not exist")
    }
    mle_coefs <- fit$coefficients
    if (any(is.na(mle_coefs))) {
      stop(
        "Design matrix X is rank deficient / collinear. Remove redundant columns before calling glim()."
      )
    }
    vcov <- vcov(fit)
    p_i <- 1 / (1 + exp(-X %*% mle_coefs))
    dispersion <- 1
    J <- crossprod(X, X * as.vector((p_i * (1 - p_i))))
    p2p_function <- prob2poss_logis

    ll_mle_original_data <- compute_logistic_ll_r(X, y, mle_coefs)

    ## Gamma setup
  } else if (family == "gamma") {
    fit <- glm(y ~ X - 1, family = Gamma(link = "log"))
    mle_coefs <- coef(fit)
    if (any(is.na(mle_coefs))) {
      stop(
        "Design matrix X is rank deficient / collinear. Remove redundant columns before calling glim()."
      )
    }
    vcov <- vcov(fit)
    eta <- X %*% mle_coefs
    dispersion <- mle_estimate_dispersion_gamma(y, exp(X %*% mle_coefs), length(mle_coefs))
    ll_mle_original_data <- compute_gamma_ll_r(y, eta, 1 / dispersion)
    J <- crossprod(X, X)
    p2p_function <- prob2poss_gamma

    ## Poisson setup
  } else if (family == "poisson") {
    pois_fit <- glm(y ~ X - 1, family = poisson(link = "log"))
    if (!pois_fit$converged) {
      stop("Poisson GLM fails to converge")
    }
    vcov <- vcov(pois_fit)
    mle_coefs <- coef(pois_fit)
    if (any(is.na(mle_coefs))) {
      stop(
        "Design matrix X is rank deficient / collinear. Remove redundant columns before calling glim()."
      )
    }
    ll_mle_original_data <- as.numeric(logLik(pois_fit))
    eta <- X %*% mle_coefs
    lambda_i <- exp(eta)
    dispersion <- 1
    J <- crossprod(X, X * as.vector(lambda_i))
    p2p_function <- prob2poss_poisson

    ## Inverse Gaussian setup
  } else if (family == "inverse.gaussian") {
    inv_gaus_fit <- glm(y ~ X - 1, family = inverse.gaussian(link = "1/mu^2"))
    vcov <- vcov(inv_gaus_fit)
    ll_mle_original_data <- as.numeric(logLik(inv_gaus_fit))
    mle_coefs <- coef(inv_gaus_fit)
    if (any(is.na(mle_coefs))) {
      stop(
        "Design matrix X is rank deficient / collinear. Remove redundant columns before calling glim()."
      )
    }
    eta <- X %*% mle_coefs
    mu_i <- as.vector(sqrt(1 / eta))
    dispersion <- mle_estimate_dispersion_inv_gauss(y, mean(y))
    J <- crossprod(X, X * (mu_i^3) / 4)
    p2p_function <- prob2poss_invgauss

    ## Gaussian setup
  } else if (family == "gaussian") {
    fit <- lm(y ~ X - 1)
    ll_mle_original_data <- as.numeric(logLik(fit))
    mle_coefs <- coef(fit)
    if (any(is.na(mle_coefs))) {
      stop(
        "Design matrix X is rank deficient / collinear. Remove redundant columns before calling glim()."
      )
    }
    vcov <- vcov(fit)
    dispersion <- est_dispersion_normal(y, X %*% mle_coefs, length(mle_coefs))
    J <- crossprod(X, X)
    p2p_function <- prob2poss_gaussian
  } else {
    stop("Family not supported")
  }

  if (!is.null(J)) {
    J <- (J + t(J)) / 2 # symmetrize to kill some rounding asymmetries
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

  # FIX: `missing()` only works on function arguments, not local variables.
  if (
    !is.numeric(dispersion) || length(dispersion) != 1 || !is.finite(dispersion) || dispersion <= 0
  ) {
    stop("Internal error: 'dispersion' must be a strictly positive numeric value.")
  }

  show_progress <- FALSE
  eigen_vecs <- eJ$vectors
  eigen_vals <- eJ$values
  q_val <- qchisq(1 - alpha, df = length(mle_coefs)) # This is the first guess at our quantile distance, which we need to modify slightly by xi.
  base_scale <- sqrt(dispersion * q_val * (1 / eigen_vals))
  semi_axes <- base_scale * 1 # Base ellipse only somewhat needs to be elliptical. Marginals will extend to compensate
  transformed_mat <- eigen_vecs %*% diag(as.vector(semi_axes)) # Takes the principle axes (through diag) and scales them (by semi_axes), then rotate to eigen_vector span

  # To make the beta coordinate the largest, we have our transformation matrix (times u, where u is a unit sphere).
  # Then, for a particular coordinate, we have b1 = row_1 * u, where this is maximized if u is in the direction of row_1.
  # Is component-wise squaring
  H <- sqrt(rowSums(transformed_mat^2))
  step <- 2 * H / n_grid_evals
  # bringing them out, because most all need access to this variable

  message("Our MLE is: ", paste(round(mle_coefs, 4), collapse = ", "))
  if ((approx == FALSE) && (radial == FALSE)) {
    show_progress <- TRUE
    if (is.null(betas) && ncol(X) < 6) {
      message("generating grid of betas")
      generated_grid <- generate_grid(
        X,
        y,
        family,
        mle_coefs,
        eigen_vecs = eigen_vecs,
        eigen_vals = eigen_vals,
        dispersion,
        ll_mle_original_data,
        n_grid_evals = n_grid_evals,
        trim = trim,
        m = m,
        parallel = parallel
      )
      betas <- generated_grid$betas
      step <- generated_grid$approx_grid_resolution
      message(sprintf("Evaluating %d grid points, %d glm fits each.\n", nrow(betas), m))
      if (n_grid_evals^length(mle_coefs) > 100000) {
        message(sprintf(
          "Be warned of crashes due to lack of memory For 5+ predictors, use approx = TRUE"
        ))
      }
      colnames(betas) <- colnames(X)
      possibils <- glim_raw(
        X,
        y,
        family,
        betas,
        mle_coefs,
        ll_mle_original_data,
        m,
        parallel,
        approx,
        show_progress
      )
      grid_data <- list(
        possibilities = possibils,
        samples = NULL,
        xi = NULL,
        alphas = NULL,
        coordinate_marginal_bounds = NULL,
        betas_evaluated = betas,
        family = family,
        X = X,
        y = y,
        mle_coefs = mle_coefs,
        logLik = ll_mle_original_data,
        vcov = vcov,
        dispersion = dispersion
      )
      if (directly_calc_marginals == FALSE) {
        ## Now we need to see if we extend out this automatically generated set of betas
        tolerance <- .01
        extended_in_neg_betas <- vector("list", ncol(X))
        extended_in_pos_betas <- vector("list", ncol(X))
        extended_in_neg_plaus <- vector("list", ncol(X))
        extended_in_pos_plaus <- vector("list", ncol(X))

        for (col in seq_len(ncol(X))) {
          b_col <- betas[, col]

          # Find maximum plausibility at the grid boundaries for the current parameter column
          min_b <- min(b_col, na.rm = TRUE)
          max_b <- max(b_col, na.rm = TRUE)

          max_plaus_min <- max(possibils[b_col == min_b], na.rm = TRUE)
          max_plaus_max <- max(possibils[b_col == max_b], na.rm = TRUE)

          # Check if plausibility at boundaries exceeds alpha threshold
          needs_lower <- max_plaus_min > alpha
          needs_upper <- max_plaus_max > alpha
          up <- extend_profile(
            +1,
            needs_upper,
            betas,
            col,
            possibils,
            alpha,
            tolerance,
            step[col],
            family,
            X,
            y,
            dispersion,
            mle_coefs,
            ll_mle_original_data,
            m,
            parallel,
            approx
          )
          dn <- extend_profile(
            -1,
            needs_lower,
            betas,
            col,
            possibils,
            alpha,
            tolerance,
            step[col],
            family,
            X,
            y,
            dispersion,
            mle_coefs,
            ll_mle_original_data,
            m,
            parallel,
            approx
          )
          extended_in_pos_betas[[col]] <- up$betas
          extended_in_pos_plaus[[col]] <- up$plaus
          extended_in_neg_betas[[col]] <- dn$betas
          extended_in_neg_plaus[[col]] <- dn$plaus
        }
        grid_data[["extended_in_neg_plaus"]] <- extended_in_neg_plaus
        grid_data[["extended_in_neg_betas"]] <- extended_in_neg_betas
        grid_data[["extended_in_pos_plaus"]] <- extended_in_pos_plaus
        grid_data[["extended_in_pos_betas"]] <- extended_in_pos_betas
        obj_to_return <- list(grid_data = grid_data)
        class(obj_to_return) <- "glim_object"
        return(obj_to_return)
      } else {
        marginals <- generate_marginal_object(
          X,
          y,
          family,
          mle_coefs,
          ll_mle_original_data,
          vcov,
          n_grid_evals,
          dispersion,
          alpha,
          step,
          m,
          parallel
        )
        obj_to_return <- list(marginals = marginals, grid_data = grid_data)
        class(obj_to_return) <- "glim_object"
        return(obj_to_return)
      }
    }
    if (is.null(betas) && ncol(X) > 5) {
      message(
        "\n Since the number of predictors > 5, only margins are calculated. Extra control can be given by passing in beta values: \n betas = matrix(c(beta_eval1, beta_eval2), byrow = TRUE)"
      )
      marginals <- generate_marginal_object(
        X,
        y,
        family,
        mle_coefs,
        ll_mle_original_data,
        vcov,
        n_grid_evals,
        dispersion,
        alpha,
        step,
        m,
        parallel
      )
      class(marginals) <- "glim_object"
      return(marginals)
    }
    if (!is.null(betas)) {
      possibils <- glim_raw(
        X,
        y,
        family,
        betas,
        mle_coefs,
        ll_mle_original_data,
        m,
        parallel,
        approx,
        show_progress
      )
      glim_object <- list(
        possibilities = possibils,
        samples = NULL,
        xi = NULL,
        alphas = NULL,
        coordinate_marginal_bounds = NULL,
        betas_evaluated = betas,
        family = family,
        X = X,
        y = y,
        mle_coefs = mle_coefs,
        logLik = ll_mle_original_data,
        vcov = vcov,
        dispersion = dispersion
      )
      if (directly_calc_marginals == FALSE) {
        class(glim_object) <- "glim_object"
        return(glim_object)
      } else {
        marginals <- generate_marginal_object(
          X,
          y,
          family,
          mle_coefs,
          ll_mle_original_data,
          vcov,
          n_grid_evals,
          dispersion,
          alpha,
          step,
          m,
          parallel
        )
        obj_to_return <- list(glim_object = glim_object, marginals = marginals)
        class(obj_to_return) <- "glim_object"
        return(obj_to_return)
      }
    }
  }

  if ((approx == TRUE) && (radial == FALSE)) {
    inner_prob <- glim_inner_prob_approx_samples(
      X = X,
      y = y,
      family = family,
      mle_coefs = mle_coefs,
      ll_mle_original_data = ll_mle_original_data,
      m = m,
      parallel = parallel,
      eJ = eJ,
      dispersion = dispersion,
      a_val = a_val,
      b_val = b_val,
      max_it = max_it,
      tol = tol
    )
    elliptical_data <- list(
      samples = inner_prob$samples,
      xi = inner_prob$xi,
      alphas = inner_prob$alphas,
      coordinate_marginal_bounds = inner_prob$coordinate_marginal_bounds
    )
    margin <- generate_marginal_object(
      X,
      y,
      family,
      mle_coefs,
      ll_mle_original_data,
      vcov,
      n_grid_evals,
      dispersion,
      alpha,
      step,
      m,
      parallel
    )
    obj_to_return <- list(margin = margin, elliptical_data = elliptical_data)
    class(obj_to_return) <- "glim_object"
    return(obj_to_return)
  }

  if (radial == TRUE && approx == TRUE) {
    stop("Radial and Elliptical approximation methods cannot both be used")
  }
  if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    old <- RhpcBLASctl::blas_get_num_procs()
    RhpcBLASctl::blas_set_num_threads(1)
    on.exit(RhpcBLASctl::blas_set_num_threads(old), add = TRUE)
  }
  # radial and elliptical ALWAYS calculate actual marginals
  if (radial == TRUE && approx == FALSE) {
    message("For 5+ predictors, the elliptical (approx) method is recommended")
    radial_list <- radial(
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
      n_grid_evals = n_grid_evals,
      num_omp_threads = num_omp_threads
    )
    margin <- generate_marginal_object(
      X,
      y,
      family,
      mle_coefs,
      ll_mle_original_data,
      vcov,
      n_grid_evals,
      dispersion,
      alpha,
      step,
      m,
      parallel
    )
    samples <- radial_list$output_samples
    betas_evaluated <- radial_list$betas_evaluated
    radial_data <- list(samples = samples, betas_evaluated = betas_evaluated)
    obj_to_return <- list(margin = margin, radial_data = radial_data)
    class(obj_to_return) <- "glim_object"
    return(obj_to_return)
  }
}

#' Probability to Possibility Mapping for Logistic Regression
#'
#' @noRd
prob2poss_logis <- function(X, y, mle, samples, the_compared_theta) {
  X <- as.matrix(X)

  # Factor response: convert FIRST, before any 0/1 check. A factor compared to
  # 0 or 1 is never TRUE (it compares labels), which is what raised
  # "y vector must only comprise of zeros or ones." The second level counts as
  # success, the same convention glim() uses.
  if (is.factor(y)) {
    if (nlevels(y) != 2) {
      stop("Factor response must have exactly 2 levels")
    }
    y <- as.numeric(y == levels(y)[2])
  }
  if (is.data.frame(y)) {
    y <- as.matrix(y)
  }

  if (is.matrix(y)) {
    # Two-column successes / failures: expand to one row per Bernoulli trial.
    if (ncol(y) != 2) {
      stop(
        "When y is a matrix or data frame, it must have exactly 2 columns (successes and failures)."
      )
    }
    if (nrow(y) != nrow(X)) {
      stop("y and X lengths differ")
    }
    if (anyNA(y) || any(y < 0) || any(y %% 1 != 0)) {
      stop("Counts in y must be non-negative integers.")
    }
    successes <- y[, 1]
    failures <- y[, 2]
    rows <- seq_len(nrow(X))
    # X[c(2, 2), ] repeats row 2; drop = FALSE keeps a matrix.
    X <- rbind(
      X[rep(rows, times = successes), , drop = FALSE],
      X[rep(rows, times = failures), , drop = FALSE]
    )
    y <- c(rep(1, sum(successes)), rep(0, sum(failures)))
  } else {
    # Plain binary vector (numeric or logical).
    if (anyNA(y) || any(!(y == 0 | y == 1))) {
      stop("y vector must only comprise of zeros or ones.")
    }
    y <- as.numeric(y)
    if (length(y) != nrow(X)) {
      stop("y and X lengths differ")
    }
  }

  theta <- as.matrix(the_compared_theta) # one column per candidate theta
  samples <- as.matrix(samples) # one column per sample

  # Numerically stable log(1 + exp(eta)).
  eta <- X %*% theta
  log_term <- pmax(eta, 0) + log1p(exp(-abs(eta)))
  ll_val <- as.vector(y %*% eta) - colSums(log_term)

  eta_samps <- X %*% samples
  log_term_samps <- pmax(eta_samps, 0) + log1p(exp(-abs(eta_samps)))
  ll_val_samps <- as.vector(y %*% eta_samps) - colSums(log_term_samps)

  vapply(ll_val, function(x) mean(ll_val_samps <= x), numeric(1))
}

#' Probability to Possibility Mapping for Gamma Regression
#'
#' Will throw an error if \code{the_compared_theta} is a scalar.
#'
#' @noRd
prob2poss_gamma <- function(X, y, mle, samples, the_compared_theta) {
  if (any(y < 0)) {
    stop("y must be non-negative")
  }
  if (is.data.frame(X)) {
    X <- as.matrix(X)
  }
  eta <- X %*% samples
  initial_coefs <- coef(lm(log(y) ~ X - 1))
  est_shape <- 1 / mle_estimate_dispersion_gamma(y, exp(X %*% mle), length(mle))

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
  if (any(y <= 0)) {
    stop("y cannot be negative or zero")
  }
  if (is.vector(mu)) {
    return(compute_invgauss_ll(y, mu, gamma_val))
  } else {
    return(compute_invgauss_ll_mat(y, mu, gamma_val))
  }
}

#' Compute logistic Log-Likelihood
#' @noRd
compute_logistic_ll_r <- function(X, y, beta_vals) {
  if (any(!(y == 0 | y == 1))) {
    stop("y is not zeros or ones")
  }
  if (is.vector(beta_vals)) {
    return(compute_logistic_ll(X, y, beta_vals))
  } else {
    return(compute_logistic_ll_mat(X, y, beta_vals))
  }
}


#' Probability to Possibility Mapping for Gaussian Regression
#'
#' @noRd
prob2poss_gaussian <- function(X, y, mle, samples, the_compared_theta) {
  if (is.data.frame(X) | is.vector(X)) {
    X <- as.matrix(X)
  }

  eta <- X %*% samples

  # Assuming identity link for Gaussian by default
  mle_coefs <- mle
  est_sd <- est_dispersion_normal(y, X %*% mle_coefs, ncol(X)) # mle
  ll_val_samps <- as.vector(compute_gaussian_ll_r(y, eta, sigma = est_sd))
  ll_val <- as.vector(compute_gaussian_ll_r(y, X %*% the_compared_theta, sigma = est_sd))

  sapply(ll_val, function(x) sum(ll_val_samps <= x) / length(ll_val_samps))
}


#' Probability to Possibility Mapping for Inverse Gaussian Regression
#'
#' Will throw an error if \code{the_compared_theta} is a scalar.
#'
#' @noRd
prob2poss_invgauss <- function(X, y, mle, samples, the_compared_theta) {
  if (any(y <= 0)) {
    stop("y must be strictly positive")
  }
  X <- as.matrix(X)
  est_dispersion <- mle_estimate_dispersion_inv_gauss(y, mean(y))

  # Log-lik per column of `betas`; -Inf where any eta <= 0 (outside parameter space)
  ll_cols <- function(betas) {
    eta <- X %*% as.matrix(betas)
    ll <- rep(-Inf, ncol(eta))
    ok <- colSums(!is.finite(eta) | eta <= 0) == 0
    if (any(ok)) {
      ll[ok] <- as.vector(compute_invgauss_ll_r(
        y,
        sqrt(1 / eta[, ok, drop = FALSE]),
        gamma_val = est_dispersion
      ))
    }
    ll
  }

  ll_val_samps <- ll_cols(samples)
  ll_val_samps <- ll_val_samps[is.finite(ll_val_samps)] # keep draws inside the parameter space
  if (length(ll_val_samps) == 0L) {
    stop("No sampled coefficient vector gives a positive linear predictor (eta > 0).")
  }
  ll_val <- ll_cols(the_compared_theta) # invalid theta -> -Inf -> possibility 0

  vapply(ll_val, function(x) mean(ll_val_samps <= x), numeric(1))
}

#' Radial C++ Bridge Function
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
#' @param b_val Hyperparameter `b`, the step-size decay. Should be between \[.5 and 1\]
#' @param n_grid_evals Resolution of grid
#' @param num_omp_threads Number of threads
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
  n_grid_evals,
  num_omp_threads
) {
  output_samples <- radial_code(
    num_samps,
    X,
    y,
    ll_mle_original_data,
    mle_coefs,
    eJ$vectors,
    eJ$values,
    family,
    dispersion,
    m,
    tol,
    max_it,
    a_val,
    b_val,
    num_omp_threads,
    base_seed = 0
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
    trim = FALSE,
    n_grid_evals = n_grid_evals,
    m = m,
    parallel = parallel
  )
  return(list(output_samples = output_samples, betas_evaluated = grid))
}

#' Probability to Possibility Mapping for Poisson Regression
#'
#' @noRd
prob2poss_poisson <- function(X, y, mle, samples, the_compared_theta) {
  if (!is.numeric(y) || anyNA(y) || any(y < 0) || any(y != floor(y))) {
    stop("y must be non-negative integers")
  }
  X <- as.matrix(X)
  p <- ncol(X)

  samples <- as.matrix(samples)
  if (nrow(samples) != p && ncol(samples) == p) {
    samples <- t(samples) # rows were samples -> columns
  }
  if (nrow(samples) != p) {
    stop(sprintf("samples must have %d rows or columns (one per coefficient)", p))
  }

  theta <- the_compared_theta
  theta <- if (is.null(dim(theta))) matrix(theta, nrow = p) else as.matrix(theta)
  if (nrow(theta) != p && ncol(theta) == p) {
    theta <- t(theta)
  }
  if (nrow(theta) != p) {
    stop(sprintf("the_compared_theta must have %d coefficients", p))
  }

  ll_val_samps <- as.vector(compute_poisson_ll_r(y, X %*% samples))
  ll_val <- as.vector(compute_poisson_ll_r(y, X %*% theta))
  sapply(ll_val, function(x) mean(ll_val_samps <= x))
}

.ext_points <- function(g, col) {
  b <- rbind(
    if (NROW(g$extended_in_neg_betas[[col]])) g$extended_in_neg_betas[[col]],
    if (NROW(g$extended_in_pos_betas[[col]])) g$extended_in_pos_betas[[col]]
  )
  p <- c(g$extended_in_neg_plaus[[col]], g$extended_in_pos_plaus[[col]])
  list(betas = b, plaus = as.vector(p))
}

# TODO figure out 2d marginalization for the elliptical case (if at all)
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
  # Radial and elliptical methods will always return margin
  args <- list(...)
  alpha <- -1
  if ("alpha" %in% names(args)) {
    if (is.numeric(args$alpha) && length(args$alpha) == 1 && args$alpha < 1 && args$alpha > 0) {
      alpha <- args$alpha
    } else {
      stop("Alpha cutoff invalid")
    }
  }

  if ("margin" %in% names(x)) {
    data <- x$margin

    num_predictors <- ncol(data[["margins"]][[1]]$beta_for_margin)
    grid_cols <- ceiling(sqrt(num_predictors))
    grid_rows <- ceiling(num_predictors / grid_cols)

    old_par <- par(no.readonly = TRUE)
    on.exit(par(old_par))

    # Scale text size for grid density
    cex_val <- max(0.6, 1 - 0.08 * max(grid_rows, grid_cols))
    par(
      mfrow = c(grid_rows, grid_cols),
      mar = c(3.5, 3.5, 2, 1),
      oma = c(1, 1, 3, 0),
      mgp = c(2, 0.7, 0),
      cex.lab = cex_val,
      cex.axis = cex_val
    )

    for (col in 1:num_predictors) {
      xs <- data[["margins"]][[col]][["beta_for_margin"]][, col]
      ys <- data[["margins"]][[col]][["plaus_for_margin"]]
      ord <- order(xs)
      xs <- xs[ord]
      ys <- ys[ord]
      xlab <- ""
      family <- ""
      if ("grid_data" %in% names(x)) {
        xlab <- colnames(x[["grid_data"]]$X)[col]
        family <- x[["grid_data"]]$family
      } else if ("elliptical_data" %in% names(x)) {
        xlab <- colnames(x[["margin"]][["extra_info"]]$X)[col]
        family <- x[["margin"]][["extra_info"]]$family
      } else if ("radial_data" %in% names(x)) {
        xlab <- colnames(x[["margin"]][["extra_info"]]$X)[col]
        family <- x[["margin"]][["extra_info"]]$family
      } else {
        warning("Something messed up")
      }
      plot.default(xs, ys, type = "l", xlab = xlab, ylab = "Possibility")

      if (alpha != -1) {
        abline(h = alpha, col = "red", lty = 3)
      }
      grid(nx = NULL, ny = NULL, col = "lightgrey", lty = "dashed")
    }

    fam_title <- if (!is.null(family) && nchar(family) > 0) {
      paste0(toupper(substr(family, 1, 1)), substr(family, 2, nchar(family)))
    } else {
      ""
    }
    mtext(
      text = paste0(fam_title, " Possibility contours (marginalized)"),
      side = 3,
      outer = TRUE,
      line = 1, # Centered in the top outer margin
      cex = 1.2,
      font = 2
    )
  } else if ("grid_data" %in% names(x)) {
    g <- x$grid_data
    betas <- g$betas_evaluated
    poss <- as.vector(g$possibilities)
    family <- g$family
    num_predictors <- ncol(betas)
    col_names <- colnames(betas)
    if (is.null(col_names)) {
      col_names <- paste0("Beta_", 1:num_predictors)
    }
    alpha_line <- if (alpha != -1) alpha else 0.05

    # Stack the grid with all extension points (each is a full beta vector)
    ext <- do.call(
      rbind,
      lapply(seq_len(num_predictors), function(k) {
        e <- .ext_points(g, k)
        if (is.null(e$betas)) {
          return(NULL)
        }
        cbind(e$betas, plaus = e$plaus)
      })
    )
    ext_betas <- if (!is.null(ext)) ext[, seq_len(num_predictors), drop = FALSE] else NULL

    # One shared range per coefficient, covering the grid and its extensions
    lims <- lapply(seq_len(num_predictors), function(k) {
      vals <- betas[, k]
      if (!is.null(ext_betas)) {
        vals <- c(vals, ext_betas[, k])
      }
      range(vals[is.finite(vals)])
    })

    old_par <- par(no.readonly = TRUE)
    on.exit(par(old_par))
    cex_val <- max(0.6, 1.15 - 0.08 * num_predictors)
    par(
      mfrow = c(num_predictors, num_predictors),
      mar = c(3, 3, 1.5, 0.8),
      oma = c(1, 1, 3.5, 0),
      mgp = c(1.8, 0.5, 0),
      cex.lab = cex_val * 1.2,
      cex.axis = cex_val * 1.2
    )

    max_or_na <- function(v) {
      v <- v[!is.na(v)]
      if (length(v) == 0) NA_real_ else max(v)
    }

    for (row in 1:num_predictors) {
      for (col in 1:num_predictors) {
        if (row == col) {
          # DIAGONAL: profile uses grid AND extension points
          own <- .ext_points(g, col)
          b_d <- betas[, col]
          p_d <- poss
          if (!is.null(own$betas)) {
            b_d <- c(b_d, own$betas[, col])
            p_d <- c(p_d, own$plaus)
          }
          max_plaus <- tapply(p_d, b_d, max_or_na)
          beta_vals <- as.numeric(names(max_plaus))
          ok <- is.finite(beta_vals) & is.finite(max_plaus)
          if (any(ok)) {
            ord <- order(beta_vals[ok])
            plot.default(
              beta_vals[ok][ord],
              max_plaus[ok][ord],
              type = "l",
              xlab = col_names[col],
              ylab = "Profile Plausibility",
              xlim = lims[[col]],
              ylim = c(0, 1)
            )
            abline(h = alpha_line, col = "red", lty = 3)
            grid(nx = NULL, ny = NULL, col = "lightgrey", lty = "dashed")
          } else {
            plot.default(1, type = "n", axes = FALSE, xlab = "", ylab = "", main = "No Data")
          }
        } else {
          # OFF-DIAGONAL: contour from the grid, extension path overlaid
          z_mat <- tapply(poss, list(betas[, col], betas[, row]), max_or_na)
          x_vals <- as.numeric(rownames(z_mat))
          y_vals <- as.numeric(colnames(z_mat))
          x_ord <- order(x_vals)
          y_ord <- order(y_vals)
          x_vals <- x_vals[x_ord]
          y_vals <- y_vals[y_ord]
          z_mat <- z_mat[x_ord, y_ord, drop = FALSE]
          z_mat[is.na(z_mat)] <- 0 # cells outside the trimmed ellipse

          if (length(x_vals) >= 2 && length(y_vals) >= 2) {
            contour(
              x_vals,
              y_vals,
              z_mat,
              levels = c(0.05, 0.1, 0.25, 0.5, 0.75, 0.9),
              xlim = lims[[col]],
              ylim = lims[[row]],
              drawlabels = TRUE,
              labcex = cex_val,
              xlab = col_names[col],
              ylab = col_names[row],
              main = ""
            )
            if (!is.null(ext)) {
              points(
                ext_betas[, col],
                ext_betas[, row],
                pch = 20,
                cex = 0.5,
                col = ifelse(ext[, "plaus"] > alpha_line, "blue", "grey60")
              )
            }
            grid(nx = NULL, ny = NULL, col = "lightgrey", lty = "dashed")
          } else {
            plot.default(1, type = "n", axes = FALSE, xlab = "", ylab = "", main = "No Data")
          }
        }
      }
    }

    fam_title <- if (!is.null(family) && nchar(family) > 0) {
      paste0(toupper(substr(family, 1, 1)), substr(family, 2, nchar(family)))
    } else {
      ""
    }
    mtext(
      paste0(fam_title, " Possibility contours (marginalized)"),
      side = 3,
      outer = TRUE,
      line = 1,
      cex = 1.2,
      font = 2
    )
  }
}


# =============================================================================
# Marginal intervals from a grid fit
# =============================================================================
#
# THE IDEA
# glim() evaluates a grid in which every row is a full candidate coefficient
# vector (b0, b1, ...) and assigns each row a plausibility. A value of
# coefficient j counts as plausible if AT LEAST ONE grid row with that value of
# b_j has plausibility above alpha. In other words, we project the plausible
# region onto the b_j axis: the other coefficients are free to take whatever
# values make this b_j look best. The interval for coefficient j is the
# smallest and largest plausible value of b_j.
#
# .grid_bounds() does the work. It is shared by print() and confint() so the
# two can never disagree.

# Marginal bounds from the grid evaluation.
#   g:     the `grid_data` list stored inside a glim_object
#   alpha: a value is plausible if its profile plausibility exceeds alpha
# Returns list(lower, upper), each a numeric vector with one entry per
# coefficient (NA where the bound is unknown).
.grid_bounds <- function(g, alpha) {
  betas <- g$betas_evaluated # one row per grid point, one column per coefficient
  poss <- as.vector(g$possibilities) # one plausibility per grid point
  p <- ncol(betas)
  lower <- upper <- rep(NA_real_, p)

  # max() that returns NA, not -Inf plus a warning, when a group is all NA.
  safe_max <- function(v) if (all(is.na(v))) NA_real_ else max(v, na.rm = TRUE)

  # "Is a at the same place as b?" Exact == is unreliable for floating point,
  # so use a small relative tolerance.
  at_edge <- function(a, b) abs(a - b) <= 1e-8 * max(1, abs(b))

  # Bound reachable through a stored profile extension.
  # If the plausible region touches the edge of the grid, it may continue past
  # it. glim() can then have walked outward from the edge (the
  # extended_in_{neg,pos}_{betas,plaus} entries) until plausibility dropped
  # below alpha. Arguments:
  #   ext_b, ext_p: extension points (matrix, one column per coefficient) and
  #                 their plausibilities
  #   col:          which coefficient we are bounding
  #   edge:         the grid-edge value, used as the fallback bound
  #   side:         "lower" or "upper"
  extension_bound <- function(ext_b, ext_p, col, edge, side) {
    # No extension stored (e.g. directly_calc_marginals = TRUE): the bound is
    # UNKNOWN. Return NA so it is not presented as if it were the grid edge.
    if (is.null(ext_b) || NROW(ext_b) == 0) {
      return(NA_real_)
    }
    vals <- as.matrix(ext_b)[, col]

    # Keep only extension points that are themselves plausible. The last point
    # of an extension may be the first one that fell below alpha.
    if (!is.null(ext_p)) {
      ext_p <- as.vector(ext_p)
      if (length(ext_p) == length(vals)) {
        vals <- vals[is.finite(ext_p) & ext_p > alpha]
      }
    }
    vals <- vals[is.finite(vals)]

    # The extension ran but nothing beyond the edge is plausible, so the true
    # bound lies between the edge and the first extension point. The edge is
    # the best available estimate.
    if (length(vals) == 0) {
      return(edge)
    }

    # Otherwise the bound is the furthest plausible extension point (the edge
    # is included so the result can never be tighter than the grid).
    if (side == "lower") min(c(vals, edge)) else max(c(vals, edge))
  }

  for (col in seq_len(p)) {
    # PROFILE MAXIMUM. A given value of b_j usually appears in many grid rows,
    # paired with different values of the other coefficients. ave() gives every
    # row the best plausibility among all rows that share its b_j value.
    # (This replaces tapply() + as.numeric(names()), which round-trips values
    # through character names and can lose precision, making the edge
    # comparison below misfire.)
    prof <- ave(poss, betas[, col], FUN = safe_max)

    # Rows whose b_j value is plausible. Their b_j values are the plausible set.
    ok <- is.finite(betas[, col]) & is.finite(prof) & prof > alpha
    if (!any(ok)) {
      next # nothing plausible: leave both bounds NA
    }
    valid <- betas[ok, col]
    grid_min <- min(valid) # candidate lower bound
    grid_max <- max(valid) # candidate upper bound

    # LOWER BOUND.
    # If grid_min is NOT at the smallest evaluated value, plausibility fell
    # below alpha inside the grid, so grid_min is a real boundary.
    # If it IS at the edge, the region may continue, so consult the extension.
    lower[col] <- if (!at_edge(grid_min, min(betas[, col]))) {
      grid_min
    } else {
      extension_bound(
        g$extended_in_neg_betas[[col]],
        g$extended_in_neg_plaus[[col]],
        col,
        grid_min,
        "lower"
      )
    }

    # UPPER BOUND. Same logic, mirrored (positive extensions, maximum).
    upper[col] <- if (!at_edge(grid_max, max(betas[, col]))) {
      grid_max
    } else {
      extension_bound(
        g$extended_in_pos_betas[[col]],
        g$extended_in_pos_plaus[[col]],
        col,
        grid_max,
        "upper"
      )
    }
  }
  list(lower = lower, upper = upper)
}


#' Returns a matrix with confidence interval values
#'
#' Same numbers as used in print(), modulo stochastic noise
#'
#' @param object A 'glim_object' from 'glim()'
#' @param parm A vector a strings specifying which variables to select
#' @param level 1-alpha
#' @param ... Unused
#' @return A matrix containing the confidence/credible interval
#' @export
confint.glim_object <- function(object, parm = NULL, level = .95, ...) {
  alpha <- 1 - level
  if ("grid_data" %in% names(object)) {
    # Grid fit: everything is stored inside grid_data (not at the top level).
    g <- object$grid_data
    b <- .grid_bounds(g, alpha)

    # One row per coefficient, columns = lower and upper bound.
    out <- cbind(b$lower, b$upper)

    # Column labels are the tail percentages, e.g. "2.5 %" and "97.5 %".
    colnames(out) <- c(
      sprintf("%.1f %%", (alpha / 2) * 100),
      sprintf("%.1f %%", (1 - alpha / 2) * 100)
    )
    rownames(out) <- colnames(g$X)
  } else if ("margin" %in% names(object)) {
    # Extract bounds from object$margin
    data <- object[["margin"]]
    X <- data[["extra_info"]]$X

    lower_bound <- rep(NA_real_, ncol(X))
    upper_bound <- rep(NA_real_, ncol(X))
    for (col in seq_len(ncol(X))) {
      beta_vals <- data[["margins"]][[col]]$beta_for_margin
      lower_bound[col] <- min(beta_vals[, col])
      upper_bound[col] <- max(beta_vals[, col])
    }

    out <- cbind(lower_bound, upper_bound)
    colnames(out) <- c(
      sprintf("%.1f %%", (alpha / 2) * 100),
      sprintf("%.1f %%", (1 - alpha / 2) * 100)
    )
    rownames(out) <- colnames(X)
  } else {
    stop("Unrecognized glim_object structure")
  }
  if (!is.null(parm)) {
    out <- out[parm, , drop = FALSE]
  }
  return(out)
}

#' Print
#'
#' Provides marginal confidence/credible regions at a specified alpha level. Can be changed by (say) alpha = .10. Produces a value of NA whenever the bound is not known (Likely the automatic grid was too small -- can be expanded through the betas argument in glim())
#'
#' @param x A 'glim_object' from 'glim()'
#' @param alpha Supports differing alpha levels, although will not provide correct results if initial model specified a higher alpha level.
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
#' @param x a 'glim_object'
#' @param alpha A specified alpha level cut
#' @param ... Unused
#' @export
print.glim_object <- function(x, alpha = .05, ...) {
  cat(sprintf("%.0f%% (marginal) confidence and credible region:\n", (1 - alpha) * 100))
  cat("------------------------------------------------------------------------\n")

  if ("margin" %in% names(x)) {
    data <- x[["margin"]]
    X <- data[["extra_info"]]$X
    mle_coefs <- data[["extra_info"]]$mle_coefs

    lower_bound <- rep(NA_real_, ncol(X))
    upper_bound <- rep(NA_real_, ncol(X))
    for (col in seq_len(ncol(X))) {
      beta_vals <- data[["margins"]][[col]]$beta_for_margin
      lower_bound[col] <- min(beta_vals[, col])
      upper_bound[col] <- max(beta_vals[, col])
    }
    row_names <- colnames(X)
  } else if ("grid_data" %in% names(x)) {
    g <- x[["grid_data"]]
    b <- .grid_bounds(g, alpha)
    lower_bound <- b$lower
    upper_bound <- b$upper
    mle_coefs <- g$mle_coefs
    row_names <- colnames(g$betas_evaluated)
  } else {
    warning("Object does not seem to have a valid structure")
    return(invisible(x))
  }

  summary_table <- data.frame(
    Estimate = sprintf("%.4f", mle_coefs),
    Lower = sprintf("%.4f", lower_bound),
    Upper = sprintf("%.4f", upper_bound),
    row.names = row_names
  )

  colnames(summary_table) <- c(
    "MLE",
    paste0("Lower ", (1 - alpha) * 100, "%"),
    paste0("Upper ", (1 - alpha) * 100, "%")
  )

  print(summary_table, quote = FALSE)
  cat("------------------------------------------------------------------------\n")
  invisible(x)
}


# # Note: Initially written by the author, Gemini helped with formatting of output

# #' Print
# #'
# #' Provides marginal confidence/credible regions at a specified alpha level. Can be changed by (say) alpha = .10. Produces a value of NA whenever the bound is not known (Likely the automatic grid was too small -- can be expanded through the betas argument in glim())
# #'
# #' @param x A 'glim_object' from 'glim()'
# #' @param ... Supports differing alpha levels. EX: alpha = .05
# #' @return Print output
# #' @examples
# #' \dontrun{
# #' fit <- glim(y ~ x, data = dat, family = "gaussian")
# #'
# #' # Print the 95% confidence and credible region (default)
# #' print(fit)
# #'
# #' # Print at a different alpha level
# #' print(fit, alpha = .10)
# #' }
# #' @export
# print.glim_object <- function(x, alpha = .05, ...) {
#   args <- list(...)

#   cat(sprintf("%.0f%% (marginal) confidence and credible region:\n", (1 - alpha) * 100))
#   cat("------------------------------------------------------------------------\n")
#   if ("margin" %in% names(x)) {
#     data <- x[["margin"]]
#     lower_bound <- rep(NA_real_, ncol(data[["extra_info"]]$X))
#     upper_bound <- rep(NA_real_, ncol(data[["extra_info"]]$X))
#     mle_coefs <- data[["extra_info"]]$mle_coefs
#     for (col in 1:ncol(data[["extra_info"]]$X)) {
#       beta_vals <- data[["margins"]][[col]]$beta_for_margin
#       lower_bound[col] <- min(beta_vals[, col])
#       upper_bound[col] <- max(beta_vals[, col])
#     }
#     # Build the structured data frame
#     summary_table <- data.frame(
#       Estimate = sprintf("%.4f", mle_coefs),
#       Lower = sprintf("%.4f", lower_bound),
#       Upper = sprintf("%.4f", upper_bound),
#       row.names = colnames(data[["extra_info"]]$X)
#     )

#     # Rename columns to show the interval width dynamically
#     colnames(summary_table) <- c(
#       "MLE",
#       paste0("Lower ", (1 - alpha) * 100, "%"),
#       paste0("Upper ", (1 - alpha) * 100, "%")
#     )

#     # Print the formatted table cleanly
#     print(summary_table, quote = FALSE)
#     cat("------------------------------------------------------------------------\n")
#     invisible(x)
#   } else if ("grid_data" %in% names(x)) {
#     # margins not found, so either in elliptical_data, radial_data, grid_data
#     grid_object <- x[["grid_data"]]
#     betas <- grid_object$betas_evaluated
#     poss <- grid_object$possibilities
#     family <- grid_object$family
#     mle_coefs <- grid_object$mle_coefs

#     row_names <- colnames(betas)
#     ll_mle_original_data <- numeric(ncol(betas))
#     lower_bound <- rep(NA_real_, ncol(betas))
#     upper_bound <- rep(NA_real_, ncol(betas))

#     for (col in 1:ncol(betas)) {
#       max_plaus <- tapply(poss, betas[, col], max, na.rm = TRUE)

#       beta_vals <- as.numeric(names(max_plaus))
#       valid <- all(is.finite(beta_vals)) && all(is.finite(max_plaus))
#       alpha_cut <- valid && (max_plaus > alpha)
#       valid_betas <- beta_vals[alpha_cut]

#       if (length(valid_betas) > 0) {
#         # ---------- LOWER BOUND ----------
#         grid_min <- min(valid_betas)
#         if (grid_min != min(betas[, col])) {
#           lower_bound[col] <- grid_min
#         } else {
#           # Boundary hit: Check if negative extensions exist for this parameter
#           if (
#             !is.null(x$extended_in_neg_betas) &&
#               !is.null(x$extended_in_neg_betas[[col]]) &&
#               nrow(x$extended_in_neg_betas[[col]]) > 0
#           ) {
#             # The lowest value reached in the extension
#             lower_bound[col] <- min(x$extended_in_neg_betas[[col]][, col])
#           } else {
#             lower_bound[col] <- NA_real_
#           }
#         }

#         # ---------- UPPER BOUND ----------
#         grid_max <- max(valid_betas)
#         if (grid_max != max(betas[, col])) {
#           upper_bound[col] <- grid_max
#         } else {
#           # Boundary hit: Check if positive extensions exist for this parameter
#           if (
#             !is.null(x$extended_in_pos_betas) &&
#               !is.null(x$extended_in_pos_betas[[col]]) &&
#               nrow(x$extended_in_pos_betas[[col]]) > 0
#           ) {
#             # The highest value reached in the extension
#             upper_bound[col] <- max(x$extended_in_pos_betas[[col]][, col])
#           } else {
#             upper_bound[col] <- NA_real_
#           }
#         }
#       }
#     }
#     # Build the structured data frame
#     summary_table <- data.frame(
#       Estimate = sprintf("%.4f", mle_coefs),
#       Lower = sprintf("%.4f", lower_bound),
#       Upper = sprintf("%.4f", upper_bound),
#       row.names = row_names
#     )

#     # Rename columns to show the interval width dynamically
#     colnames(summary_table) <- c(
#       "MLE",
#       paste0("Lower ", (1 - alpha) * 100, "%"),
#       paste0("Upper ", (1 - alpha) * 100, "%")
#     )

#     # Print the formatted table cleanly
#     print(summary_table, quote = FALSE)
#     cat("------------------------------------------------------------------------\n")
#     invisible(x)
#   } else {
#     warning("Object does not seem to have a valid structure")
#   }
# }

#' Extract the log likelihood at the MLE
#'
#' Extracts the logLik at the MLE value. Will be very similar to glm's implementation, but will be minor differences based on the implementation to arrive at the MLE. Most notably, since the gamma and inverse gaussian case use a MLE estimator (instead of glm's default Pearson MOM estimator) for the dispersion parameter, this will provide different results in this case.
#'
#' @param object A 'glim_object' from 'glim()'
#' @param ... Additional arguments
#' @return A logLik class object
#' @export
logLik.glim_object <- function(object, ...) {
  if ("grid_data" %in% names(object)) {
    val <- object$grid_data$logLik
    attr(val, "df") <- length(object$grid_data$mle_coefs)
    attr(val, "nobs") <- length(object$grid_data$y)
    class(val) <- "logLik"
    return(val)
  } else if ("margin" %in% names(object)) {
    val <- object$margin$extra_info$ll_mle_original_data
    attr(val, "df") <- length(object$mle_coefs)
    attr(val, "nobs") <- length(object$y)
    class(val) <- "logLik"
    return(val)
  }
}

#' Extract the MLE coefficients
#'
#' Extract the MLE coefficients
#'
#' @param object A 'glim_object' from 'glim()'
#' @param ... Additional arguments
#' @return A vector of MLE coefficients
#' @export
coef.glim_object <- function(object, ...) {
  if ("grid_data" %in% names(object)) {
    return(object$grid_data$mle_coefs)
  } else if ("margin" %in% names(object)) {
    return(object$margin$extra_info$mle_coefs)
  }
}

# #' Extract the log likelihood at the MLE
# #'
# #' Extracts the logLik at the MLE value. Will be very similar to glm's implementation, but will be minor differences based on the implementation to arrive at the MLE. Most notably, since the gamma and inverse gaussian case use a MLE estimator (instead of glm's default Pearson MOM estimator) for the dispersion parameter, this will provide different results in this case.
# #'
# #' @param object A 'glim_object' from 'glim()'
# #' @param ... Additional arguments
# #' @return A matrix containing the confidence/credible interval
# #' @export
# confint.glim_object <- function(object, alpha = .05) {
#   if ("grid_data" %in% names(object)) {
#     row_names <- colnames(object$X)
#     lower_bound <- numeric(ncol(object$X))
#     upper_bound <- numeric(ncol(object$X))

#     for (col in 1:ncol(object$X)) {
#       max_plaus <- tapply(object$possibilities, object$betas[, col], max, na.rm = TRUE)

#       beta_vals <- as.numeric(names(max_plaus))
#       valid_alpha_cut <- is.finite(beta_vals) & is.finite(max_plaus) & (max_plaus > alpha)
#       valid_betas <- beta_vals[valid_alpha_cut]

#       # Save results into our vectors
#       lower_bound[col] <- if (length(valid_betas) > 0) min(valid_betas) else NA
#       upper_bound[col] <- if (length(valid_betas) > 0) max(valid_betas) else NA
#     }
#     out <- matrix(c(lower_bound, upper_bound), ncol = 2)
#     lower_a <- (alpha / 2) * 100
#     upper_a <- (1 - (alpha / 2)) * 100
#     colnames(out) <- c(sprintf("%.1f %%", lower_a), sprintf("%.1f %%", upper_a))
#     rownames(out) <- row_names
#     return(out)
#   } else if ("margin" %in% names(object)) {
#     return(object$margin)
#   }
#   stop("Unrecognized glim_object structure")
# }

#' Returns the variance-covariance at the MLE
#'
#' Extracts the variance-covariance matrix from the glim object.
#'
#' @param object A 'glim_object' from 'glim()'
#' @param ... Additional arguments
#' @return A logLik class object
#' @export
vcov.glim_object <- function(object, ...) {
  if ("margin" %in% names(object)) {
    return(object$margin$extra_info$vcov)
  } else {
    return(object$vcov)
  }
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
#' @param mle_coefs a vector of MLE coefficients
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
compute_ll <- function(family, y, X, betas, mle_coefs = NULL) {
  if (is.matrix(betas)) {
    betas <- t(betas)
  }
  ll <- NULL
  if (family == "gaussian") {
    if (is.null(mle_coefs)) {
      stop("mle_coefs must be provided when family is 'gaussian'")
    }
    ll <- compute_gaussian_ll_r(
      y,
      X %*% betas,
      sqrt(est_dispersion_normal(y, X %*% mle_coefs, ncol(X)))
    )
    # need to compute the dispersion, which requires the MLE
  } else if (family == "gamma") {
    if (is.null(mle_coefs)) {
      stop("mle_coefs must be provided when family is 'gamma'")
    }
    ll <- compute_gamma_ll_r(
      y = y,
      X %*% betas,
      shape = 1 / mle_estimate_dispersion_gamma(y, exp(X %*% mle_coefs), ncol(X))
    )
  } else if (family == "poisson") {
    ll <- compute_poisson_ll_r(y, X %*% betas)
  } else if (family == "binomial") {
    ll <- compute_logistic_ll_r(X, y, betas)
  } else if (family == "inverse.gaussian") {
    # Again, note that 1/eta will be element-wise
    eta <- X %*% betas
    # invgauss uses (y, mu) instead of (y, eta)
    ll <- compute_invgauss_ll_r(
      y,
      sqrt(1 / eta),
      gamma_val = mle_estimate_dispersion_inv_gauss(y = y, mean(y))
    )
  } else {
    stop("family not defined")
  }
  if (!all(is.finite(ll))) {
    return(-Inf)
  }
  return(ll)
}

#' Probability to Possibility Transform
#'
#' From a set of samples (of beta values), return the possibility at any new beta value.
#' @param family Either "guassian", "gamma", "inverse.gaussian", "poisson", "binomial"
#' @param X Design X matrix
#' @param y Vector of y values
#' @param mle Vector of mle values
#' @param samples Each column is a new beta vector sample
#' @param the_compared_theta Each row is vector of betas which we want the possibility
#' @examples
#' \dontrun{
#' X <- cbind(intercept = 1, x1 = c(4, 3.3, 5))
#' y <- rgamma(3, shape = 2, scale = 1.4)
#'
#' samples <- glim(y ~ X, family = "gamma", approx = TRUE)$samples
#'
#' beta_1_grid <- seq(-2, 2, by = .2)
#' beta_2_grid <- seq(-1, 2, by = .2)
#' evaluated_new_theta <- expand.grid(beta_1_grid, beta_2_grid)
#' prob2poss("gamma", X, y, samples, evaluated_new_theta)
#' }
#' @export
prob2poss <- function(family, X, y, mle, samples, the_compared_theta) {
  if (family == "gaussian") {
    p2p_function <- prob2poss_gaussian
  } else if (family == "gamma") {
    p2p_function <- prob2poss_gamma
  } else if (family == "inverse.gaussian") {
    p2p_function <- prob2poss_invgauss
  } else if (family == "poisson") {
    p2p_function <- prob2poss_poisson
  } else if (family == "binomial") {
    p2p_function <- prob2poss_logis
  } else {
    stop("Family is incorrectly specified")
  }
  if (is.data.frame(the_compared_theta)) {
    the_compared_theta <- as.matrix(the_compared_theta)
  }
  if (!is.vector(the_compared_theta)) {
    if (ncol(the_compared_theta) != ncol(X)) {
      stop(
        "Dimension mismatch. Check that the_compared_theta has each 'row' corresponding to a new parameter value"
      )
    }
  }
  if (length(y) != nrow(X)) {
    stop("Dimension mismatch of X")
  }
  out <- p2p_function(X, y, mle, samples, t(the_compared_theta))
  return(out)
}


# ============================================================================
# glim_prof.R -- R side of the profiler. Source/include this in the package.
# Usage:
#   glim_prof_start("prof_out")
#   fit <- glim(...)                 # your normal call
#   glim_prof_report()
#   glim_prof_stop()
# Everything is a no-op unless glim_prof_start() has been called.
# ============================================================================

.glim_prof <- new.env(parent = emptyenv())
.glim_prof$on <- FALSE
.glim_prof$dir <- NULL
.glim_prof$last <- list() # most recent duration per label
.glim_prof$tot <- list() # cumulative duration per label
.glim_prof$cnt <- list() # call counts per label

#' @export
glim_prof_start <- function(dir = "glim_prof") {
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  dir <- normalizePath(dir)
  for (f in list.files(dir, pattern = "\\.csv$", full.names = TRUE)) {
    file.remove(f)
  }
  .glim_prof$dir <- dir
  .glim_prof$on <- TRUE
  .glim_prof$last <- list()
  .glim_prof$tot <- list()
  .glim_prof$cnt <- list()
  prof_cpp_start(dir)
  invisible(dir)
}

#' @export
glim_prof_stop <- function() {
  prof_cpp_stop()
  .glim_prof$on <- FALSE
  invisible(NULL)
}

# Time an expression, accumulate by label, return its value.
.glim_timed <- function(label, expr) {
  if (!isTRUE(.glim_prof$on)) {
    return(expr)
  }
  t0 <- proc.time()[["elapsed"]]
  val <- expr # forces the promise
  dt <- proc.time()[["elapsed"]] - t0
  .glim_prof$last[[label]] <- dt
  prev <- .glim_prof$tot[[label]]
  .glim_prof$tot[[label]] <- (if (is.null(prev)) 0 else prev) + dt
  prevn <- .glim_prof$cnt[[label]]
  .glim_prof$cnt[[label]] <- (if (is.null(prevn)) 0L else prevn) + 1L
  val
}

.glim_last <- function(label) {
  v <- .glim_prof$last[[label]]
  if (is.null(v)) NA_real_ else v
}

# Append one row to <dir>/<event>.csv (header written on first use).
.glim_log <- function(event, ...) {
  if (!isTRUE(.glim_prof$on)) {
    return(invisible(NULL))
  }
  f <- file.path(.glim_prof$dir, paste0(event, ".csv"))
  new <- !file.exists(f)
  row <- as.data.frame(list(...), stringsAsFactors = FALSE)
  utils::write.table(row, f, sep = ",", append = !new, col.names = new, row.names = FALSE)
  invisible(NULL)
}

#' Fits GLIM (Raw Implementation)
#'
#' Evaluates possibility for beta/dispersion values.
#'
#' @param X Input predictor matrix. If a data frame is provided, it will be coerced to a model matrix.
#' @param y Response vector.
#' @param family A string indicating the error distribution. Options include `"gaussian"`, `"poisson"`, `"gamma"`, `"binomial"`.
#' @param betas A grid of beta values to evaluate, where each column is a new beta vector. Not required if `approx = TRUE`.
#' @param mle_coefs Maximum likelihood estimates of the coefficients.
#' @param ll_mle_original_data The log-likelihood value at the maximum likelihood estimates.
#' @param m Number of evaluations per beta (number of samples).
#' @param parallel Logical indicating whether to run in parallel (primarily for debugging).
#' @param approx Logical indicating whether to use the elliptical approximation.
#' @param ... A way to pass in .Random.seed
#' @return A matrix of evaluated possibility outputs.
#' @noRd
glim_raw <- function(
  X,
  y,
  family = "gaussian",
  betas,
  mle_coefs,
  ll_mle_original_data,
  m,
  parallel,
  approx,
  show_progress,
  base_seed = 31415L
) {
  num_omp_threads <- 1
  if (parallel) {
    cores <- parallel::detectCores(logical = FALSE)
    if (is.na(cores)) {
      cores <- parallel::detectCores()
    }
    num_omp_threads <- max(1L, cores - 1L)
  }
  if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    old <- RhpcBLASctl::blas_get_num_procs()
    RhpcBLASctl::blas_set_num_threads(1)
    on.exit(RhpcBLASctl::blas_set_num_threads(old), add = TRUE)
  }

  out <- .glim_timed(
    "glim_raw_total",
    fit_glm_omp_cpp(
      X = X,
      y = y,
      mle_coefs = mle_coefs,
      mle_val = ll_mle_original_data,
      betas = betas,
      family = family,
      num_threads = num_omp_threads,
      m = m,
      parallel = parallel,
      approx = approx,
      show_progress = show_progress,
      base_seed = as.integer(base_seed)
    )
  )
  .glim_log(
    "glim_raw",
    n_evals = nrow(betas),
    m = m,
    parallel = parallel,
    approx = approx,
    elapsed = .glim_last("glim_raw_total")
  )
  out
}

# ---------------------------------------------------------------------------
# glim_raw: time the C++ call from the R side. Compare with wall_s in
# calls.csv; a big gap means R<->C++ overhead (argument copies, conversions).
# Change the last statement of glim_raw to:
#
#   out <- .glim_timed("glim_raw_total", fit_glm_omp_cpp(...))
#   .glim_log("glim_raw", n_evals = nrow(betas), m = m, parallel = parallel,
#             approx = approx, elapsed = .glim_last("glim_raw_total"))
#   out
# ---------------------------------------------------------------------------

# ---- generate_grid -----------------------------------------------------------
# Changes: imvar runs with a smaller m and a noise-aware tolerance, can use
# threads, and trimming returns a matrix (drop = FALSE). Default trim = TRUE:
# corners of the bounding box outside the ellipse are not worth evaluating.
generate_grid <- function(
  X,
  y,
  family,
  mle_coefs,
  eigen_vecs,
  eigen_vals,
  dispersion,
  ll_mle_original_data,
  trim = TRUE,
  n_grid_evals = 25,
  m = 1000,
  parallel = FALSE
) {
  initial_xi <- rep(1, length(mle_coefs))
  alpha_target <- 0.01

  # A rough scale factor does not need m = 1000. Keep the tolerance above the
  # Monte Carlo standard error of a proportion near alpha.
  m_imvar <- min(m, 500)
  se_alpha <- sqrt(alpha_target * (1 - alpha_target) / m_imvar)
  tol_imvar <- max(0.005, 1.5 * se_alpha)

  n_threads <- if (parallel) {
    cores <- parallel::detectCores(logical = FALSE)
    if (is.na(cores)) {
      cores <- parallel::detectCores()
    }
    max(1L, cores - 1L)
  } else {
    1L
  }

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
    generate_grid = TRUE,
    tol = tol_imvar,
    a_val = 2,
    b_val = .65,
    max_it = 25,
    parallel = parallel,
    m = m_imvar,
    base_seed = 31,
    num_threads = n_threads
  )

  q_val <- qchisq(1 - alpha_target, df = length(mle_coefs))
  base_scale <- sqrt(dispersion * q_val * (1 / eigen_vals))
  semi_axes <- base_scale * sqrt(imvar_xi)
  transformed_mat <- eigen_vecs %*% diag(as.vector(semi_axes), nrow = length(semi_axes))
  H <- sqrt(rowSums(transformed_mat^2))

  p <- length(mle_coefs)
  beta <- vector("list", p)
  n_left <- floor((n_grid_evals - 1) / 2)
  n_right <- n_grid_evals - 1 - n_left
  approx_grid_resolution <- numeric(p)
  for (i in seq_len(p)) {
    beta[[i]] <- c(
      seq(mle_coefs[i] - H[i], mle_coefs[i], length.out = n_left + 1)[-(n_left + 1)],
      mle_coefs[i],
      seq(mle_coefs[i], mle_coefs[i] + H[i], length.out = n_right + 1)[-1]
    )
    approx_grid_resolution[i] <- 2 * H[i] / n_grid_evals
  }

  betas <- as.matrix(expand.grid(beta))
  colnames(betas) <- colnames(X)

  shifted_points <- sweep(betas, 2, mle_coefs, FUN = "-")
  rotated_points <- shifted_points %*% eigen_vecs
  scaled_sq_points <- sweep(rotated_points^2, 2, semi_axes^2, FUN = "/")
  inside_indices <- rowSums(scaled_sq_points) <= 1.001

  if (!any(inside_indices)) {
    stop(
      "GLIM error: no grid points fell inside the confidence ellipse. ",
      "Try increasing 'n_grid_evals' or check that the model fit ",
      "(mle_coefs / eigen decomposition) is not degenerate.",
      call. = FALSE
    )
  }

  if (isTRUE(trim)) {
    betas <- betas[inside_indices, , drop = FALSE]
  }
  list(betas = betas, approx_grid_resolution = approx_grid_resolution)
}

# ---------------------------------------------------------------------------
# generate_grid: wrap imvar and record what it produced.
#
#   imvar_xi <- .glim_timed("imvar", imvar(...))
#   ... after betas / inside_indices are computed:
#   .glim_log("grid", n_points_full = nrow(betas), n_inside = sum(inside_indices),
#             frac_inside = mean(inside_indices), imvar_sec = .glim_last("imvar"),
#             xi = paste(round(imvar_xi, 4), collapse = " "),
#             H = paste(signif(H, 4), collapse = " "))
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main function: wrap the three stages.
#
#   generated_grid <- .glim_timed("stage_grid_gen",  generate_grid(...))
#   possibils      <- .glim_timed("stage_grid_eval", glim_raw(...))
#   (extension loop)                                  ^ wrap the whole
#   `for (col in seq_len(ncol(X))) { ... }` in:
#        .glim_timed("stage_extend", { for (...) {...} })
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# extend_profile: per-step log. Wrap the three costly calls and log each step.
# In the growth loop, replace the corresponding lines with:
#
#   tang <- .glim_timed("ext_tangent", profile_tangent(family, X, prev_beta, col, dispersion))
#   ...
#   new_beta <- .glim_timed("ext_correct",
#       correct_profile_point(family, X, y, cand, col, dispersion, mle_coefs))
#   tag <- tag + 1L
#   pl <- .glim_timed("ext_eval", eval_plaus(new_beta, tag, m_ext))
#   .glim_log("extend", col = col, dir = direction, phase = "grow", k = k,
#             h = h, beta_col = new_beta[col], plaus = pl,
#             t_tangent = .glim_last("ext_tangent"),
#             t_correct = .glim_last("ext_correct"),
#             t_eval = .glim_last("ext_eval"))
#
# and in the bisection loop:
#
#   mid <- .glim_timed("ext_correct", correct_profile_point(...))
#   pl  <- .glim_timed("ext_eval", eval_plaus(mid, tag, m_ext))
#   .glim_log("extend", col = col, dir = direction, phase = "bisect", k = b,
#             h = NA_real_, beta_col = mid[col], plaus = pl,
#             t_tangent = NA_real_, t_correct = .glim_last("ext_correct"),
#             t_eval = .glim_last("ext_eval"))
#
# Finally log why it stopped (before the return):
#   .glim_log("extend_end", col = col, dir = direction, n_points = length(out_betas),
#             crossed_alpha = !is.null(bracket),
#             reason = if (is.null(stop_reason)) NA_character_ else stop_reason)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Report: summarise everything that was logged.
# ---------------------------------------------------------------------------
#' @export
glim_prof_report <- function(dir = .glim_prof$dir) {
  if (is.null(dir)) {
    stop("Run glim_prof_start() first.")
  }
  rd <- function(name) {
    f <- file.path(dir, name)
    if (file.exists(f) && file.size(f) > 0) utils::read.csv(f) else NULL
  }

  cat("\n== R-level timings (cumulative seconds) ==\n")
  if (length(.glim_prof$tot)) {
    tot <- unlist(.glim_prof$tot)
    cnt <- unlist(.glim_prof$cnt)
    print(data.frame(
      label = names(tot),
      seconds = round(tot, 3),
      calls = cnt[names(tot)],
      mean_s = round(tot / cnt[names(tot)], 4),
      row.names = NULL
    )[order(-tot), ])
  } else {
    cat("(none; add .glim_timed() wrappers)\n")
  }

  calls <- rd("calls.csv")
  if (!is.null(calls)) {
    calls$kind <- ifelse(calls$n_evals == 1, "single-point", "grid")
    cat("\n== C++ calls (fit_glm_omp_cpp) ==\n")
    agg <- do.call(
      rbind,
      lapply(split(calls, calls$kind), function(d) {
        data.frame(
          calls = nrow(d),
          total_wall_s = round(sum(d$wall_s), 2),
          total_fits = sum(d$n_evals * d$m),
          us_per_fit = round(1e6 * sum(d$wall_s) / sum(d$n_evals * d$m), 1),
          eff_cores = round(sum(d$cpu_s) / sum(d$wall_s), 2),
          threads = max(d$threads),
          outer_frac = round(mean(d$outer), 2),
          inner_frac = round(mean(d$inner), 2),
          imbalance = round(mean(d$imbalance), 2)
        )
      })
    )
    print(agg)

    cat(
      "\nPlausibility summary (all calls): ",
      sprintf(
        "min=%.4f max=%.4f; exact zeros in %.1f%% of evals; NaN in %d\n",
        min(calls$plaus_min, na.rm = TRUE),
        max(calls$plaus_max, na.rm = TRUE),
        100 * sum(calls$n_zero) / sum(calls$n_evals),
        sum(calls$n_nan)
      )
    )

    # Automatic hints
    sp <- calls[calls$kind == "single-point" & calls$threads > 1, ]
    if (nrow(sp) && sum(sp$cpu_s) / sum(sp$wall_s) < 1.5) {
      cat("HINT: single-point calls use ~1 core despite threads > 1: inner loop not parallel.\n")
    }
    gr <- calls[calls$kind == "grid" & calls$outer == 1, ]
    if (nrow(gr) && sum(gr$cpu_s) / sum(gr$wall_s) < 0.6 * max(gr$threads)) {
      cat("HINT: grid eval uses < 60% of threads: check BLAS oversubscription / imbalance.\n")
    }
    if (nrow(gr) && mean(gr$imbalance) > 1.3) {
      cat("HINT: load imbalance > 1.3: try schedule(dynamic) in the outer loop.\n")
    }
    if (
      any(calls$n_evals > 1) &&
        mean(calls$n_zero[calls$n_evals > 1] / calls$n_evals[calls$n_evals > 1]) > 0.3
    ) {
      cat("HINT: >30% of grid points have plausibility exactly 0: grid is too wide / untrimmed.\n")
    }
  }

  cat("\n== C++ phase totals (thread-seconds) ==\n")
  s <- prof_cpp_summary()
  print(unlist(lapply(s, signif, 4)))
  ph <- c(simulate = s$thread_sec_simulate, fit = s$thread_sec_fit, loglik = s$thread_sec_loglik)
  if (sum(ph) > 0) {
    cat(
      "\nShare of time: ",
      paste(names(ph), sprintf("%.0f%%", 100 * ph / sum(ph)), collapse = ", "),
      "\n"
    )
  }
  if (!is.na(s$frac_nonconverged) && s$frac_nonconverged > 0.01) {
    cat("HINT: >1% of GLM refits hit the iteration cap.\n")
  }

  iv <- rd("imvar.csv")
  if (!is.null(iv)) {
    cat("\n== imvar iterations ==\n")
    print(
      do.call(
        rbind,
        lapply(split(iv, iv$d), function(d) {
          data.frame(
            d = d$d[1],
            iterations = nrow(d),
            last_g = round(d$g_xi[nrow(d)], 4),
            sign_flips = sum(diff(sign(d$g_xi)) != 0),
            secs = round(sum(d$wall_s), 2)
          )
        })
      ),
      row.names = FALSE
    )
  }

  ex <- rd("extend.csv")
  if (!is.null(ex)) {
    cat("\n== extend_profile ==\n")
    print(
      do.call(
        rbind,
        lapply(split(ex, list(ex$col, ex$dir)), function(d) {
          data.frame(
            col = d$col[1],
            dir = d$dir[1],
            steps = nrow(d),
            last_plaus = round(d$plaus[nrow(d)], 4),
            t_tangent = round(sum(d$t_tangent, na.rm = TRUE), 2),
            t_correct = round(sum(d$t_correct, na.rm = TRUE), 2),
            t_eval = round(sum(d$t_eval, na.rm = TRUE), 2)
          )
        })
      ),
      row.names = FALSE
    )
  }
  ee <- rd("extend_end.csv")
  if (!is.null(ee)) {
    cat("\nExtension stop reasons:\n")
    print(ee, row.names = FALSE)
  }

  gd <- rd("grid.csv")
  if (!is.null(gd)) {
    cat("\n== grid ==\n")
    print(gd, row.names = FALSE)
  }
  invisible(NULL)
}
