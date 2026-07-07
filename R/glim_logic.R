#' @useDynLib GLIM, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom progress progress_bar
#' @importFrom parallel detectCores
#' @importFrom stats logLik glm Gamma poisson inverse.gaussian lm coef runif qchisq model.matrix rnorm
#' @importFrom utils head
NULL


# Original code written by Joe Harrison (jrharr25@ncsu.edu), translated to cpp by Gemini
# pkgbuild::compile_dll() validates the package directory differently. Rcpp might implement it's directory check differently
# After making changes, restart R, get in the package directory
# devtools::document()
# devtools::install() or devtools::load_all()

#' Scale the design matrix
#'
#' Scales the design matrix for numerical stability when fitting
#'
#' @param X design matrix
#' @return Scaled design matrix (excludes intercepts and dummy variables)
#' @export
scale_design_matrix <- function(X) {
  # Find columns with more than 2 unique values
  continuous_cols <- apply(X, 2, function(col) length(unique(col)) > 2)

  # Scale only those columns (center and scale)
  X[, continuous_cols] <- scale(X[, continuous_cols], center = FALSE)

  return(X)
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
#' @param mle_val The log-likelihood value at the maximum likelihood estimates.
#' @param m Number of evaluations per beta (number of samples).
#' @param parallel Logical indicating whether to run in parallel (primarily for debugging).
#' @param approx Logical indicating whether to use the elliptical approximation.
#' @return A matrix of evaluated possibility outputs.
#' @noRd
glim_raw <- function(X, y, family = "gaussian", betas, mle_coefs, mle_val, m, parallel, approx) {
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
  column_names,
  n_grid_evals = 25,
  max_sd = 1,
  m = 500
) {
  initial_xi <- rep(1, length(mle_coefs))

  alpha_target <- 0.01

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
    .02,
    2,
    .65,
    FALSE,
    m # This is our m value
  )
  # sqrt(dispersion * xi / eigen_val)
  q_val <- qchisq(1 - alpha_target, df = length(mle_coefs) - 1) # This is the first guess at our quantile distance, which we need to modify slightly by xi.
  base_scale <- sqrt(dispersion * q_val * abs(1 / eigen_vals))
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
  colnames(betas) <- column_names
  # Written by Gemini
  shifted_points <- sweep(betas, 2, mle_coefs, FUN = "-") # 2 means that it applies along the columns. Betas - mle_coefs for each column
  rotated_points <- shifted_points %*% eigen_vecs
  scaled_sq_points <- sweep(rotated_points^2, 2, semi_axes^2, FUN = "/")
  inside_indices <- rowSums(scaled_sq_points) <= 1
  # Check if they are inside the unit sphere (i.e. inside the ellipse)
  # From betas, select all the columns (proposed betas) that are inside
  points_inside <- betas[inside_indices, ]
  return(points_inside)
}

#' Generates a grid of parameter values (aligned in eigen-vector space)
#'
#' Each direction has a default of 20 steps
#'
#' @param mle_coefs The mle coef
#' @noRd
generate_eigen_grid <- function(
  X,
  y,
  family,
  mle_coefs,
  eigen_vecs,
  eigen_vals,
  dispersion,
  ll_mle_original_data,
  column_names,
  n_grid_evals = n_grid_evals,
  max_sd = 2.5
) {
  print("Generating a grid of beta values")
  initial_xi <- rep(1, length(mle_coefs))
  imvar_xi <- imvar(
    X,
    y,
    initial_xi,
    family,
    .1,
    mle_coefs,
    ll_mle_original_data,
    eigen_vecs,
    eigen_vals,
    dispersion,
    .05, # this is the alpha level to which imvar is attempting to scale to
    2, # alpha_val
    .65, # beta_val
    30, # max_it
    FALSE
  )
  scaling <- sqrt(eigen_vals) * imvar_xi
  print(scaling)
  slices <- lapply(1:length(mle_coefs), function(i) {
    seq(-1, 1, length.out = n_grid_evals)
  })
  eigenspace_grid <- as.matrix(expand.grid(slices))
  print(head(eigenspace_grid))
  scaled_eigenspace <- t(t(eigenspace_grid) * as.vector(scaling))
  print(head(scaled_eigenspace))
  param_deltas <- scaled_eigenspace %*% t(eigen_vecs)
  print(head(param_deltas))
  betas <- t(t(param_deltas) + as.vector(mle_coefs))
  print(head(betas))
  colnames(betas) <- column_names

  return(betas)
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
  B <- 100
  AA <- seq(0.001, 0.999, length.out = B)

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
#' Main wrapper function to fit a GLIM model. Users can pass in their own grid of beta values to evaluate by setting betas = MATRIX.
#' A typical use would be MATRIX = as.matrix(expand.grid(beta_0_seq, beta_1_seq))
#'
#' If using the elliptical approximation, can tweak the Robbins-Monroe algorithm by changing a_val, b_val, and max_it.
#'
#' @param X Matrix of predictors. Note that any dataframe should be first run through model.matrix()
#' @param y Vector of response variables. (Can be a nx2 matrix of successes and failures for binomial data)
#' @param family String denoting the exponential family. Choices are `"gaussian"`, `"binomial"`, `"gamma"`, `"poisson"`, `"inverse.gaussian"`.
#' @param betas A matrix (or column vector) of different beta values to evaluate the possibility over.
#' @param m Number of samples/evaluations to perform (default `1000`).
#' @param approx Logical indicating whether to use the elliptical approximation (default `FALSE`).
#' @param parallel Logical indicating whether to process in parallel.
#' @return If using an elliptical approximation, returns samples of parameter values. Else, returns a list containing the matrix of betas upon which it was evaluated, and a vector of the corresponding possibilities.
#' @export
glim <- function(
  formula,
  data,
  family = "gaussian",
  m = 1000,
  approx = FALSE,
  appendix = FALSE,
  parallel = TRUE,
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
  mf <- eval(mf, parent.frame())
  mt <- attr(mf, "terms") # figure out the variables to expand out in the model.matrix. Contains further attributes of factors, etc
  offset <- model.offset(mf) # would need to put into glim()
  X <- model.matrix(mt, mf, NULL) # model.matrix() creates the dummy columns. NULL is the contrasts (which could be set as default setting within a useR's R session, but I don't want to incorporate this)
  y <- model.response(mf, "numeric")
  args <- list(...)
  print("here")
  if (approx == TRUE) {
    if ("a_val" %in% names(args)) {
      if (is.numeric(args$a_val) & length(args$a_val) == 1 & args$a_val > 0) {
        a_val <- args$a_val
      } else {
        stop("Input Error: a_val must be a positive number")
      }
    } else {
      a_val <- 2
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
    if ("max_it" %in% names(args)) {
      if (is.integer(args$max_it) & length(args$max_it) == 1 & args$max_it > 0) {
        max_it <- args$max_it
      } else {
        stop("Input Error: max_it must be a positive integer")
      }
      max_it <- args$max_it
    } else {
      max_it <- 25
    }
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
      is.integer(as.integer(args$n_grid_evals)) &
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
  if (!is.null(names(args))) {
    if (!(all(names(args) %in% c("a_val", "b_val", "max_it", "betas", "n_grid_evals")))) {
      stop("Incorrect names of additional arguments passed")
    }
  }

  if (!is.matrix(X)) {
    stop(
      "X must be a matrix. To convert a dataframe into a matrix, either run matrix(), or model.matrix() depending on whether you have categories"
    )
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

  if (any(is.na(X)) || any(is.na(y))) {
    warning(
      "Input Warning: Missing values (NA) detected in X or y. This will likely cause crashes."
    )
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
  column_names <- c()
  if (is.null(colnames(X))) {
    if (ncol(X) > 1) {
      for (col in 1:ncol(X)) {
        column_names[col] <- paste0("b", col)
      }
    } else {
      column_names <- "b1"
    }
  } else {
    column_names <- colnames(X)
  }
  # Got rid of intercept logic, since covered.

  ## Binomial setup
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

      if (ncol(X) == 1) {
        idx_success <- rep(1:length(X), times = successes)
        idx_fail <- rep(1:length(X), times = failures)
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
    ll_mle_original_data <- as.numeric(logLik(glm(y ~ X - 1, family = "binomial")))
    res <- glm(y ~ X - 1, family = "binomial")
    mle_coefs <- res$coefficients
    p_i <- res$fitted.values
    dispersion <- 1
    if (is.null(betas)) {
      J <- crossprod(X, X * as.vector((p_i * (1 - p_i))))
    }
    p2p_function <- prob2poss_logis

    ## Gamma setup
  } else if (family == "gamma") {
    mle_coefs <- glm(y ~ X - 1, family = Gamma(link = "log"))$coefficients
    eta <- X %*% mle_coefs
    dispersion <- mle_estimate_dispersion_gamma(y, exp(X %*% mle_coefs), length(mle_coefs))
    ll_mle_original_data <- compute_gamma_ll_r(y, eta, 1 / dispersion)
    if (is.null(betas)) {
      J <- crossprod(X, X)
    }
    p2p_function <- prob2poss_gamma

    ## Poisson setup
  } else if (family == "poisson") {
    ll_mle_original_data <- as.numeric(logLik(glm(y ~ X - 1, family = poisson(link = "log"))))
    mle_coefs <- glm(y ~ X - 1, family = poisson(link = "log"))$coefficients
    eta <- X %*% mle_coefs
    lambda_i <- exp(eta)
    dispersion <- 1
    if (is.null(betas)) {
      J <- crossprod(X, X * as.vector(lambda_i))
    }
    p2p_function <- prob2poss_poisson

    ## Inverse Gaussian setup
  } else if (family == "inverse.gaussian") {
    ll_mle_original_data <- as.numeric(logLik(glm(
      y ~ X - 1,
      family = inverse.gaussian(link = "1/mu^2")
    )))
    mle_coefs <- glm(y ~ X - 1, family = inverse.gaussian(link = "1/mu^2"))$coefficients
    eta <- X %*% mle_coefs
    mu_i <- as.vector(sqrt(1 / eta))
    dispersion <- mle_estimate_dispersion_inv_gauss(y, mean(y))
    if (is.null(betas)) {
      J <- crossprod(X, X * (mu_i^3) / 4)
    }
    p2p_function <- prob2poss_invgauss

    ## Gaussian Setup
  } else if (family == "normal" || family == "gaussian") {
    ll_mle_original_data <- as.numeric(logLik(lm(y ~ X - 1)))
    mle_coefs <- fit_gaussian_cpp(X, y)
    dispersion <- 1
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
    stop("Input Error: 'dispersion' must be a strictly positive numeric value.")
  }

  if (approx == FALSE & appendix == FALSE) {
    if (is.null(betas)) {
      print("generating_grid_of_betas")
      betas <- generate_grid(
        X,
        y,
        family,
        mle_coefs,
        eigen_vecs = eJ$vectors,
        eigen_vals = eJ$values,
        dispersion,
        ll_mle_original_data,
        column_names,
        n_grid_evals = n_grid_evals,
        max_sd = 3,
        m = m
      )
      colnames(betas) <- column_names
    }
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
      betas = betas,
      family = family
    )
    class(obj_to_return) <- "glim_object"
    return(obj_to_return)
  }

  if (approx == TRUE & appendix == FALSE) {
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
    beta_seq_list <- list()
    for (i in 1:nrow(samples)) {
      beta_seq_list[[i]] <- seq(min(samples[i, ]), max(samples[i, ]), length.out = 51)
    }
    beta_p2p_grid <- as.matrix(expand.grid(beta_seq_list))
    poss <- p2p_function(X, y, samples, t(beta_p2p_grid))
    colnames(beta_p2p_grid) <- column_names
    obj_to_return <- list(possibilities = poss, betas = beta_p2p_grid, family = family)
    class(obj_to_return) <- "glim_object"
    return(obj_to_return)
  }

  if (appendix == TRUE & approx == TRUE) {
    stop("Appendix and approximation methods cannot both be used")
  }
  if (appendix == TRUE & approx == FALSE) {
    # TODO implement
    return(appendix(eJ, num_samps, X, y, mle_coefs, family, dispersion, m, tol, max_it, a, b))
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
  if (is.data.frame(X)) {
    X <- as.matrix(X)
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
#' @return A vector of mapped possibility values.
#' @export
prob2poss_gaussian <- function(X, y, samples, the_compared_theta) {
  if (is.data.frame(X)) {
    X <- as.matrix(X)
  }

  eta <- X %*% samples

  # Assuming identity link for Gaussian by default
  mle_coefs <- fit_gaussian_cpp(X, y)
  est_sd <- sqrt(est_dispersion(y, X %*% mle_coefs, length(mle_coefs)))

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

#' Appendix C++ Bridge Function
#'
#' Helper function acting as a bridge to underlying C++ routines for sample generation.
#'
#' @param eJ Eigen decomposition object.
#' @param num_samps Number of samples to generate.
#' @param X Predictor matrix.
#' @param y Response vector.
#' @param mle_coefs Maximum likelihood estimates for coefficients.
#' @param family String denoting the exponential family.
#' @param dispersion The dispersion parameter.
#' @param m Parameter `m` defining scaling or sampling limits.
#' @param tol Tolerance level for convergence criteria.
#' @param max_it Maximum number of iterations the stochastic algorithm runs for for each grid value.
#' @param a Hyperparameter `a` (should be between X and Y TODO)
#' @param b Hyperparameter `b` (should be between X and Y TODO)
#' @return A matrix of output samples evaluated by the C++ backend.
#' @noRd
appendix <- function(eJ, num_samps, X, y, mle_coefs, family, dispersion, m, tol, max_it, a, b) {
  eig_vecs <- eJ$vectors
  eig_vals <- eJ$values
  output_samples <- appendix_code(
    eig_vecs,
    eig_vals,
    num_samps,
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
#' Note that a 95% confidence interval corresponds to alpha = .05
#'
#' @param alpha Note that a 95% confidence interval corresponds to alpha = .05
#' @param betas The grid of beta values which we are evaluating on
#' @param possibilities The vector of possibilities
#' @return A matrix of compatable beta values from the grid.
#' @export
get_CI_manual <- function(alpha, betas, possibilities) {
  return(betas[possibilities > alpha, ])
}

#' Returns a confidence interval
#'
#' Note that a 95% confidence interval corresponds to alpha = .05
#'
#' @param glim_object Note that a 95% confidence interval corresponds to alpha = .05
#' @return A matrix of compatable beta values from the grid.
#' @export
get_CI <- function(glim_object, alpha = .05) {
  if (class(glim_object) != "glim_object") {
    stop("Object which was passed is not a direct result from glim()")
  }
  return(glim_object$betas[glim_object$possibilities > alpha, ])
}


comput_gauss <- function(y, mu, sigma) {
  return(compute_gaussian_ll(y, mu, sigma))
}

est_sig_sq <- function(y, mu, p) {
  return(est_dispersion(y, mu, p))
}

#' Plot
#'
#' Plotting for glim objects
#'
#' @param object from 'glim()'
#' @return Marginal plots for betas
#' @export
plot.glim_object <- function(output) {
  betas <- output$betas
  poss <- output$possibilities
  family <- output$family

  num_predictors <- ncol(betas)
  grid_cols <- ceiling(sqrt(num_predictors))
  grid_rows <- ceiling(num_predictors / grid_cols)

  old_par <- par(no.readonly = TRUE)
  # Add 'oma' (outer margin area) to par(). c(bottom, left, top, right)
  # We add 3 lines of space to the top
  par(mfrow = c(grid_rows, grid_cols), mar = c(4, 4, 3, 1), oma = c(0, 0, 3, 0))

  on.exit(par(old_par)) # if any crashes, don't have the user's state altered
  #where the marginalization happens
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
      axis(1, tck = 1, lty = 2, col = "grey")
      axis(2, tck = 1, lty = 2, col = "grey")
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
  # Reset back to default
  par(mfrow = c(1, 1), oma = c(0, 0, 0, 0))
}
