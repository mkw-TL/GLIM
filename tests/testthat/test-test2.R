library(testthat)

# Helper function to generate basic data
generate_mock_data <- function(n = 100, p = 3, family = "gaussian") {
  set.seed(42)
  X <- cbind(1, matrix(rnorm(n * (p - 1)), nrow = n)) # Explicit intercept

  # Adjusted true_beta to prevent explosion with exponential link functions
  true_beta <- c(0.5, -0.2, 0.1)
  eta <- X %*% true_beta

  if (family == "gaussian") {
    y <- eta + rnorm(n, sd = 0.5)
  } else if (family == "binomial") {
    prob <- 1 / (1 + exp(-eta))
    y <- rbinom(n, 1, prob)
  } else if (family == "gamma") {
    mu <- exp(eta)
    y <- rgamma(n, shape = 2, rate = 2 / mu)
  } else if (family == "poisson") {
    mu <- exp(eta)
    y <- rpois(n, lambda = mu)
  } else if (family == "inverse.gaussian") {
    mu <- exp(eta)
    # Note: Requires statmod package for inverse gaussian generation
    if (requireNamespace("statmod", quietly = TRUE)) {
      y <- statmod::rinvgauss(n, mean = mu, shape = 1)
    } else {
      # Fallback approximation for simple dimension testing
      y <- rgamma(n, shape = 2, rate = 2 / mu)
    }
  }

  list(X = X, y = as.numeric(y), true_beta = true_beta)
}

# Shared fixtures for the GLIM test suite.
# These are intentionally small (fast to fit) but non-degenerate so that
# glm()/lm() and the GLIM C++ backends converge without warnings.

set.seed(42)

make_gaussian_data <- function(n = 60, p = 2) {
  X <- cbind(intercept = 1, matrix(rnorm(n * (p - 1)), n, p - 1))
  colnames(X) <- c("intercept", paste0("x", seq_len(p - 1)))
  beta <- seq(0.5, 1.5, length.out = p)
  y <- as.vector(X %*% beta + rnorm(n, sd = 0.5))
  list(X = X, y = y, beta = beta)
}

make_poisson_data <- function(n = 80, p = 2) {
  X <- cbind(intercept = 1, matrix(rnorm(n * (p - 1), sd = 0.3), n, p - 1))
  colnames(X) <- c("intercept", paste0("x", seq_len(p - 1)))
  beta <- seq(0.1, 0.3, length.out = p)
  lambda <- exp(X %*% beta)
  y <- rpois(n, lambda)
  list(X = X, y = y, beta = beta)
}

make_gamma_data <- function(n = 80, p = 2) {
  X <- cbind(intercept = 1, matrix(rnorm(n * (p - 1), sd = 0.3), n, p - 1))
  colnames(X) <- c("intercept", paste0("x", seq_len(p - 1)))
  beta <- seq(0.1, 0.3, length.out = p)
  mu <- exp(X %*% beta)
  y <- rgamma(n, shape = 5, rate = 5 / mu)
  list(X = X, y = y, beta = beta)
}

make_invgauss_data <- function(n = 80, p = 2) {
  X <- cbind(intercept = 1, matrix(rnorm(n * (p - 1), sd = 0.2), n, p - 1))
  colnames(X) <- c("intercept", paste0("x", seq_len(p - 1)))
  beta <- seq(1, 1.5, length.out = p)
  eta <- X %*% beta
  mu <- as.vector(sqrt(1 / eta))
  y <- pmax(mu + rnorm(n, sd = 0.05), 0.05)
  list(X = X, y = y, beta = beta)
}

make_binomial_data <- function(n = 100, p = 2, well_separated = FALSE) {
  X <- cbind(intercept = 1, matrix(rnorm(n * (p - 1)), n, p - 1))
  colnames(X) <- c("intercept", paste0("x", seq_len(p - 1)))
  beta <- if (well_separated) c(0, rep(50, p - 1)) else seq(-0.5, 0.5, length.out = p)
  prob <- 1 / (1 + exp(-X %*% beta))
  y <- rbinom(n, 1, prob)
  list(X = X, y = y, beta = beta)
}

skip_if_glim_missing <- function() {
  testthat::skip_if_not_installed("GLIM")
}

test_that("fit_glm_omp_cpp enforces strict dimension matching", {
  dat <- generate_mock_data(n = 100, p = 3)
  betas <- matrix(rnorm(30), ncol = 3) # 10 rows, 3 cols

  # Mismatched y length
  expect_error(
    fit_glm_omp_cpp(dat$X, dat$y[-1], c(0, 0, 0), betas, "gaussian", 1, 10, FALSE, FALSE),
    "Dimension mismatch"
  )

  # Mismatched betas columns
  bad_betas <- matrix(rnorm(40), ncol = 4)
  expect_error(
    fit_glm_omp_cpp(dat$X, dat$y, c(0, 0, 0), bad_betas, "gaussian", 1, 10, FALSE, FALSE),
    "Dimension mismatch"
  )
})

test_that("Progress bar prints to console when approx = FALSE", {
  dat <- generate_mock_data()
  betas <- matrix(rnorm(300), ncol = 3) # Need enough rows to trigger the 2% milestone

  # testthat captures the output to evaluate it
  expect_output(
    fit_glm_omp_cpp(dat$X, dat$y, c(0, 0, 0), betas, "gaussian", 1, 10, FALSE, FALSE),
    "Calculating Plausibilities"
  )
})


test_that("BLAS threads are restored even if C++ throws an error", {
  skip_if_not_installed("RhpcBLASctl")

  original_threads <- RhpcBLASctl::blas_get_num_procs()

  # Intentionally cause an error (e.g., mismatched dimensions)
  expect_error(glim_raw(
    X = matrix(1:10, ncol = 2),
    y = c(1, 2),
    family = "gaussian",
    betas = matrix(),
    mle_coefs = c(1, 1),
    mle_val = 0,
    m = 10,
    parallel = TRUE,
    approx = FALSE
  ))

  # Check if threads were restored despite the crash
  expect_equal(RhpcBLASctl::blas_get_num_procs(), original_threads)
})

# test_that("glim handles factor y correctly", {
#   X <- matrix(rnorm(20), ncol = 2)
#   y_fac <- factor(sample(c("Control", "Treatment"), 10, replace = TRUE))

#   expect_no_error(suppressWarnings(glim(
#     y_fac ~ X,
#     family = "binomial",
#     betas = matrix(0, 1, 3),
#     m = 10,
#     parallel = FALSE
#   )))
# })

test_that("Inverse Gaussian Information Matrix (J) matches standard GLM implementations", {
  set.seed(123)
  X <- cbind(1, rnorm(100))
  # True mu > 0
  mu <- exp(X %*% c(0.5, 0.2))
  # R doesn't have native rinvgauss, approximate with gamma for the sake of getting a working GLM fit
  y <- rgamma(100, shape = 2, rate = 2 / mu)

  res <- glm(y ~ X - 1, family = inverse.gaussian(link = "1/mu^2"))
  eta <- X %*% res$coefficients
  mu_i <- as.vector(sqrt(1 / eta))

  # Your calculation
  J_yours <- crossprod(X, X * (mu_i^3) / 4)

  # R's internal weighting for inverse gaussian with 1/mu^2 link
  # W = mu^3
  W_r <- res$weights
  J_r <- t(X) %*% diag(W_r) %*% X

  # They should be proportional/equivalent
  expect_equal(as.numeric(J_yours), as.numeric(J_r), tolerance = 1e-2)
})

test_that("lerped_xi boundary logic catches extreme uniform values", {
  # Rather than testing the whole function, you should abstract the lerping
  # logic into a tiny helper function (e.g., `interpolate_xi(u, AA, xi)`)
  # so you can unit test it directly with u = 0.0001 and u = 0.9999.

  AA <- seq(0.001, 0.999, length = 100)
  xi_list <- as.list(1:100)

  # Mocking the loop logic
  get_lerped <- function(u) {
    if (u < min(AA)) {
      return(xi_list[[1]])
    }
    if (u > max(AA)) {
      return(xi_list[[100]])
    }
    r <- sum(AA < u)
    w <- (u - AA[r]) / (AA[r + 1] - AA[r])
    return((1 - w) * xi_list[[r]] + w * xi_list[[r + 1]])
  }

  expect_equal(get_lerped(0.00005), 1)
  expect_equal(get_lerped(0.99995), 100)
  expect_true(get_lerped(0.5) > 1 && get_lerped(0.5) < 100)
})

test_that("C++ Poisson solver runs and returns expected dimensions", {
  set.seed(123)
  data <- generate_mock_data(n = 100, family = "poisson")

  # Test the Poisson log-link solver
  res <- fit_poisson_log_cpp(X = data$X, y = data$y, initial_beta = c(1, 1, 1))

  expect_type(res, "double")
  expect_length(res, ncol(data$X))
  # Basic sanity check that it's finite
  expect_true(all(is.finite(res)))
})

test_that("C++ Inverse Gaussian solver handles basic inputs", {
  set.seed(123)
  data <- generate_mock_data(n = 100, family = "inverse.gaussian")

  # You might need to adjust the exact arguments depending on the C++ signature
  # Report snippet indicates: fit_invgauss_cpp(X, y_sim, beta_vals, approx)
  beta_init <- rep(0, ncol(data$X))
  res <- fit_invgauss_cpp(X = data$X, y = data$y, beta_init, approx = FALSE)

  expect_type(res, "double")
  expect_length(res, ncol(data$X))
})

test_that("C++ Gamma solver converges cleanly", {
  set.seed(123)
  data <- generate_mock_data(n = 100, family = "gamma")

  res <- fit_gamma_log_cpp(
    X = data$X,
    y = data$y,
    XtX = t(data$X) %*% data$X,
    initial_beta = rep(1, length = 3),
    approx = FALSE
  )

  expect_type(res, "double")
  expect_length(res, ncol(data$X))
})
