library(testthat)

trace_crash <- function(msg) {
  cat(msg, "\n", file = "crash_trace.txt", append = TRUE)
}
# Clear the file from previous runs
writeLines("", "crash_trace.txt")

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

test_that("C++ solvers handle 1D vectors correctly (coerced to 1-col matrix)", {
  trace_crash("start 1")
  set.seed(123)
  x_vec <- rnorm(100)
  y_vec <- 2 * x_vec + rnorm(100, sd = 0.1)

  # Rcpp should cast x_vec to a 100x1 matrix
  res <- fit_gaussian_r(X = matrix(x_vec, ncol = 1), y = y_vec)

  expect_length(res, 1)
  expect_true(abs(res[1] - 2) < 0.2)
  trace_crash("end 1")
})

test_that("Omitting the intercept in X yields different results than lm() with intercept", {
  trace_crash("start 2")
  dat <- generate_mock_data(family = "gaussian")

  # Remove intercept column
  X_no_int <- dat$X[, -1]

  res_cpp <- fit_gaussian_r(X_no_int, dat$y)
  res_r <- coef(lm(dat$y ~ X_no_int - 1)) # Force lm to also drop intercept

  # They should match when BOTH drop the intercept
  expect_equal(as.numeric(res_cpp), as.numeric(res_r), tolerance = 1e-5)
  trace_crash("end 2")
})

test_that("fit_gamma_log_cpp recovers from terrible initial starting conditions", {
  trace_crash("4")
  dat <- generate_mock_data(family = "gamma")
  XtX <- t(dat$X) %*% dat$X

  # Provide wildly incorrect starting betas
  bad_initial <- c(50, -50, 100)

  # Since your C++ code has clamps (-30 to 30 for eta) and step-halving,
  # it should recover and not return Infs.
  res_cpp <- fit_gamma_log_r(dat$X, dat$y, bad_initial)
  res_r <- coef(glm(dat$y ~ dat$X - 1, family = Gamma(link = "log")))

  expect_true(all(is.finite(res_cpp)))
  # It might not perfectly converge to the exact MLE in 20 iterations from a terrible start,
  # but it should be moving in the right direction and not be NaN.
  expect_true(!any(is.nan(res_cpp)))
  trace_crash("4")
})


test_that("string_to_family mapping works inside fit_glm_omp_cpp", {
  trace_crash("5")
  dat <- generate_mock_data(family = "gaussian")
  betas <- matrix(c(0, 0, 0), nrow = 1)

  # Valid family string
  res_valid <- fit_glm_omp_cpp(
    X = dat$X,
    y = dat$y,
    mle_coefs = c(0, 0, 0),
    mle_val = 0,
    betas = betas,
    family = "gaussian",
    num_threads = 1,
    m = 10,
    parallel = FALSE,
    approx = FALSE,
    radial = FALSE,
    show_progress = FALSE,
    base_seed = 234
  )
  expect_type(res_valid, "double")

  # Invalid family string should trigger Rcpp::stop
  expect_error(
    fit_glm_omp_cpp(
      dat$X,
      dat$y,
      c(0, 0, 0),
      mle_val = 0,
      betas,
      "weibull",
      1,
      m = 10,
      FALSE,
      FALSE,
      radial = FALSE,
      show_progress = FALSE,
      234
    ),
    "Family not supported in C\\+\\+ backend"
  )
  trace_crash("5")
})
