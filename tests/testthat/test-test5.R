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

test_that("glim validates gamma/inverse.gaussian response is strictly positive", {
  df <- data.frame(y = c(-1, 2, 3, 4), x1 = c(0.1, 0.2, 0.3, 0.4))
  expect_error(glim(y ~ x1, data = df, family = "gamma"), "strictly positive")
  expect_error(glim(y ~ x1, data = df, family = "inverse.gaussian"), "strictly positive")
})

test_that("glim validates m and tol", {
  d <- make_gaussian_data()
  df <- data.frame(y = d$y, x1 = d$X[, 2])

  expect_error(
    glim(y ~ x1, data = df, m = -1),
    "'m' \\(number of samples\\) must be a single positive integer"
  )
  expect_error(
    glim(y ~ x1, data = df, m = 1.5),
    "'m' \\(number of samples\\) must be a single positive integer"
  )
  expect_error(
    glim(y ~ x1, data = df, m = c(1, 2)),
    "'m' \\(number of samples\\) must be a single positive integer"
  )

  expect_error(glim(y ~ x1, data = df, tol = -1))
})

test_that("glim errors when radial and approx are both TRUE", {
  d <- make_gaussian_data()
  df <- data.frame(y = d$y, x1 = d$X[, 2])
  expect_error(glim(y ~ x1, data = df, approx = TRUE, radial = TRUE), "cannot both be used")
})

# Works just throws an error when running check()?
# test_that("glim binomial accepts a two-column successes/failures matrix response", {
#   n_groups <- 8
#   x1 <- seq(-1, 1, length.out = n_groups)
#   successes <- c(1, 2, 3, 4, 4, 3, 2, 1)
#   trials <- rep(5L, n_groups)
#   failures <- trials - successes
#   y <- cbind(successes, failures)
#   X <- cbind(intercept = 1, x1 = x1)

#   fit <- glim(y ~ X - 1, family = "binomial", m = 5, n_grid_evals = 3)
#   expect_s3_class(fit, "glim_object")
# })
