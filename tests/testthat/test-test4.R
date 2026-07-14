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

test_that("compute_gamma_ll_r dispatches to vector and matrix C++ backends", {
  y <- c(1.2, 2.3, 0.8, 3.1)
  eta_vec <- c(1.1, 2.0, 0.9, 3.0)
  eta_mat <- cbind(eta_vec, eta_vec * 1.05)

  vec_res <- compute_gamma_ll_r(y, eta_vec, shape = 2)
  mat_res <- compute_gamma_ll_r(y, eta_mat, shape = 2)

  expect_true(is.numeric(vec_res))
  expect_length(vec_res, 1)
  expect_true(is.numeric(mat_res))
  expect_length(mat_res, ncol(eta_mat))
  expect_equal(mat_res[1], vec_res, tolerance = 1e-8)
})

test_that("compute_poisson_ll_r dispatches to vector and matrix C++ backends", {
  y <- c(1, 4, 0, 3)
  eta_vec <- c(0.1, 1.2, -0.5, 0.9)
  eta_mat <- cbind(eta_vec, eta_vec)

  vec_res <- compute_poisson_ll_r(y, eta_vec)
  mat_res <- compute_poisson_ll_r(y, eta_mat)

  expect_length(vec_res, 1)
  expect_length(mat_res, ncol(eta_mat))
  expect_equal(mat_res[1], vec_res, tolerance = 1e-8)
})

test_that("compute_gaussian_ll_r dispatches to vector and matrix C++ backends", {
  y <- c(1.0, 2.5, 3.1, 0.4)
  mu_vec <- c(1.1, 2.4, 3.0, 0.5)
  mu_mat <- cbind(mu_vec, mu_vec + 0.1)

  vec_res <- compute_gaussian_ll_r(y, mu_vec, sigma = 1)
  mat_res <- compute_gaussian_ll_r(y, mu_mat, sigma = 1)

  expect_length(vec_res, 1)
  expect_length(mat_res, ncol(mu_mat))
  expect_equal(mat_res[1], vec_res, tolerance = 1e-8)
})

test_that("compute_invgauss_ll_r dispatches to vector and matrix C++ backends", {
  y <- c(1.0, 1.5, 0.8, 2.0)
  mu_vec <- c(1.1, 1.4, 0.9, 1.9)
  mu_mat <- cbind(mu_vec, mu_vec * 0.95)

  vec_res <- compute_invgauss_ll_r(y, mu_vec, gamma_val = 1)
  mat_res <- compute_invgauss_ll_r(y, mu_mat, gamma_val = 1)

  expect_length(vec_res, 1)
  expect_length(mat_res, ncol(mu_mat))
  expect_equal(mat_res[1], vec_res, tolerance = 1e-8)
})

# Each prob2poss_* function maps a grid of candidate thetas to a possibility
# value in [0, 1] by comparing simulated-sample log-likelihoods against the
# log-likelihood at each candidate theta. `samples` mimics what glim()
# would generate internally: a (p x m) matrix of coefficient draws.

test_that("prob2poss_poisson returns possibilities in [0, 1] with expected shape", {
  d <- make_poisson_data()
  samples <- matrix(rnorm(ncol(d$X) * 200, mean = d$beta, sd = 0.05), nrow = ncol(d$X))
  theta_grid <- matrix(rnorm(ncol(d$X) * 5, mean = d$beta, sd = 0.05), nrow = ncol(d$X))

  poss <- prob2poss_poisson(d$X, d$y, samples, theta_grid)

  expect_length(poss, ncol(theta_grid))
  expect_true(all(poss >= 0 & poss <= 1))
})

test_that("prob2poss_poisson coerces a data.frame X to a matrix", {
  d <- make_poisson_data()
  samples <- matrix(rnorm(ncol(d$X) * 50, mean = d$beta, sd = 0.05), nrow = ncol(d$X))
  theta_grid <- matrix(d$beta, ncol = 1)

  poss_matrix <- prob2poss_poisson(d$X, d$y, samples, theta_grid)
  poss_df <- prob2poss_poisson(as.data.frame(d$X), d$y, samples, theta_grid)

  expect_equal(poss_matrix, poss_df, tolerance = 1e-8)
})

test_that("prob2poss_gamma returns possibilities in [0, 1]", {
  d <- make_gamma_data()
  samples <- matrix(rnorm(ncol(d$X) * 200, mean = d$beta, sd = 0.02), nrow = ncol(d$X))
  theta_grid <- matrix(rnorm(ncol(d$X) * 5, mean = d$beta, sd = 0.02), nrow = ncol(d$X))

  poss <- prob2poss_gamma(d$X, d$y, samples, theta_grid)

  expect_length(poss, ncol(theta_grid))
  expect_true(all(poss >= 0 & poss <= 1))
})

test_that("prob2poss_gaussian returns possibilities in [0, 1]", {
  d <- make_gaussian_data()
  samples <- matrix(rnorm(ncol(d$X) * 200, mean = d$beta, sd = 0.1), nrow = ncol(d$X))
  theta_grid <- matrix(rnorm(ncol(d$X) * 5, mean = d$beta, sd = 0.1), nrow = ncol(d$X))

  poss <- prob2poss_gaussian(d$X, d$y, samples, theta_grid)

  expect_length(poss, ncol(theta_grid))
  expect_true(all(poss >= 0 & poss <= 1))
})

test_that("prob2poss_invgauss returns possibilities in [0, 1]", {
  d <- make_invgauss_data()
  samples <- matrix(rnorm(ncol(d$X) * 200, mean = d$beta, sd = 0.05), nrow = ncol(d$X))
  theta_grid <- matrix(rnorm(ncol(d$X) * 5, mean = d$beta, sd = 0.05), nrow = ncol(d$X))

  poss <- prob2poss_invgauss(d$X, d$y, samples, theta_grid)

  expect_length(poss, ncol(theta_grid))
  expect_true(all(poss >= 0 & poss <= 1))
})

test_that("prob2poss_logis handles 0/1 response, success/failure matrix, and factor response", {
  d <- make_binomial_data()
  samples <- matrix(rnorm(ncol(d$X) * 200, mean = d$beta, sd = 0.05), nrow = ncol(d$X))
  theta_grid <- matrix(d$beta, ncol = 1)

  poss_binary <- prob2poss_logis(d$X, d$y, samples, theta_grid)
  expect_length(poss_binary, 1)
  expect_true(poss_binary >= 0 && poss_binary <= 1)

  # y and X length mismatch should error
  expect_error(prob2poss_logis(d$X, d$y[-1], samples, theta_grid), "differ")

  # Two-column successes/failures matrix
  n_groups <- 10
  Xg <- cbind(intercept = 1, x1 = seq(-1, 1, length.out = n_groups))
  successes <- rbinom(n_groups, 5, 0.5)
  failures <- 5L - successes
  y_mat <- cbind(successes, failures)
  samples_g <- matrix(rnorm(ncol(Xg) * 100), nrow = ncol(Xg))
  theta_g <- matrix(c(0, 0.1), ncol = 1)
  poss_grouped <- prob2poss_logis(Xg, y_mat, samples_g, theta_g)
  expect_length(poss_grouped, 1)

  # Factor response
  y_factor <- factor(ifelse(d$y == 1, "yes", "no"), levels = c("no", "yes"))
  poss_factor <- prob2poss_logis(d$X, y_factor, samples, theta_grid)
  expect_equal(poss_factor, poss_binary, tolerance = 1e-8)
})
