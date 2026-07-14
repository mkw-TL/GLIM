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


test_that("glim_raw works with the gaussian family", {
  # Generate dummy data for a Gaussian distribution
  set.seed(123)
  X <- matrix(rnorm(20), ncol = 2)
  y <- X %*% c(1.5, -2) + rnorm(10)

  # Assuming betas, mle_coefs, etc. are calculated or mocked
  # You need to fill in the valid mock arguments based on your package's workflow
  mle_fit <- lm(y ~ X - 1)
  mle_coefs <- coef(mle_fit)

  # Call glim_raw with family = "gaussian"
  # Note: Add remaining required arguments like 'betas', 'm', 'parallel', 'approx'
  result <- glim_raw(
    X = X,
    y = y,
    family = "gaussian",
    betas = matrix(c(.3, .5), nrow = 1),
    mle_coefs = mle_coefs,
    mle_val = logLik(mle_fit),
    m = 100,
    parallel = FALSE,
    approx = TRUE
  )

  # Assert expected output type or structure
  expect_true(is.matrix(result))
})

test_that("glim_raw executes successfully for gaussian and inverse.gaussian families", {
  set.seed(42)
  # Dummy design matrix (10 rows, 2 predictors)
  X <- matrix(rnorm(20), ncol = 2)

  # Dummy beta matrix for the grid (2 rows for predictors, 1 column for 1 evaluation)
  betas <- matrix(c(0.5, -0.5), nrow = 1)
  mle_coefs <- c(0.5, -0.5)
  mle_val <- -15.5

  # --- Test 1: Gaussian ---
  # Response can be any real number
  y_gauss <- rnorm(10)

  res_gauss <- glim_raw(
    X = X,
    y = y_gauss,
    family = "gaussian",
    betas = betas,
    mle_coefs = mle_coefs,
    mle_val = mle_val,
    m = 10,
    parallel = FALSE,
    approx = TRUE
  )
  # Just expecting it to return a matrix (the possibility outputs) without error
  expect_true(is.matrix(res_gauss))

  # --- Test 2: Inverse Gaussian ---
  # Response must be strictly positive
  y_inv_gauss <- runif(10, min = 0.5, max = 5.0)

  res_inv_gauss <- glim_raw(
    X = X,
    y = y_inv_gauss,
    family = "inverse.gaussian",
    betas = betas,
    mle_coefs = mle_coefs,
    mle_val = mle_val,
    m = 10,
    parallel = FALSE,
    approx = TRUE
  )
  expect_true(is.matrix(res_inv_gauss))
})

test_that("C++ backend robustly handles collinear design matrices via fallback solver", {
  set.seed(123)

  # Create a linearly dependent design matrix
  # Col 3 is exactly Col 1 multiplied by 2
  X_base <- matrix(rnorm(20), ncol = 2)
  X_collinear <- cbind(X_base, X_base[, 1] * 2)

  y_pois <- rpois(10, lambda = 3)

  # 3 predictors means we need 3 coefficients
  betas <- matrix(rep(0.1, 3), nrow = 1)
  mle_coefs <- rep(0.1, 3)

  # If the C++ code doesn't have the fallback, this would likely throw an error
  # or crash due to the matrix inversion failing. With the fallback, it should succeed.
  expect_no_error({
    res <- glim_raw(
      X = X_collinear,
      y = y_pois,
      family = "poisson",
      betas = betas,
      mle_coefs = mle_coefs,
      mle_val = -10,
      m = 10,
      parallel = FALSE,
      approx = TRUE
    )
  })
})

test_that("glim_raw works with gamma and poisson families", {
  set.seed(123)
  X <- matrix(rnorm(20), ncol = 2)
  betas <- matrix(c(0.1, 0.2), nrow = 1)
  mle_coefs <- c(0.1, 0.2)
  mle_val <- -10.0

  # --- Gamma ---
  # Gamma requires strictly positive response values
  y_gamma <- runif(10, min = 1, max = 10)

  print(dim(betas))

  res_gamma <- glim_raw(
    X = X,
    y = y_gamma,
    family = "gamma",
    betas = betas,
    mle_coefs = mle_coefs,
    mle_val = mle_val,
    m = 10,
    parallel = FALSE,
    approx = TRUE
  )
  expect_true(is.matrix(res_gamma))

  # --- Poisson ---
  # Poisson requires non-negative integer response values
  y_pois <- rpois(10, lambda = 3)

  res_pois <- glim_raw(
    X = X,
    y = y_pois,
    family = "poisson",
    betas = betas,
    mle_coefs = mle_coefs,
    mle_val = mle_val,
    m = 10,
    parallel = FALSE,
    approx = TRUE
  )
  expect_true(is.matrix(res_pois))
})


test_that("glim_raw triggers parallel processing logic safely", {
  data <- generate_mock_data(family = "poisson")
  mle_fit <- glm(data$y ~ data$X - 1, family = poisson)

  # Trigger parallel = TRUE block (lines 56-66)
  result <- glim_raw(
    X = data$X,
    y = data$y,
    family = "poisson",
    betas = matrix(c(2, 3, 1), nrow = 1),
    mle_coefs = coef(mle_fit),
    mle_val = as.numeric(logLik(mle_fit)),
    m = 50,
    parallel = TRUE, # This will test the parallel core setup and RhpcBLASctl check
    approx = TRUE
  )

  expect_type(result, "double")
})

test_that("scale_design_matrix scales continuous columns and leaves dummies alone", {
  X <- cbind(intercept = 1, age = c(20, 45, 70, 33), male = c(1, 0, 1, 0))
  out <- scale_design_matrix(X)

  expect_true(is.matrix(out))
  # intercept (1 unique value) and male (2 unique values) untouched
  expect_equal(out[, "intercept"], X[, "intercept"])
  expect_equal(out[, "male"], X[, "male"])
  # age (continuous) has been rescaled by its RMS
  expect_false(isTRUE(all.equal(out[, "age"], X[, "age"])))
  expect_equal(out[, "age"], as.vector(scale(X[, "age", drop = FALSE], center = FALSE)))
})

test_that("scale_design_matrix returns matrix unchanged when there are no continuous columns", {
  X <- cbind(intercept = 1, male = c(1, 0, 1, 0))
  out <- scale_design_matrix(X)
  expect_equal(out, X)
})

test_that("scale_design_matrix coerces a data.frame input to a matrix", {
  df <- data.frame(intercept = 1, age = c(20, 45, 70, 33))
  out <- scale_design_matrix(df)
  expect_true(is.matrix(out))
  expect_equal(out[, "age"], as.vector(scale(df$age, center = FALSE)))
})

test_that("scale_design_matrix handles a single continuous column without collapsing to a vector", {
  X <- matrix(c(1, 2, 3, 4, 5), ncol = 1)
  out <- scale_design_matrix(X)
  expect_true(is.matrix(out))
  expect_equal(dim(out), dim(X))
})

test_that("scale_design_matrix rejects non-numeric / non-matrix input", {
  expect_error(scale_design_matrix("not a matrix"), "must be a numeric matrix")
  expect_error(scale_design_matrix(matrix(letters[1:4], 2, 2)), "must be a numeric matrix")
})

test_that("scale_design_matrix rejects empty matrices", {
  expect_error(scale_design_matrix(matrix(numeric(0), nrow = 0, ncol = 2)), "at least one row")
  expect_error(scale_design_matrix(matrix(numeric(0), nrow = 2, ncol = 0)), "at least one row")
})

test_that("scale_design_matrix rejects non-finite values", {
  X <- cbind(intercept = 1, age = c(20, NA, 70, 33))
  expect_error(scale_design_matrix(X), "NA/NaN/Inf")

  X_inf <- cbind(intercept = 1, age = c(20, Inf, 70, 33))
  expect_error(scale_design_matrix(X_inf), "NA/NaN/Inf")
})

# These wrappers each do: if(is.vector(x)) call the vector C++ routine,
# else call the *_mat C++ routine. Both branches are exercised below for
# every family.
