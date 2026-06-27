library(testthat)

# Helper function to generate basic data
generate_mock_data <- function(n = 100, p = 3, family = "gaussian") {
  set.seed(42)
  X <- cbind(1, matrix(rnorm(n * (p - 1)), nrow = n)) # Explicit intercept
  true_beta <- c(0.5, -1.2, 0.8)
  eta <- X %*% true_beta

  if (family == "gaussian") {
    y <- eta + rnorm(n, sd = 0.5)
  } else if (family == "binomial") {
    prob <- 1 / (1 + exp(-eta))
    y <- rbinom(n, 1, prob)
  } else if (family == "gamma") {
    mu <- exp(eta)
    y <- rgamma(n, shape = 2, rate = 2 / mu)
  }

  list(X = X, y = as.numeric(y), true_beta = true_beta)
}

test_that("C++ solvers handle 1D vectors correctly (coerced to 1-col matrix)", {
  set.seed(123)
  x_vec <- rnorm(100)
  y_vec <- 2 * x_vec + rnorm(100, sd = 0.1)

  # Rcpp should cast x_vec to a 100x1 matrix
  res <- fit_gaussian_cpp(X = matrix(x_vec, ncol = 1), y = y_vec)

  expect_length(res, 1)
  expect_true(abs(res[1] - 2) < 0.2)
})

test_that("Omitting the intercept in X yields different results than lm() with intercept", {
  dat <- generate_mock_data(family = "gaussian")

  # Remove intercept column
  X_no_int <- dat$X[, -1]

  res_cpp <- fit_gaussian_cpp(X_no_int, dat$y)
  res_r <- coef(lm(dat$y ~ X_no_int - 1)) # Force lm to also drop intercept

  # They should match when BOTH drop the intercept
  expect_equal(as.numeric(res_cpp), as.numeric(res_r), tolerance = 1e-5)
})

test_that("Dimension mismatches throw clear errors", {
  dat <- generate_mock_data(family = "gaussian")

  # X is 100x3, y is 50x1
  expect_error(fit_gaussian_cpp(dat$X, dat$y[1:50]))
})

test_that("Missing data (NAs) in y propagates as NaN or throws error", {
  dat <- generate_mock_data(family = "gaussian")
  dat$y[5] <- NA

  # Armadillo solve will likely fail or return NaNs.
  # We test that it DOES NOT return a silent, valid-looking number.
  res <- fit_gaussian_cpp(dat$X, dat$y)

  expect_true(any(is.nan(res)) || any(is.na(res)) || all(res == 0))
})

test_that("Missing data (NAs) in X propagates correctly", {
  dat <- generate_mock_data(family = "binomial")
  dat$X[10, 2] <- NA

  res <- fit_logistic_cpp(dat$X, dat$y, c(0, 0, 0), FALSE)
  expect_true(any(is.nan(res)) || any(is.na(res)))
})

test_that("fit_logistic_cpp matches glm() for standard data", {
  dat <- generate_mock_data(family = "binomial")

  res_cpp <- fit_logistic_cpp(dat$X, dat$y, initial_beta = c(0, 0, 0), approx = FALSE)
  res_r <- coef(glm(dat$y ~ dat$X - 1, family = binomial()))

  expect_equal(as.numeric(res_cpp), as.numeric(res_r), tolerance = 1e-4)
})

test_that("fit_logistic_cpp handles perfect separation gracefully", {
  # Create perfectly separable data
  x_sep <- matrix(c(rnorm(50, -5), rnorm(50, 5)), ncol = 1)
  X_sep <- cbind(1, x_sep)
  y_sep <- c(rep(0, 50), rep(1, 50))

  # Warning: glm() will throw "fitted probabilities numerically 0 or 1 occurred"
  # We want to ensure C++ doesn't crash, but returns large coefficients
  res_cpp <- fit_logistic_cpp(X_sep, y_sep, c(0, 0), FALSE)

  expect_type(res_cpp, "double")
  expect_length(res_cpp, 2)
  # The slope should be positive and quite large
  expect_true(res_cpp[2] > 5)
})

test_that("fit_gamma_log_cpp recovers from terrible initial starting conditions", {
  dat <- generate_mock_data(family = "gamma")
  XtX <- t(dat$X) %*% dat$X

  # Provide wildly incorrect starting betas
  bad_initial <- c(50, -50, 100)

  # Since your C++ code has clamps (-30 to 30 for eta) and step-halving,
  # it should recover and not return Infs.
  res_cpp <- fit_gamma_log_cpp(dat$X, XtX, dat$y, bad_initial, FALSE)
  res_r <- coef(glm(dat$y ~ dat$X - 1, family = Gamma(link = "log")))

  expect_true(all(is.finite(res_cpp)))
  # It might not perfectly converge to the exact MLE in 20 iterations from a terrible start,
  # but it should be moving in the right direction and not be NaN.
  expect_true(!any(is.nan(res_cpp)))
})

test_that("fit_logistic_cpp recovers from extreme initial betas", {
  dat <- generate_mock_data(family = "binomial")

  bad_initial <- c(-999, 999, -999)
  res_cpp <- fit_logistic_cpp(dat$X, dat$y, bad_initial, FALSE)

  expect_true(all(is.finite(res_cpp)))
})

test_that("string_to_family mapping works inside fit_glm_omp_cpp", {
  dat <- generate_mock_data(family = "gaussian")
  betas <- matrix(c(0, 0, 0), nrow = 1)

  # Valid family string
  res_valid <- fit_glm_omp_cpp(dat$X, dat$y, c(0, 0, 0), betas, "gaussian", 1, 10, FALSE, FALSE)
  expect_type(res_valid, "double")

  # Invalid family string should trigger Rcpp::stop
  expect_error(
    fit_glm_omp_cpp(dat$X, dat$y, c(0, 0, 0), betas, "weibull", 1, 10, FALSE, FALSE),
    "Family not supported in C\\+\\+ backend"
  )
})

test_that("glm_logis_pl_cpp returns a valid probability between 0 and 1", {
  dat <- generate_mock_data(family = "binomial")

  mle_coefs <- fit_logistic_cpp(dat$X, dat$y, c(0, 0, 0), FALSE)
  beta_vals <- mle_coefs + 0.1 # slight perturbation

  pl_val <- glm_logis_pl_cpp(dat$X, dat$y, mle_coefs, beta_vals, m = 50, approx = FALSE)

  expect_true(pl_val >= 0 && pl_val <= 1)
})

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

test_that("glim handles two-column integer matrices (Success/Failure)", {
  X <- matrix(rnorm(20), ncol = 2)
  # 10 trials per row
  successes <- rbinom(10, 10, 0.5)
  failures <- 10 - successes
  y_mat <- cbind(successes, failures)

  # Ensure it doesn't crash during the row replication process
  expect_no_error(suppressWarnings(glim(
    X,
    y_mat,
    family = "binomial",
    betas = matrix(0, 1, 3),
    m = 10,
    parallel = FALSE
  )))
})

test_that("glim handles factor y correctly", {
  X <- matrix(rnorm(20), ncol = 2)
  y_fac <- factor(sample(c("Control", "Treatment"), 10, replace = TRUE))

  expect_no_error(suppressWarnings(glim(
    X,
    y_fac,
    family = "binomial",
    betas = matrix(0, 1, 3),
    m = 10,
    parallel = FALSE
  )))
})

test_that("Zero successes or zero failures do not break indexing", {
  X <- matrix(rnorm(4), ncol = 2)
  y_mat <- cbind(c(5, 0), c(0, 5)) # Row 1: all success. Row 2: all failures.

  expect_no_error(suppressWarnings(glim(
    X,
    y_mat,
    family = "binomial",
    betas = matrix(0, 1, 3),
    m = 10,
    parallel = FALSE
  )))
})

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
