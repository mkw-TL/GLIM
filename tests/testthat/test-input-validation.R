# glim() argument validation. All of these should fail before any expensive
# computation, so they are fast and do not need the C++ backend to do real work.

dat <- make_data("gaussian")$dat

test_that("bad family is rejected", {
  expect_error(glim(y ~ x, data = dat, family = "weibull"))
})

test_that("family supports partial matching", {
  d <- make_data("poisson")
  expect_error(glim(y ~ x, data = d$dat, family = "pois", m = 5, approx = TRUE, parallel = FALSE), NA)
})

test_that("m must be a positive integer", {
  expect_error(glim(y ~ x, data = dat, m = 0), "m must be")
  expect_error(glim(y ~ x, data = dat, m = -5), "m must be")
  expect_error(glim(y ~ x, data = dat, m = 2.5), "m must be")
  expect_error(glim(y ~ x, data = dat, m = c(5, 6)), "m must be")
})

test_that("tol must lie in (0, 1)", {
  expect_error(glim(y ~ x, data = dat, tol = 0), "tol")
  expect_error(glim(y ~ x, data = dat, tol = 1), "tol")
  expect_error(glim(y ~ x, data = dat, tol = -0.1), "tol")
})

test_that("approx / parallel must be single logicals", {
  expect_error(glim(y ~ x, data = dat, approx = NA), "approx")
  expect_error(glim(y ~ x, data = dat, approx = "yes"), "approx")
  expect_error(glim(y ~ x, data = dat, parallel = NA), "parallel")
  expect_error(glim(y ~ x, data = dat, parallel = c(TRUE, FALSE)), "parallel")
})

test_that("radial and approx cannot both be requested", {
  expect_error(glim(y ~ x, data = dat, radial = TRUE, approx = TRUE), "cannot both")
})

test_that("unknown extra arguments are rejected", {
  expect_error(glim(y ~ x, data = dat, bogus = 1), "Incorrect names")
})

test_that("a_val, b_val, max_it, n_grid_evals are validated", {
  expect_error(glim(y ~ x, data = dat, a_val = -1), "a_val")
  expect_error(glim(y ~ x, data = dat, a_val = "a"), "a_val")
  expect_error(glim(y ~ x, data = dat, b_val = 0), "b_val")
  expect_error(glim(y ~ x, data = dat, max_it = 0), "max_it")
  expect_error(glim(y ~ x, data = dat, max_it = 2.5), "max_it")
  expect_error(glim(y ~ x, data = dat, n_grid_evals = 1), "n_grid_evals")
  expect_error(glim(y ~ x, data = dat, n_grid_evals = 3.5), "n_grid_evals")
  expect_error(glim(y ~ x, data = dat, n_grid_evals = "a"), "n_grid_evals")
})

test_that("trim and directly_calc_marginals must be single logicals", {
  expect_error(glim(y ~ x, data = dat, trim = "a"), "trim")
  expect_error(glim(y ~ x, data = dat, directly_calc_marginals = 1), "directly_calc_marginals")
})

test_that("alpha outside (0,1) is rejected", {
  # alpha = 5 or -1 should be an error; currently only > 0 is checked, so
  # alpha = 5 slips through. This documents the intended behaviour.
  expect_error(glim(y ~ x, data = dat, alpha = -0.1), "alpha")
  expect_error(glim(y ~ x, data = dat, alpha = 5), "alpha")
})

test_that("response domain checks per family", {
  d <- data.frame(y = c(-1, 2, 3, 4, 5), x = 1:5)
  expect_error(glim(y ~ x, data = d, family = "poisson"), "non-negative integers")
  expect_error(glim(y ~ x, data = d, family = "gamma"), "strictly positive")
  expect_error(glim(y ~ x, data = d, family = "inverse.gaussian"), "strictly positive")
  d2 <- data.frame(y = c(0.5, 1, 2, 3, 4), x = 1:5)
  expect_error(glim(y ~ x, data = d2, family = "poisson"), "non-negative integers")
  d3 <- data.frame(y = c(0, 1, 2, 3, 4), x = 1:5)
  expect_error(glim(y ~ x, data = d3, family = "gamma"), "strictly positive")
})

test_that("NA values in data are caught (or removed by model.frame) without silent misalignment", {
  d <- data.frame(y = c(1, 2, NA, 4, 5, 6), x = c(1, 2, 3, NA, 5, 6))
  # model.frame drops incomplete rows by default, so this must either error or
  # run on the 4 complete rows; it must never error with an obscure message.
  res <- tryCatch(
    glim(y ~ x, data = d, m = 5, approx = TRUE, parallel = FALSE),
    error = function(e) e
  )
  if (inherits(res, "error")) {
    expect_match(conditionMessage(res), "NA")
  } else {
    expect_equal(nobs(res), 4)
  }
})

test_that("p > n triggers a warning", {
  d <- data.frame(y = c(1, 2), x1 = c(1, 2), x2 = c(3, 1), x3 = c(0, 5))
  expect_warning(
    try(glim(y ~ x1 + x2 + x3, data = d, m = 5, approx = TRUE, parallel = FALSE), silent = TRUE),
    "greater than the number of observations"
  )
})

test_that("rank-deficient design is reported clearly", {
  d <- data.frame(y = rnorm(20), x = rnorm(20))
  d$x2 <- 2 * d$x
  expect_error(
    glim(y ~ x + x2, data = d, m = 5, approx = TRUE, parallel = FALSE),
    "rank deficient|collinear"
  )
})

test_that("user-supplied betas must be a numeric matrix of the right width", {
  expect_error(glim(y ~ x, data = dat, betas = "a"), "Grid of betas")
  expect_error(glim(y ~ x, data = dat, betas = matrix(1:4, 2)), "Grid of betas") # integer matrix
})

test_that("a vector of betas is accepted as a single grid point (intended behaviour)", {
  # glim() converts a vector to a 1-row matrix but then validates args$betas
  # (the original vector) and stops; this test pins down the intended behaviour.
  fit <- glim(y ~ x, data = dat, betas = c(0.5, 0.3), m = 20, parallel = FALSE)
  expect_equal(nrow(fit$betas_evaluated), 1)
})

test_that("a data.frame of betas is accepted", {
  g <- expand.grid(b0 = c(0.4, 0.5), b1 = c(0.2, 0.3))
  fit <- glim(y ~ x, data = dat, betas = g, m = 20, parallel = FALSE)
  expect_equal(nrow(fit$betas_evaluated), 4)
})

test_that("binomial perfect separation is detected", {
  d <- data.frame(y = c(0, 0, 0, 1, 1, 1), x = 1:6)
  expect_error(glim(y ~ x, data = d, family = "binomial", m = 5), "separ")
})
