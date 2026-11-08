# glim() argument validation. All of these should fail before any expensive
# computation. Known defects in validation live in test-known-issues.R.
trace_crash <- function(msg) {
  cat(msg, "\n", file = "crash_trace.txt", append = TRUE)
}
# Clear the file from previous runs
writeLines("", "crash_trace.txt")


dat <- make_data("gaussian")$dat

test_that("bad family is rejected", {
  message("a")
  expect_error(glim(y ~ x, data = dat, family = "weibull"))
})

test_that("family supports partial matching", {
  message("b")
  d <- make_data("poisson")
  skip_on_cran()
  expect_error(
    glim(y ~ x, data = d$dat, family = "pois", m = 5, betas = matrix(d$beta, 1), parallel = FALSE),
    NA
  )
})

test_that("m must be a positive integer", {
  message("c")
  expect_error(glim(y ~ x, data = dat, m = 0), "m must be")
  expect_error(glim(y ~ x, data = dat, m = -5), "m must be")
  expect_error(glim(y ~ x, data = dat, m = 2.5), "m must be")
  expect_error(glim(y ~ x, data = dat, m = c(5, 6)), "m must be")
})

test_that("tol must lie in (0, 1)", {
  message("d")
  expect_error(glim(y ~ x, data = dat, tol = 0), "tol")
  expect_error(glim(y ~ x, data = dat, tol = 1), "tol")
  expect_error(glim(y ~ x, data = dat, tol = -0.1), "tol")
})

test_that("approx / parallel must be single logicals", {
  message("e")
  expect_error(glim(y ~ x, data = dat, approx = NA), "approx")
  expect_error(glim(y ~ x, data = dat, approx = "yes"), "approx")
  expect_error(glim(y ~ x, data = dat, parallel = NA), "parallel")
  expect_error(glim(y ~ x, data = dat, parallel = c(TRUE, FALSE)), "parallel")
})

test_that("radial and approx cannot both be requested", {
  message("f")
  expect_error(glim(y ~ x, data = dat, radial = TRUE, approx = TRUE), "cannot both")
})

test_that("unknown extra arguments are rejected", {
  message("g")
  expect_error(glim(y ~ x, data = dat, bogus = 1), "Incorrect names")
})

test_that("a_val, b_val, max_it, n_grid_evals are validated", {
  message("h")
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
  trace_crash("Starting test i")
  trace_crash("Running trim = 'a'")
  expect_error(glim(y ~ x, data = dat, trim = "a"), "trim")
  trace_crash("Running directly_calc_marginals = 1")
  expect_error(glim(y ~ x, data = dat, directly_calc_marginals = 1), "directly_calc_marginals")
  trace_crash("Finished test i")
})

test_that("alpha <= 0 is rejected", {
  trace_crash("Starting test j")
  expect_error(glim(y ~ x, data = dat, alpha = -0.1), "alpha")
  expect_error(glim(y ~ x, data = dat, alpha = 0), "alpha")
  trace_crash("Finished test j")
})

test_that("response domain checks per family", {
  trace_crash("Starting test k")
  d <- data.frame(y = c(-1, 2, 3, 4, 5), x = 1:5)
  expect_error(glim(y ~ x, data = d, family = "poisson"), "non-negative integers")
  expect_error(glim(y ~ x, data = d, family = "gamma"), "strictly positive")
  expect_error(glim(y ~ x, data = d, family = "inverse.gaussian"), "strictly positive")
  d2 <- data.frame(y = c(0.5, 1, 2, 3, 4), x = 1:5)
  expect_error(glim(y ~ x, data = d2, family = "poisson"), "non-negative integers")
  d3 <- data.frame(y = c(0, 1, 2, 3, 4), x = 1:5)
  expect_error(glim(y ~ x, data = d3, family = "gamma"), "strictly positive")
  trace_crash("End test k")
})

test_that("pass in betas", {
  trace_crash("Start test l")
  skip_on_cran()
  d <- data.frame(y = c(1, 2, 5, 6, 7, 8), x = c(1, 2, 5, 6, 7, 9))
  res <- tryCatch(
    glim(y ~ x, data = d, m = 5, betas = matrix(c(0, 1), 1), parallel = FALSE),
    error = function(e) e
  )
  if (inherits(res, "error")) {
    message("error")
    expect_match(conditionMessage(res), "NA")
  } else {
    message("equal")
    parts <- fit_parts(res)
    expect_equal(nrow(parts$X), 6)
    expect_equal(length(parts$y), 6)
  }
  trace_crash("End test l")
})

test_that("NA values in data are removed by model.frame without silent misalignment", {
  trace_crash("Start test l2")
  skip_on_cran()
  d <- data.frame(y = c(1, 2, NA, 4, 5, 6, 7, 8), x = c(1, 2, 3, NA, 5, 6, 7, 9))
  res <- tryCatch(
    glim(y ~ x, data = d, m = 5, betas = matrix(c(0, 1), 1), parallel = FALSE),
    error = function(e) e
  )
  if (inherits(res, "error")) {
    message("error")
    expect_match(conditionMessage(res), "NA")
  } else {
    message("equal")
    parts <- fit_parts(res)
    expect_equal(nrow(parts$X), 6)
    expect_equal(length(parts$y), 6)
  }
  trace_crash("End test l2")
})

test_that("p > n triggers a warning", {
  trace_crash("Start test m")
  d <- data.frame(y = c(1, 2), x1 = c(1, 2), x2 = c(3, 1), x3 = c(0, 5))
  expect_warning(
    try(glim(y ~ x1 + x2 + x3, data = d, m = 5, approx = TRUE, parallel = FALSE), silent = TRUE),
    "Input Warning: The number of predictors (p) is greater than the number of observations (n).",
    fixed = TRUE
  )
  trace_crash("end test m")
})

test_that("rank-deficient design is reported clearly", {
  trace_crash("start test n")
  d <- data.frame(y = rnorm(20), x = rnorm(20))
  d$x2 <- 2 * d$x
  expect_error(
    glim(y ~ x + x2, data = d, m = 5, approx = TRUE, parallel = FALSE),
    "rank deficient|collinear"
  )
  trace_crash("end test n")
})

test_that("user-supplied betas must be a numeric matrix", {
  trace_crash("start test o")
  expect_error(glim(y ~ x, data = dat, betas = "a"), "Grid of betas")
  expect_error(glim(y ~ x, data = dat, betas = matrix(1:4, 2)), "Grid of betas") # integer matrix
  trace_crash("end test o")
})

test_that("a data.frame of betas is accepted and stored in the flat grid object", {
  trace_crash("start test p")
  g <- expand.grid(b0 = c(0.4, 0.5), b1 = c(0.2, 0.3))
  fit <- glim(y ~ x, data = dat, betas = g, m = 20, parallel = FALSE)
  expect_equal(nrow(fit_parts(fit)$betas_evaluated), 4)
  expect_equal(length(fit_parts(fit)$possibilities), 4)
  trace_crash("end test p")
})

test_that("binomial perfect separation is detected", {
  trace_crash("start test q")
  d <- data.frame(y = c(0, 0, 0, 1, 1, 1), x = 1:6)
  expect_error(
    glim(y ~ x, data = d, family = "binomial", m = 5),
    "Data is completely separable. MLE does not exist"
  )
  trace_crash("end test q")
})
