# Tests that pin down suspected bugs found by reading the code. Each states the
# *intended* behaviour, so a failure here points at a real defect.

test_that("prob2poss_poisson accepts whole-number doubles (e.g. from a data.frame)", {
  # Code checks is.integer(y), which is FALSE for c(1, 2, 3) stored as double,
  # even though glim() itself accepts such y.
  set.seed(1)
  x <- rnorm(30)
  y <- as.numeric(rpois(30, exp(0.3 * x)))
  X <- cbind(1, x)
  mle <- coef(glm(y ~ X))
  S <- matrix(c(0, 0.3), 2, 50) + matrix(rnorm(100, sd = 0.05), 2)
  expect_error(prob2poss("poisson", X, y, mle, S, c(0, 0.3)), NA)
})

test_that("confint() and get_CI() read the grid via the correct field name", {
  # Both access object$betas while glim() stores betas_evaluated. This works
  # only through partial matching with `$`, which is fragile.
  src_ci <- paste(deparse(body(GLIM::get_CI)), collapse = "\n")
  src_cf <- paste(deparse(body(getFromNamespace("confint.glim_object", "GLIM"))), collapse = "\n")
  expect_false(grepl("\\$betas\\b(?!_)", src_ci, perl = TRUE))
  expect_false(grepl("\\$betas\\b(?!_)", src_cf, perl = TRUE))
})

test_that("alpha validation uses a correctly placed length() check", {
  # `length(args$alpha == 1)` evaluates length of a logical vector, so
  # alpha = c(0.1, 0.2) passes the length check. Intended: rejected.
  d <- make_data("poisson")
  fit <- glim(
    y ~ x,
    data = d$dat,
    family = "poisson",
    betas = matrix(d$beta, 1),
    m = 5,
    parallel = FALSE
  )
  expect_error(get_CI(fit, alpha = c(0.1, 0.2)), "Alpha cutoff invalid")
})

test_that("gaussian family string 'normal' is handled consistently", {
  # glm_weights() accepts "normal" but glim()'s match.arg() does not list it.
  d <- make_data("gaussian")
  expect_error(
    glim(
      y ~ x,
      data = d$dat,
      family = "normal",
      betas = matrix(d$beta, 1),
      m = 5,
      parallel = FALSE
    ),
    NA
  )
})
