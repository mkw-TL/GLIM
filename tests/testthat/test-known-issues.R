# Tests that pin down suspected bugs found by reading the code. Each states the
# *intended* behaviour, so a failure here points at a real defect.

test_that("prob2poss_poisson accepts whole-number doubles (e.g. from a data.frame)", {
  # Code checks is.integer(y), which is FALSE for c(1, 2, 3) stored as double,
  # even though glim() itself accepts such y.
  set.seed(1)
  x <- rnorm(30)
  y <- as.numeric(rpois(30, exp(0.3 * x)))
  X <- cbind(1, x)
  S <- matrix(c(0, 0.3), 2, 50) + matrix(rnorm(100, sd = 0.05), 2)
  expect_error(prob2poss("poisson", X, y, S, c(0, 0.3)), NA)
})

test_that("calculate_marginal_ftn loops over every column and defines its variables", {
  # `seq_along(ncol(X))` iterates once (col = 1) regardless of p, and
  # dispersion / step / off_idx / out_betas are never defined in that function.
  f <- internal("calculate_marginal_ftn")
  body_txt <- paste(deparse(body(f)), collapse = "\n")
  expect_false(grepl("seq_along\\(ncol\\(X\\)\\)", body_txt))
  expect_true(all(c("dispersion", "step") %in% names(formals(f))) ||
    grepl("dispersion <-", body_txt))
})

test_that("generate_marginal_object / glim_marginal are implemented (not empty stubs)", {
  gm <- internal("generate_marginal_object")
  expect_gt(length(deparse(body(gm))), 3)
  expect_false(is.null(gm(matrix(1, 2, 1), c(1, 2), "poisson", 0, 0, 1, FALSE, 1)))
})

test_that("generate_grid does not sleep (Sys.sleep(1) left in code)", {
  f <- internal("generate_grid")
  expect_false(grepl("Sys.sleep", paste(deparse(body(f)), collapse = "\n")))
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
  fit <- glim(y ~ x, data = d$dat, family = "poisson", betas = matrix(d$beta, 1), m = 5, parallel = FALSE)
  expect_error(get_CI(fit, alpha = c(0.1, 0.2)), "Alpha cutoff invalid")
})

test_that("gaussian family string 'normal' is handled consistently", {
  # glm_weights() accepts "normal" but glim()'s match.arg() does not list it.
  d <- make_data("gaussian")
  expect_error(
    glim(y ~ x, data = d$dat, family = "normal", betas = matrix(d$beta, 1), m = 5, parallel = FALSE),
    NA
  )
})
