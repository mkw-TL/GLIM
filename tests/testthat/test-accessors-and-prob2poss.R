# S3 methods (coef, logLik, print, plot) on the new output layouts.
# Accessors that are currently broken for some layouts are pinned in test-known-issues.R.

with_null_device <- function(code) {
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off(), add = TRUE)
  force(code)
}

test_that("coef() on the default grid fit matches glm()", {
  fit <- fit_grid_pois()
  d <- make_data("poisson", n = 40)
  ref <- glm(y ~ x, data = d$dat, family = poisson)
  expect_equal(unname(coef(fit)), unname(coef(ref)), tolerance = 1e-6)
})

test_that("logLik() on the default grid fit matches glm() and carries df / nobs", {
  fit <- fit_grid_pois()
  d <- make_data("poisson", n = 40)
  ref <- glm(y ~ x, data = d$dat, family = poisson)
  ll <- logLik(fit)
  expect_s3_class(ll, "logLik")
  expect_equal(as.numeric(ll), as.numeric(logLik(ref)), tolerance = 1e-6)
  expect_equal(attr(ll, "df"), 2)
  expect_equal(attr(ll, "nobs"), 40)
})

test_that("coef() and logLik() values work on an approx (margin) fit", {
  skip_on_cran()
  fit <- fit_approx_pois()
  d <- make_data("poisson", n = 40)
  ref <- glm(y ~ x, data = d$dat, family = poisson)
  expect_equal(unname(coef(fit)), unname(coef(ref)), tolerance = 1e-6)
  expect_equal(as.numeric(logLik(fit)), as.numeric(logLik(ref)), tolerance = 1e-6)
})

test_that("coef() and logLik() work when directly_calc_marginals keeps grid_data", {
  skip_on_cran()
  d <- make_data("poisson", n = 40)
  fit <- glim(
    y ~ x,
    data = d$dat,
    family = "poisson",
    m = 20,
    n_grid_evals = 5,
    directly_calc_marginals = TRUE,
    parallel = FALSE
  )
  ref <- glm(y ~ x, data = d$dat, family = poisson)
  expect_equal(unname(coef(fit)), unname(coef(ref)), tolerance = 1e-6)
  expect_equal(as.numeric(logLik(fit)), as.numeric(logLik(ref)), tolerance = 1e-6)
})

# ---- print ------------------------------------------------------------------

test_that("print() on a grid fit gives a table with MLE and interval columns", {
  fit <- fit_grid_pois()
  out <- capture.output(res <- print(fit))
  txt <- paste(out, collapse = "\n")
  expect_match(txt, "95% \\(marginal\\) confidence and credible region")
  expect_match(txt, "MLE")
  expect_match(txt, "Lower 95%")
  expect_match(txt, "Upper 95%")
  expect_match(txt, "\\(Intercept\\)")
  expect_match(txt, "\\bx\\b")
})

test_that("print() returns its argument invisibly", {
  fit <- fit_grid_pois()
  capture.output(v <- withVisible(print(fit)))
  expect_false(v$visible)
  expect_s3_class(v$value, "glim_object")
})

test_that("print() honours alpha", {
  fit <- fit_grid_pois()
  txt <- paste(capture.output(print(fit, alpha = 0.10)), collapse = "\n")
  expect_match(txt, "90% \\(marginal\\)")
  expect_match(txt, "Lower 9")
})

test_that("print() on a grid fit brackets the MLE where bounds are known", {
  fit <- fit_grid_pois()
  out <- capture.output(print(fit))
  rows <- out[grepl("^\\(Intercept\\)|^x ", out)]
  expect_length(rows, 2)
  nums <- lapply(strsplit(trimws(rows), "\\s+"), function(r) suppressWarnings(as.numeric(r[-1])))
  for (v in nums) {
    mle <- v[1]
    lo <- v[2]
    hi <- v[3]
    if (!is.na(lo)) {
      expect_lte(lo, mle + 1e-8)
    }
    if (!is.na(hi)) expect_gte(hi, mle - 1e-8)
  }
})

test_that("print() works on an approx (margin) fit and shows one row per coefficient", {
  skip_on_cran()
  fit <- fit_approx_pois()
  out <- capture.output(print(fit))
  expect_true(any(grepl("^\\(Intercept\\)", out)))
  expect_true(any(grepl("^x ", out)))
})

# ---- plot -------------------------------------------------------------------

test_that("plot() validates alpha before drawing", {
  fit <- fit_grid_pois()
  expect_error(plot(fit, alpha = 2), "Alpha cutoff invalid")
  expect_error(plot(fit, alpha = -0.5), "Alpha cutoff invalid")
  expect_error(plot(fit, alpha = c(0.1, 0.2)), "Alpha cutoff invalid")
})

test_that("plot() on an approx (margin) fit draws without error", {
  skip_on_cran()
  fit <- fit_approx_pois()
  with_null_device(expect_error(plot(fit), NA))
  with_null_device(expect_error(plot(fit, alpha = 0.05), NA))
})

test_that("plot() restores graphical parameters", {
  skip_on_cran()
  fit <- fit_approx_pois()
  with_null_device({
    before <- par("mfrow")
    plot(fit)
    expect_equal(par("mfrow"), before)
  })
})
