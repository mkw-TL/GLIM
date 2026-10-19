# S3 accessors, get_CI helpers and prob2poss.

fit_pois <- local({
  d <- make_data("poisson", n = 40)
  set.seed(5)
  list(
    d = d,
    fit = glim(y ~ x, data = d$dat, family = "poisson", m = 50, n_grid_evals = 7, parallel = FALSE)
  )
})

test_that("coef / logLik / nobs / vcov accessors are consistent with glm", {
  fit <- fit_pois$fit
  ref <- glm(y ~ x, data = fit_pois$d$dat, family = poisson)
  expect_equal(unname(coef(fit)), unname(coef(ref)), tolerance = 1e-6)
  expect_equal(as.numeric(logLik(fit)), as.numeric(logLik(ref)), tolerance = 1e-6)
  expect_equal(attr(logLik(fit), "df"), 2)
  expect_equal(attr(logLik(fit), "nobs"), 40)
  expect_equal(nobs(fit), 40)
  expect_equal(unname(vcov(fit)), unname(vcov(ref)), tolerance = 1e-6)
})

test_that("get_CI_manual() filters rows above alpha", {
  betas <- cbind(a = c(1, 2, 3, 4), b = c(5, 6, 7, 8))
  poss <- c(0.01, 0.2, 1, 0.04)
  out <- get_CI_manual(0.05, betas, poss)
  expect_equal(nrow(out), 2)
  expect_equal(unname(out[, "a"]), c(2, 3))
})

test_that("get_CI_manual() with one surviving row still returns a matrix-like object", {
  betas <- cbind(a = 1:3, b = 4:6)
  out <- get_CI_manual(0.5, betas, c(0.1, 0.9, 0.2))
  # `betas[mask, ]` drops to a vector for a single row; callers expect a matrix.
  expect_true(is.matrix(out))
})

test_that("get_CI() works on a grid glim_object", {
  # Uses object$betas, but the field is named betas_evaluated, so partial
  # matching ($betas does not partial-match betas_evaluated for lists with
  # `$`... it does for exact=FALSE) -- either way the result must be correct.
  ci <- get_CI(fit_pois$fit, alpha = 0.05)
  expect_true(is.matrix(ci))
  expect_equal(ncol(ci), 2)
  expect_true(nrow(ci) >= 1)
})

test_that("get_CI() validates input", {
  expect_error(get_CI(list()), "not a direct result")
  expect_error(get_CI(fit_pois$fit, alpha = 2), "Alpha cutoff invalid")
  expect_error(get_CI(fit_pois$fit, alpha = -1), "Alpha cutoff invalid")
})

test_that("get_CI() region shrinks as alpha grows", {
  big <- get_CI(fit_pois$fit, alpha = 0.01)
  small <- get_CI(fit_pois$fit, alpha = 0.5)
  expect_gte(NROW(big), NROW(small))
})

test_that("confint() brackets the MLE and is ordered", {
  ci <- confint(fit_pois$fit, alpha = 0.05)
  expect_equal(dim(ci), c(2, 2))
  expect_equal(rownames(ci), colnames(fit_pois$fit$X))
  mle <- coef(fit_pois$fit)
  ok <- !is.na(ci[, 1]) & !is.na(ci[, 2])
  expect_true(all(ci[ok, 1] <= mle[ok] + 1e-8))
  expect_true(all(ci[ok, 2] >= mle[ok] - 1e-8))
})

test_that("confint() column labels reflect alpha/2", {
  ci <- confint(fit_pois$fit, alpha = 0.10)
  expect_equal(colnames(ci), c("5.0 %", "95.0 %"))
})

test_that("prob2poss() input validation", {
  d <- make_data("gaussian")
  S <- matrix(rnorm(2 * 20, c(0.5, 0.3), 0.1), nrow = 2)
  expect_error(prob2poss("nope", d$X, d$y, S, c(0.5, 0.3)), "incorrectly specified")
  expect_error(prob2poss("gaussian", d$X, d$y, S, matrix(1, 3, 3)), "Dimension mismatch")
  expect_error(prob2poss("gaussian", d$X, d$y[-1], S, c(0.5, 0.3)), "Dimension mismatch")
})

test_that("prob2poss() is in [0,1], and ~1 near the sample centre / ~0 far away (gaussian)", {
  d <- make_data("gaussian", n = 80)
  mle <- unname(coef(lm(y ~ x, data = d$dat)))
  set.seed(2)
  cov <- vcov(lm(y ~ x, data = d$dat))
  S <- t(MASS::mvrnorm(500, mle, cov))
  near <- prob2poss("gaussian", d$X, d$y, S, mle)
  far <- prob2poss("gaussian", d$X, d$y, S, mle + c(5, 5))
  expect_true(near >= 0 && near <= 1)
  expect_gt(near, 0.9)
  expect_lt(far, 0.01)
})

test_that("prob2poss() accepts multiple evaluation points as rows or a data.frame", {
  d <- make_data("poisson", n = 40)
  mle <- unname(coef(glm(y ~ x, data = d$dat, family = poisson)))
  set.seed(2)
  S <- t(MASS::mvrnorm(300, mle, vcov(glm(y ~ x, data = d$dat, family = poisson))))
  pts <- rbind(mle, mle + c(0.2, 0.2), mle + c(1, 1))
  out <- prob2poss("poisson", d$X, d$y, S, pts)
  expect_length(out, 3)
  expect_equal(prob2poss("poisson", d$X, d$y, S, as.data.frame(pts)), out)
  expect_true(out[1] >= out[3])
})
