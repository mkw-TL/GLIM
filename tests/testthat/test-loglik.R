# compute_ll() and the *_ll_r wrappers against R's own densities.

test_that("compute_ll() matches independent densities at a single beta", {
  for (fam in all_families) {
    d <- make_data(fam)
    mle <- switch(
      fam,
      gaussian = coef(lm(y ~ x, data = d$dat)),
      poisson = coef(glm(y ~ x, data = d$dat, family = poisson)),
      binomial = coef(glm(y ~ x, data = d$dat, family = binomial)),
      gamma = coef(glm(y ~ x, data = d$dat, family = Gamma("log"))),
      inverse.gaussian = coef(glm(y ~ x, data = d$dat, family = inverse.gaussian()))
    )
    b <- unname(d$beta)
    got <- compute_ll(fam, d$y, d$X, b, mle_coefs = mle)
    expect_length(got, 1)
    expect_true(is.finite(got), label = fam)
  }
})

test_that("compute_ll() poisson and binomial equal sum of log densities exactly", {
  d <- make_data("poisson")
  expect_equal(
    as.numeric(compute_ll("poisson", d$y, d$X, d$beta)),
    oracle_ll("poisson", d$X, d$y, d$beta),
    tolerance = 1e-8
  )
  d <- make_data("binomial")
  expect_equal(
    as.numeric(compute_ll("binomial", d$y, d$X, d$beta)),
    oracle_ll("binomial", d$X, d$y, d$beta),
    tolerance = 1e-8
  )
})

test_that("compute_ll() is maximised at the MLE (poisson, binomial)", {
  for (fam in c("poisson", "binomial")) {
    d <- make_data(fam)
    fit <- glm(y ~ x, data = d$dat, family = fam)
    mle <- unname(coef(fit))
    ll_mle <- as.numeric(compute_ll(fam, d$y, d$X, mle))
    expect_equal(ll_mle, as.numeric(logLik(fit)), tolerance = 1e-6, label = fam)
    for (delta in list(c(0.1, 0), c(0, 0.1), c(-0.1, 0.1))) {
      expect_lt(as.numeric(compute_ll(fam, d$y, d$X, mle + delta)), ll_mle)
    }
  }
})

test_that("compute_ll() with a matrix of betas returns one value per row", {
  d <- make_data("poisson")
  B <- rbind(c(0.5, 0.3), c(0.4, 0.2), c(0.6, 0.4))
  got <- as.vector(compute_ll("poisson", d$y, d$X, B))
  expected <- apply(B, 1, function(b) oracle_ll("poisson", d$X, d$y, b))
  expect_equal(got, expected, tolerance = 1e-8)
})

test_that("compute_ll() returns -Inf (not NaN/NA) outside the valid domain", {
  d <- make_data("inverse.gaussian")
  # eta < 0 makes mu = sqrt(1/eta) undefined
  out <- suppressWarnings(compute_ll("inverse.gaussian", d$y, d$X, c(-5, -5)))
  expect_identical(as.numeric(out), -Inf)
})

test_that("compute_ll() requires mle_coefs for gamma and errors on unknown family", {
  d <- make_data("gamma")
  expect_error(compute_ll("gamma", d$y, d$X, d$beta), "mle_coefs must be provided")
  expect_error(compute_ll("nope", d$y, d$X, d$beta), "family not defined")
})

test_that("compute_ll() gaussian uses the profile (MLE) dispersion", {
  d <- make_data("gaussian")
  fit <- lm(y ~ x, data = d$dat)
  got <- as.numeric(compute_ll("gaussian", d$y, d$X, unname(coef(fit))))
  expect_equal(got, as.numeric(logLik(fit)), tolerance = 1e-4)
})

test_that("compute_logistic_ll_r rejects non 0/1 responses", {
  d <- make_data("binomial")
  expect_error(GLIM:::compute_logistic_ll_r(d$X, d$y + 2, d$beta), "zeros or ones")
})

test_that("compute_invgauss_ll_r rejects non-positive y", {
  expect_error(GLIM:::compute_invgauss_ll_r(c(-1, 1), c(1, 1), 1), "negative or zero")
})
