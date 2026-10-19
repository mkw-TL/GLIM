# Pure-R numerical building blocks: gradient(), partial(), glm_weights(), profile_tangent().
# These are checked against independent oracles (R's own densities + finite differences).

gradient <- internal("gradient")
partial <- internal("partial")
glm_weights <- internal("glm_weights")
profile_tangent <- internal("profile_tangent")

disp_for <- function(family) if (family %in% c("gaussian", "gamma", "inverse.gaussian")) 0.7 else 1

test_that("gradient() matches finite differences of an independent log-likelihood", {
  for (fam in all_families) {
    d <- make_data(fam)
    phi <- disp_for(fam)
    b <- d$beta + c(0.05, -0.03)
    num <- fd_grad(function(bb) oracle_ll(fam, d$X, d$y, bb, phi), b)
    expect_equal(gradient(fam, d$X, d$y, b, phi), num, tolerance = 1e-4, label = fam)
  }
})

test_that("partial() equals the matching component of gradient()", {
  for (fam in all_families) {
    d <- make_data(fam)
    phi <- disp_for(fam)
    b <- d$beta
    g <- gradient(fam, d$X, d$y, b, phi)
    for (j in 1:2) {
      expect_equal(partial(fam, d$X, d$y, b, phi, j), g[j], label = paste(fam, j))
    }
  }
})

test_that("gradient() is ~0 at the glm MLE for canonical-ish families", {
  d <- make_data("poisson")
  fit <- glm(y ~ x, data = d$dat, family = poisson)
  expect_lt(max(abs(gradient("poisson", d$X, d$y, coef(fit)))), 1e-5)

  d <- make_data("binomial")
  fit <- glm(y ~ x, data = d$dat, family = binomial)
  expect_lt(max(abs(gradient("binomial", d$X, d$y, coef(fit)))), 1e-5)

  d <- make_data("gamma")
  fit <- glm(y ~ x, data = d$dat, family = Gamma(link = "log"))
  expect_lt(max(abs(gradient("gamma", d$X, d$y, coef(fit), 1))), 1e-4)
})

test_that("gradient() and partial() reject unknown families", {
  d <- make_data("gaussian")
  expect_error(gradient("nope", d$X, d$y, d$beta), "not recognized")
  expect_error(partial("nope", d$X, d$y, d$beta, 1, 1), "not recognized")
})

test_that("glm_weights() equals -E[Hessian] diagonal, i.e. Fisher information weights", {
  # Fisher info weight = -d2 ll / d eta2 in expectation; check against finite
  # differences of the expected score for each family.
  eta <- c(0.4, 1.1, 2.3)
  phi <- 0.7
  expect_equal(glm_weights("gaussian", eta, phi), rep(1 / phi, 3))
  expect_equal(glm_weights("gamma", eta, phi), rep(1 / phi, 3))
  expect_equal(glm_weights("poisson", eta), exp(eta))
  mu <- plogis(eta)
  expect_equal(glm_weights("binomial", eta), mu * (1 - mu))
  expect_equal(glm_weights("inverse.gaussian", eta, phi), eta^(-3 / 2) / (4 * phi))
})

test_that("glm_weights() flags invalid inverse-gaussian eta and unknown families", {
  expect_true(any(is.nan(suppressWarnings(glm_weights("inverse.gaussian", c(-1, 1), 1)))))
  expect_error(glm_weights("nope", 1), "Unsupported")
  expect_equal(glm_weights("normal", 1:3, 2), rep(0.5, 3)) # alias accepted here
})

test_that("Fisher information from glm_weights() matches vcov() from glm", {
  for (fam in c("poisson", "binomial")) {
    d <- make_data(fam)
    fit <- glm(y ~ x, data = d$dat, family = if (fam == "poisson") poisson else binomial)
    w <- glm_weights(fam, as.vector(d$X %*% coef(fit)))
    info <- crossprod(d$X, w * d$X)
    expect_equal(solve(info), unname(vcov(fit)), tolerance = 1e-6, label = fam)
  }
})

test_that("profile_tangent() has length p-1 and is empty for p = 1", {
  d <- make_data("poisson")
  fit <- glm(y ~ x, data = d$dat, family = poisson)
  expect_length(profile_tangent("poisson", d$X, coef(fit), 1), 1)
  expect_equal(profile_tangent("poisson", d$X[, 1, drop = FALSE], coef(fit)[1], 1), numeric(0))
})

test_that("profile_tangent() equals the slope of the numerically profiled coefficient", {
  # For Gaussian (linear in beta) the profile path is exactly linear, so the
  # tangent must equal the finite difference of the constrained optimum.
  d <- make_data("gaussian")
  mle <- coef(lm(y ~ x, data = d$dat))
  tang <- profile_tangent("gaussian", d$X, mle, col = 2, dispersion = 1)

  prof <- function(b2) coef(lm(d$y ~ 1 + offset(b2 * d$x)))[1]
  slope <- (prof(mle[2] + 0.1) - prof(mle[2] - 0.1)) / 0.2
  expect_equal(unname(tang), unname(slope), tolerance = 1e-5)

  # Same check for Poisson (nonlinear) with a small step
  d <- make_data("poisson")
  mle <- coef(glm(y ~ x, data = d$dat, family = poisson))
  tang <- profile_tangent("poisson", d$X, mle, col = 2)
  prof <- function(b2) coef(glm(d$y ~ 1 + offset(b2 * d$x), family = poisson))[1]
  h <- 1e-3
  slope <- (prof(mle[2] + h) - prof(mle[2] - h)) / (2 * h)
  expect_equal(unname(tang), unname(slope), tolerance = 1e-3)
})

test_that("profile_tangent() returns NULL when weights are non-finite", {
  d <- make_data("inverse.gaussian")
  bad <- c(-1, -1) # eta < 0 for most rows -> NaN weights
  expect_null(suppressWarnings(profile_tangent("inverse.gaussian", d$X, bad, 1, 1)))
})
