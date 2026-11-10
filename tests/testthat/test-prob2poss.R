# prob2poss() and the per-family prob2poss_*() helpers.
#
# Signatures:  prob2poss(family, X, y, mle, samples, the_compared_theta)
#              prob2poss_<family>(X, y, mle, samples, the_compared_theta)
# `samples` is p x m (one draw per column). The user-facing prob2poss() takes
# one evaluation point per ROW of `the_compared_theta` and transposes it.
# The internal helpers take one evaluation point per COLUMN.

prob2poss_poisson <- internal("prob2poss_poisson")
prob2poss_gamma <- internal("prob2poss_gamma")
prob2poss_gaussian <- internal("prob2poss_gaussian")
prob2poss_invgauss <- internal("prob2poss_invgauss")
prob2poss_logis <- internal("prob2poss_logis")

draw_samples <- function(center, sd, m) {
  set.seed(21)
  matrix(rnorm(length(center) * m, mean = center, sd = sd), nrow = length(center))
}

# ---- user-facing wrapper -----------------------------------------------------

test_that("prob2poss() input validation", {
  d <- make_data("gaussian")
  S <- matrix(rnorm(2 * 20, c(0.5, 0.3), 0.1), nrow = 2)
  mle <- c(0.5, 0.3)
  one <- matrix(c(0.5, 0.3), nrow = 1)
  expect_error(prob2poss("nope", d$X, d$y, mle, S, one), "incorrectly specified")
  expect_error(prob2poss("gaussian", d$X, d$y, mle, S, matrix(1, 3, 3)), "Dimension mismatch")
  expect_error(prob2poss("gaussian", d$X, d$y[-1], mle, S, one), "Dimension mismatch")
})

test_that("prob2poss() is in [0,1], ~1 near the sample centre and ~0 far away (gaussian)", {
  d <- make_data("gaussian", n = 80)
  ref <- lm(y ~ x, data = d$dat)
  mle <- unname(coef(ref))
  S <- draw_samples(mle, sd = sqrt(diag(vcov(ref))), m = 500)
  near <- prob2poss("gaussian", d$X, d$y, mle, S, matrix(mle, nrow = 1))
  far <- prob2poss("gaussian", d$X, d$y, mle, S, matrix(mle + c(5, 5), nrow = 1))
  expect_true(near >= 0 && near <= 1)
  expect_gt(near, 0.9)
  expect_lt(far, 0.01)
})

test_that("prob2poss() accepts several evaluation points as rows or as a data.frame", {
  d <- make_data("poisson", n = 40)
  ref <- glm(y ~ x, data = d$dat, family = poisson)
  mle <- unname(coef(ref))
  S <- draw_samples(mle, sd = sqrt(diag(vcov(ref))), m = 300)
  pts <- rbind(mle, mle + c(0.2, 0.2), mle + c(1, 1))
  out <- prob2poss("poisson", d$X, d$y, mle, S, pts)
  expect_length(out, 3)
  expect_equal(prob2poss("poisson", d$X, d$y, mle, S, as.data.frame(pts)), out)
  expect_true(out[1] >= out[3])
})

# ---- per-family helpers (theta: one column per evaluation point) --------------

test_that("prob2poss_poisson returns possibilities in [0, 1] with expected shape", {
  d <- make_poisson_data()
  S <- draw_samples(d$beta, 0.05, 200)
  theta <- draw_samples(d$beta, 0.05, 5)
  poss <- prob2poss_poisson(d$X, d$y, d$beta, S, theta)
  expect_length(poss, ncol(theta))
  expect_true(all(poss >= 0 & poss <= 1))
})

test_that("prob2poss_poisson coerces a data.frame X to a matrix", {
  d <- make_poisson_data()
  S <- draw_samples(d$beta, 0.05, 50)
  theta <- matrix(d$beta, ncol = 1)
  a <- prob2poss_poisson(d$X, d$y, d$beta, S, theta)
  b <- prob2poss_poisson(as.data.frame(d$X), d$y, d$beta, S, theta)
  expect_equal(a, b, tolerance = 1e-8)
})

test_that("prob2poss_gamma returns possibilities in [0, 1]", {
  d <- make_gamma_data()
  S <- draw_samples(d$beta, 0.02, 200)
  theta <- draw_samples(d$beta, 0.02, 5)
  poss <- prob2poss_gamma(d$X, d$y, d$beta, S, theta)
  expect_length(poss, ncol(theta))
  expect_true(all(poss >= 0 & poss <= 1))
})

test_that("prob2poss_gaussian returns possibilities in [0, 1]", {
  d <- make_gaussian_data()
  S <- draw_samples(d$beta, 0.1, 200)
  theta <- draw_samples(d$beta, 0.1, 5)
  poss <- prob2poss_gaussian(d$X, d$y, d$beta, S, theta)
  expect_length(poss, ncol(theta))
  expect_true(all(poss >= 0 & poss <= 1))
})

test_that("prob2poss_invgauss returns possibilities in [0, 1]", {
  d <- make_invgauss_data()
  S <- draw_samples(d$beta, 0.05, 200)
  theta <- draw_samples(d$beta, 0.05, 5)
  poss <- prob2poss_invgauss(d$X, d$y, d$beta, S, theta)
  expect_length(poss, ncol(theta))
  expect_true(all(poss >= 0 & poss <= 1))
})

test_that("prob2poss_logis handles a 0/1 response and rejects length mismatches", {
  d <- make_binomial_data()
  S <- draw_samples(d$beta, 0.05, 200)
  theta <- matrix(d$beta, ncol = 1)
  poss <- prob2poss_logis(d$X, d$y, d$beta, S, theta)
  expect_length(poss, 1)
  expect_true(poss >= 0 && poss <= 1)
  expect_error(prob2poss_logis(d$X, d$y[-1], d$beta, S, theta), "differ")
})

test_that("prob2poss_logis rejects responses that are not zeros and ones (vector input)", {
  d <- make_binomial_data()
  S <- draw_samples(d$beta, 0.05, 20)
  expect_error(
    prob2poss_logis(d$X, d$y + 2, d$beta, S, matrix(d$beta, ncol = 1)),
    "zeros or ones"
  )
})