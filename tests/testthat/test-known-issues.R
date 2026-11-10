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
