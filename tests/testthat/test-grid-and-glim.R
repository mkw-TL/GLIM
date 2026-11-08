# generate_marginal_object(): profile-likelihood walks away from the MLE for each coefficient.

generate_marginal_object <- internal("generate_marginal_object")
gradient <- internal("gradient")

# Correct setup helper to properly suppress messages on creation
setup_marg <- function() {
  cached("marg_setup", {
    d <- make_data("poisson", n = 40)
    fit <- glm(y ~ x, data = d$dat, family = poisson)
    mle <- unname(coef(fit))
    step <- sqrt(diag(vcov(fit))) / 2
    set.seed(10)
    mo <- suppressMessages(generate_marginal_object(
      X = d$X,
      y = d$y,
      family = "poisson",
      mle_coefs = mle,
      ll_mle_original_data = as.numeric(logLik(fit)),
      vcov = vcov(fit),
      n_grid_evals = 25,
      dispersion = 1,
      alpha = 0.05,
      step = step,
      m = 50,
      parallel = FALSE
    ))
    list(d = d, fit = fit, mle = mle, step = step, mo = mo)
  })
}

test_that("generate_marginal_object() returns a glim_object with margins + extra_info", {
  s <- setup_marg()
  expect_s3_class(s$mo, "glim_object")
  expect_named(s$mo, c("margins", "extra_info"))
  expect_length(s$mo$margins, 2)
  expect_equal(s$mo$extra_info$alpha, 0.05)
  expect_equal(s$mo$extra_info$family, "poisson")
  expect_equal(s$mo$extra_info$step, s$step)
  expect_equal(unname(s$mo$extra_info$mle_coefs), s$mle)
})

test_that("each margin starts at the MLE and extends on both sides of it", {
  s <- setup_marg()
  for (col in 1:2) {
    m <- s$mo$margins[[col]]
    expect_equal(unname(m$beta_for_margin[1, ]), s$mle)
    expect_equal(m$plaus_for_margin[1], 1)
    expect_lt(min(m$beta_for_margin[, col]), s$mle[col])
    expect_gt(max(m$beta_for_margin[, col]), s$mle[col])
    expect_true(all(m$plaus_for_margin >= 0 & m$plaus_for_margin <= 1))
  }
})

test_that("each margin walk stops once possibility falls to alpha - 0.01", {
  s <- setup_marg()
  for (col in 1:2) {
    pl <- s$mo$margins[[col]]$plaus_for_margin
    expect_lte(min(pl), 0.05 - 0.01)
  }
})

test_that("steps along the profile move by +/- step in the profiled coordinate", {
  s <- setup_marg()
  for (col in 1:2) {
    b <- s$mo$margins[[col]]$beta_for_margin[, col]
    # rows: MLE, then the negative-direction walk, then the positive-direction walk
    dev <- abs(b[-1] - s$mle[col])
    k <- round(dev / s$step[col])
    expect_equal(dev, k * s$step[col], tolerance = 1e-8)
  }
})

test_that("margin points are constrained optima (off-profile gradient ~ 0)", {
  s <- setup_marg()
  for (col in 1:2) {
    B <- s$mo$margins[[col]]$beta_for_margin
    for (i in seq_len(nrow(B))) {
      g <- gradient("poisson", s$d$X, s$d$y, B[i, ])
      expect_lt(max(abs(g[-col])), 1e-3, label = sprintf("col %d row %d", col, i))
    }
  }
})
