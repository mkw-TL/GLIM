# correct_profile_point() and extend_profile(): the new constrained-profile machinery.

correct_profile_point <- internal("correct_profile_point")
extend_profile <- internal("extend_profile")

test_that("correct_profile_point() recovers the constrained optimum (poisson)", {
  d <- make_data("poisson")
  mle <- coef(glm(y ~ x, data = d$dat, family = poisson))
  target <- mle[2] + 0.3
  start <- c(mle[1] + 0.2, target) # deliberately wrong intercept
  out <- correct_profile_point("poisson", d$X, d$y, start, col = 2, mle_coefs = mle)
  ref <- coef(glm(d$y ~ 1 + offset(target * d$x), family = poisson))[1]
  expect_equal(out[2], target, ignore_attr = TRUE) # fixed column untouched
  expect_equal(unname(out[1]), unname(ref), tolerance = 1e-5)
})

test_that("correct_profile_point() never decreases the log-likelihood", {
  for (fam in c("poisson", "binomial", "gaussian")) {
    d <- make_data(fam)
    mle <- unname(switch(
      fam,
      poisson = coef(glm(y ~ x, data = d$dat, family = poisson)),
      binomial = coef(glm(y ~ x, data = d$dat, family = binomial)),
      gaussian = coef(lm(y ~ x, data = d$dat))
    ))
    start <- c(mle[1] + 0.5, mle[2] + 0.2)
    out <- correct_profile_point(fam, d$X, d$y, start, col = 2, mle_coefs = mle)
    expect_gte(
      as.numeric(compute_ll(fam, d$y, d$X, out, mle)),
      as.numeric(compute_ll(fam, d$y, d$X, start, mle)) - 1e-10,
      label = fam
    )
  }
})

test_that("correct_profile_point() is a no-op when already at the constrained optimum", {
  d <- make_data("poisson")
  mle <- unname(coef(glm(y ~ x, data = d$dat, family = poisson)))
  out <- correct_profile_point("poisson", d$X, d$y, mle, col = 2, mle_coefs = mle)
  expect_equal(out, mle, tolerance = 1e-6)
})

test_that("correct_profile_point() returns input unchanged if start has non-finite ll", {
  d <- make_data("inverse.gaussian")
  start <- c(-5, -5)
  out <- suppressWarnings(correct_profile_point("inverse.gaussian", d$X, d$y, start, col = 2, mle_coefs = start))
  expect_equal(out, start)
})

test_that("correct_profile_point() works when p = 1 (no off-diagonal params)", {
  d <- make_data("poisson")
  X1 <- d$X[, 1, drop = FALSE]
  b <- coef(glm(d$y ~ X1 - 1, family = poisson))
  out <- correct_profile_point("poisson", X1, d$y, unname(b), col = 1, mle_coefs = unname(b))
  expect_equal(out, unname(b))
})

test_that("extend_profile() returns empty result when no extension is needed", {
  res <- extend_profile(
    direction = 1, needs_extension = FALSE, betas = matrix(0, 1, 2), col = 1,
    possibilities = 1, alpha = 0.05, tolerance = 0.01, step = 0.1,
    family = "poisson", X = matrix(1, 1, 2), y = 1, dispersion = 1,
    mle_coefs = c(0, 0), ll_mle_original_data = 0, m = 5, parallel = FALSE,
    approx = FALSE, base_seed = 1
  )
  expect_equal(nrow(res$betas), 0)
  expect_length(res$plaus, 0)
})

test_that("extend_profile() walks outward monotonically in the requested direction", {
  d <- make_data("poisson", n = 40)
  fit <- glim(y ~ x, data = d$dat, family = "poisson", m = 50, n_grid_evals = 5, parallel = FALSE)
  col <- 2
  for (dir in c(-1, 1)) {
    ext <- if (dir > 0) fit$extended_in_pos_betas[[col]] else fit$extended_in_neg_betas[[col]]
    if (length(ext) > 0) {
      steps <- diff(ext[, col])
      expect_true(all(sign(steps) == dir), info = paste("direction", dir))
    }
  }
})

test_that("extend_profile() stops once plausibility drops below alpha - tolerance", {
  d <- make_data("poisson", n = 40)
  fit <- glim(y ~ x, data = d$dat, family = "poisson", m = 50, n_grid_evals = 5, parallel = FALSE)
  for (col in 1:2) {
    for (pl in list(fit$extended_in_pos_plaus[[col]], fit$extended_in_neg_plaus[[col]])) {
      if (length(pl) > 1) {
        # every point before the last must still be above the stopping threshold
        expect_true(all(utils::head(pl, -1) > 0.05 - 0.01))
      }
    }
  }
})

test_that("extend_profile() respects max_steps with a warning", {
  d <- make_data("poisson", n = 40)
  mle <- unname(coef(glm(y ~ x, data = d$dat, family = poisson)))
  ll <- as.numeric(logLik(glm(y ~ x, data = d$dat, family = poisson)))
  expect_warning(
    extend_profile(
      direction = 1, needs_extension = TRUE, betas = matrix(mle, 1), col = 2,
      possibilities = 1, alpha = -1, tolerance = 0, step = 1e-4, # never reaches alpha
      family = "poisson", X = d$X, y = d$y, dispersion = 1, mle_coefs = mle,
      ll_mle_original_data = ll, m = 5, parallel = FALSE, approx = FALSE,
      base_seed = 1, max_steps = 3
    ),
    "max_steps"
  )
})
