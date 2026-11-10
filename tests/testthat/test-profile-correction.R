# correct_profile_point() and extend_profile(): the constrained-profile machinery.

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
  out <- suppressWarnings(correct_profile_point(
    "inverse.gaussian",
    d$X,
    d$y,
    start,
    col = 2,
    mle_coefs = start
  ))
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
    direction = 1,
    needs_extension = FALSE,
    betas = matrix(0, 1, 2),
    col = 1,
    possibilities = 1,
    alpha = 0.05,
    tolerance = 0.01,
    step = 0.1,
    family = "poisson",
    X = matrix(1, 1, 2),
    y = 1,
    dispersion = 1,
    mle_coefs = c(0, 0),
    ll_mle_original_data = 0,
    m = 5,
    parallel = FALSE,
    approx = FALSE
  )
  expect_named(res, c("betas", "plaus"))
  expect_equal(dim(res$betas), c(0, 2))
  expect_length(res$plaus, 0)
})

test_that("extend_profile() returns a p-column matrix plus matching plausibilities", {
  d <- make_data("poisson", n = 40)
  glmfit <- glm(y ~ x, data = d$dat, family = poisson)
  mle <- unname(coef(glmfit))
  res <- extend_profile(
    direction = 1,
    needs_extension = TRUE,
    betas = matrix(mle, 1),
    col = 2,
    possibilities = 1,
    alpha = 0.05,
    tolerance = 0.01,
    step = 0.05,
    family = "poisson",
    X = d$X,
    y = d$y,
    dispersion = 1,
    mle_coefs = mle,
    ll_mle_original_data = as.numeric(logLik(glmfit)),
    m = 50,
    parallel = FALSE,
    approx = FALSE
  )
  expect_equal(ncol(res$betas), 2)
  expect_equal(nrow(res$betas), length(res$plaus))
  expect_gt(nrow(res$betas), 0)
  expect_equal(res$betas[1, 2], mle[2] + 0.05, tolerance = 1e-10)
})
test_that("extend_profile walks outward and stops once plausibility falls below alpha", {
  gd <- fit_grid_pois()[["grid_data"]]
  col <- 2
  alpha <- 0.05
  tol <- 0.01
  max_steps <- 200
  b <- gd$betas_evaluated

  # Cut the grid down to a narrow band around the MLE (about one step either
  # side) so the plausible region lies well beyond the band and the extension
  # has real work to do. The 1.5 absorbs floating-point error in the spacing.
  step <- median(diff(sort(unique(b[, col]))))
  keep <- abs(b[, col] - gd$mle_coefs[col]) <= 1.5 * step
  b_small <- b[keep, , drop = FALSE]
  p_small <- gd$possibilities[keep]

  # Precondition: both edges of the narrow grid are plausible, so extension is
  # requested in both directions.
  edge_plaus <- function(v) max(p_small[b_small[, col] == v], na.rm = TRUE)
  expect_gt(edge_plaus(min(b_small[, col])), alpha)
  expect_gt(edge_plaus(max(b_small[, col])), alpha)

  for (dir in c(-1, 1)) {
    edge_val <- if (dir > 0) max(b_small[, col]) else min(b_small[, col])
    ext <- extend_profile(
      dir,
      TRUE,
      b_small,
      col,
      p_small,
      alpha,
      tol,
      step,
      gd$family,
      gd$X,
      gd$y,
      gd$dispersion,
      gd$mle_coefs,
      gd$logLik,
      200,
      FALSE,
      FALSE,
      max_steps = max_steps
    )
    n <- nrow(ext$betas)
    label <- paste("direction", dir)

    # The contract: at least one step is taken, and betas and plausibilities
    # line up.
    expect_gte(n, 1, label = label)
    expect_equal(length(ext$plaus), n)

    # Each step moves the profiled coordinate exactly one step outward from the
    # edge, in the requested direction.
    path <- c(edge_val, ext$betas[, col])
    expect_true(all(sign(diff(path)) == dir), info = label)

    # The walk only stops when a point is at or below alpha - tolerance. So
    # every point except the last is still above that level, and the last point
    # is the one that ended the walk (unless max_steps was exhausted).
    if (n > 1) {
      expect_true(all(ext$plaus[-n] > alpha - tol), info = label)
    }
    if (n < max_steps) {
      expect_lte(ext$plaus[n], alpha - tol, label = paste("last plausibility,", label))
    }
  }
})

test_that("glim() extensions stop once plausibility drops below alpha - tolerance", {
  gd <- fit_grid_pois()[["grid_data"]]
  for (col in 1:2) {
    for (pl in list(gd$extended_in_pos_plaus[[col]], gd$extended_in_neg_plaus[[col]])) {
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
      direction = 1,
      needs_extension = TRUE,
      betas = matrix(mle, 1),
      col = 2,
      possibilities = 1,
      alpha = -1,
      tolerance = 0,
      step = 1e-4, # never reaches alpha
      family = "poisson",
      X = d$X,
      y = d$y,
      dispersion = 1,
      mle_coefs = mle,
      ll_mle_original_data = ll,
      m = 5,
      parallel = FALSE,
      approx = FALSE,
      max_steps = 3
    ),
    "max_steps"
  )
})

test_that("extend_profile returns an empty result when no extension is requested", {
  gd <- fit_grid_pois()[["grid_data"]]
  p <- length(gd$mle_coefs)

  for (dir in c(-1, 1)) {
    ext <- extend_profile(
      dir,
      FALSE,
      gd$betas_evaluated,
      2,
      gd$possibilities,
      0.05,
      0.01,
      0.1,
      gd$family,
      gd$X,
      gd$y,
      gd$dispersion,
      gd$mle_coefs,
      gd$logLik,
      50,
      FALSE,
      FALSE
    )
    expect_equal(nrow(ext$betas), 0)
    expect_equal(ncol(ext$betas), p)
    expect_length(ext$plaus, 0)
  }
})

# .grid_bounds() on hand-built objects: no model fitting, no Monte Carlo noise.
test_that(".grid_bounds uses only plausible extension points", {
  g <- list(
    betas_evaluated = cbind(b0 = c(-1, 0, 1)),
    possibilities = c(0.5, 1, 0.5), # plausible all the way to both edges
    extended_in_neg_betas = list(cbind(b0 = c(-2, -3))),
    extended_in_neg_plaus = list(c(0.3, 0.01)), # -3 is the point that ended the walk
    extended_in_pos_betas = list(cbind(b0 = 2)),
    extended_in_pos_plaus = list(0.01) # the only extension point is implausible
  )
  b <- .grid_bounds(g, 0.05)
  expect_equal(b$lower, -2) # the implausible -3 is dropped
  expect_equal(b$upper, 1) # nothing plausible past the edge: fall back to the edge
})

test_that(".grid_bounds gives NA when the edge is reached and no extension exists", {
  g <- list(betas_evaluated = cbind(b0 = c(-1, 0, 1)), possibilities = c(0.5, 1, 0.5))
  b <- .grid_bounds(g, 0.05)
  expect_true(is.na(b$lower))
  expect_true(is.na(b$upper))
})

test_that(".grid_bounds uses the grid directly when plausibility falls inside it", {
  g <- list(betas_evaluated = cbind(b0 = c(-1, 0, 1)), possibilities = c(0.01, 1, 0.01))
  b <- .grid_bounds(g, 0.05)
  expect_equal(b$lower, 0)
  expect_equal(b$upper, 0)
})
