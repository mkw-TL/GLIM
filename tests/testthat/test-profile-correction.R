# correct_profile_point() and extend_profile(): the constrained-profile machinery.

correct_profile_point <- internal("correct_profile_point")
extend_profile <- internal("extend_profile")

# ---- helpers for the extend_profile() tests ----------------------------------
# A narrow band of the pre-computed grid, about one grid step either side of the
# MLE, so the plausible region lies well beyond the band and the extension has
# real work to do. The 1.5 absorbs floating-point error in the spacing.
narrow_pois_band <- function(col = 2) {
  gd <- fit_grid_pois()[["grid_data"]]
  b <- gd$betas_evaluated
  step <- median(diff(sort(unique(b[, col]))))
  keep <- abs(b[, col] - gd$mle_coefs[col]) <= 1.5 * step
  list(gd = gd, col = col, step = step, b = b[keep, , drop = FALSE], p = gd$possibilities[keep])
}

# extend_profile() on the narrow band. `...` carries the optional arguments
# (max_steps, n_bisect, growth, m_ext, seed0). m_ext is left at its default
# unless supplied; with m = 200 that is min(m, 300) = m, so nothing is
# re-evaluated at the end and the stored plausibilities are exactly the ones the
# walk saw.
call_extend <- function(band, dir, alpha = 0.05, tol = 0.01, m = 200, ...) {
  gd <- band$gd
  extend_profile(
    dir,
    TRUE,
    band$b,
    band$col,
    band$p,
    alpha,
    tol,
    band$step,
    gd$family,
    gd$X,
    gd$y,
    gd$dispersion,
    gd$mle_coefs,
    gd$logLik,
    m,
    FALSE,
    FALSE,
    ...
  )
}

edge_of <- function(band, dir) {
  if (dir > 0) max(band$b[, band$col]) else min(band$b[, band$col])
}

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

  # Rows come back sorted outward from the edge, and every point is beyond it.
  expect_false(is.unsorted(res$betas[, 2], strictly = TRUE))
  expect_true(all(res$betas[, 2] > mle[2]))

  # The first growth step is exactly one `step` out. It is not necessarily row 1
  # any more: if the walk crosses alpha immediately, the bisection points land
  # between the edge and that first step and sort in front of it.
  expect_true(any(abs(res$betas[, 2] - (mle[2] + 0.05)) < 1e-10))
})

test_that("extend_profile grows geometrically from the edge until plausibility falls below alpha", {
  band <- narrow_pois_band()
  alpha <- 0.05
  growth <- 1.5
  max_steps <- 40

  # Precondition: both edges of the narrow grid are plausible, so extension is
  # requested in both directions.
  edge_plaus <- function(v) max(band$p[band$b[, band$col] == v], na.rm = TRUE)
  expect_gt(edge_plaus(min(band$b[, band$col])), alpha)
  expect_gt(edge_plaus(max(band$b[, band$col])), alpha)

  for (dir in c(-1, 1)) {
    edge_val <- edge_of(band, dir)
    label <- paste("direction", dir)

    # n_bisect = 0 isolates the growth phase.
    ext <- call_extend(
      band,
      dir,
      alpha = alpha,
      max_steps = max_steps,
      n_bisect = 0,
      growth = growth
    )
    n <- nrow(ext$betas)

    # The contract: at least one step is taken, and betas and plausibilities
    # line up.
    expect_gte(n, 1, label = label)
    expect_equal(length(ext$plaus), n)

    # Step k moves the profiled coordinate step * growth^(k - 1) further out
    # from the previous point (the first from the edge), in the requested
    # direction.
    path <- c(edge_val, ext$betas[, band$col])
    expect_equal(
      diff(path),
      dir * band$step * growth^(seq_len(n) - 1),
      tolerance = 1e-8,
      ignore_attr = TRUE,
      label = label
    )

    # The walk stops at the first point below alpha: every earlier point is at
    # or above it, and the last point is the one that crossed (unless the walk
    # ran out of steps).
    if (n > 1) {
      expect_true(all(ext$plaus[-n] >= alpha), info = label)
    }
    if (n < max_steps) {
      expect_lt(ext$plaus[n], alpha, label = paste("last plausibility,", label))
    }
  }
})

test_that("extend_profile bisects between the last two growth points", {
  band <- narrow_pois_band()
  alpha <- 0.05
  tol <- 0.01
  n_bisect <- 6
  col <- band$col

  for (dir in c(-1, 1)) {
    edge_val <- edge_of(band, dir)
    label <- paste("direction", dir)

    # Same seeds and evaluation order for the growth phase in both runs (the
    # seed depends only on the evaluation counter), so the n_bisect = 0 run is
    # exactly the growth phase of the full run.
    grow <- call_extend(band, dir, alpha = alpha, tol = tol, n_bisect = 0)
    both <- call_extend(band, dir, alpha = alpha, tol = tol, n_bisect = n_bisect)

    n <- nrow(grow$betas)
    expect_gte(n, 1, label = label)

    # Output is sorted outward from the edge, strictly.
    expect_false(is.unsorted(dir * both$betas[, col], strictly = TRUE), label = label)

    # Every growth point is in the full result with the same plausibility.
    idx <- match(round(grow$betas[, col], 10), round(both$betas[, col], 10))
    expect_false(anyNA(idx), label = label)
    expect_equal(both$plaus[idx], grow$plaus, tolerance = 0, label = label)

    # The extra points are the bisection points: at most n_bisect of them, all
    # strictly between the last two growth points (or between the edge and the
    # first one, if it was the first that crossed alpha).
    extra <- setdiff(seq_len(nrow(both$betas)), idx)
    expect_lte(length(extra), n_bisect)
    expect_gte(length(extra), 1)
    lo <- if (n >= 2) grow$betas[n - 1, col] else edge_val
    hi <- grow$betas[n, col]
    expect_true(
      all(dir * both$betas[extra, col] > dir * lo & dir * both$betas[extra, col] < dir * hi),
      info = label
    )

    # The farthest point is still the growth point that crossed alpha.
    expect_equal(
      both$betas[nrow(both$betas), col],
      grow$betas[n, col],
      ignore_attr = TRUE,
      label = label
    )

    # Bisection stops early only when a point lands within tolerance of alpha.
    if (length(extra) < n_bisect) {
      expect_true(any(abs(both$plaus[extra] - alpha) <= tol), info = label)
    }
  }
})

test_that("extend_profile re-evaluates the accepted points at full m when m_ext < m", {
  d <- make_data("poisson", n = 40)
  glmfit <- glm(y ~ x, data = d$dat, family = poisson)
  mle <- unname(coef(glmfit))
  m <- 97 # chosen so that proportions out of m_ext = 20 are not proportions out of m
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
    m = m,
    parallel = FALSE,
    approx = FALSE,
    m_ext = 20
  )
  expect_gt(nrow(res$betas), 0)
  expect_equal(nrow(res$betas), length(res$plaus))
  expect_false(is.unsorted(res$betas[, 2], strictly = TRUE))
  # Plausibilities are k / m with m = 97 (the stepping itself used 20 replicates;
  # a value k / 20 is only also a multiple of 1 / 97 when it is 0 or 1).
  expect_true(all(abs(res$plaus * m - round(res$plaus * m)) < 1e-8))
})

test_that("glim() extensions run outward from the grid edge and end near the cutoff", {
  gd <- fit_grid_pois()[["grid_data"]]
  alpha <- 0.05
  for (col in 1:2) {
    for (side in c("pos", "neg")) {
      dir <- if (side == "pos") 1 else -1
      edge <- if (side == "pos") {
        max(gd$betas_evaluated[, col])
      } else {
        min(gd$betas_evaluated[, col])
      }
      pl <- gd[[paste0("extended_in_", side, "_plaus")]][[col]]
      bt <- gd[[paste0("extended_in_", side, "_betas")]][[col]]
      if (length(pl) == 0) {
        next
      }
      label <- paste("column", col, side)

      expect_equal(nrow(bt), length(pl), label = label)
      # sorted from the grid edge outward (growth points and bisection points)
      expect_true(all(dir * diff(c(edge, bt[, col])) > 0), info = label)
      expect_true(all(pl >= 0 & pl <= 1), info = label)
      # The farthest point is the one that crossed alpha during the walk. The
      # accepted points are re-evaluated at full m when m > 300, so allow Monte
      # Carlo slack rather than requiring it to still be below alpha.
      expect_lt(utils::tail(pl, 1), 2 * alpha, label = label)
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
      alpha = -1, # plausibility is never below this, so alpha is never crossed
      tolerance = 0,
      step = 1e-4,
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
    "no crossing of alpha"
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
