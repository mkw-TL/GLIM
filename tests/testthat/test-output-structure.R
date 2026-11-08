# The shape of the object returned by glim() for every code path.
# See helper-data.R for the table of layouts.

expect_marginals_object <- function(mo, p, family = NULL) {
  expect_named(mo, c("margins", "extra_info"))
  expect_length(mo$margins, p)
  for (col in seq_len(p)) {
    m <- mo$margins[[col]]
    expect_named(m, c("beta_for_margin", "plaus_for_margin"))
    expect_true(is.matrix(m$beta_for_margin))
    expect_equal(ncol(m$beta_for_margin), p)
    expect_equal(nrow(m$beta_for_margin), length(m$plaus_for_margin))
    # first row is always the MLE with possibility 1
    expect_equal(unname(m$beta_for_margin[1, ]), unname(mo$extra_info$mle_coefs))
    expect_equal(m$plaus_for_margin[1], 1)
  }
  needed <- c(
    "X",
    "y",
    "family",
    "mle_coefs",
    "ll_mle_original_data",
    "n_grid_evals",
    "dispersion",
    "alpha",
    "step",
    "m",
    "parallel",
    "base_seed"
  )
  expect_true(all(needed %in% names(mo$extra_info)))
  if (!is.null(family)) {
    expect_equal(mo$extra_info$family, family)
  }
}

test_that("default grid path returns list(grid_data = ...)", {
  fit <- fit_grid_pois()
  expect_s3_class(fit, "glim_object")
  expect_named(fit, "grid_data")

  gd <- fit[["grid_data"]]
  expected <- c(
    "possibilities",
    "samples",
    "xi",
    "alphas",
    "coordinate_marginal_bounds",
    "betas_evaluated",
    "family",
    "X",
    "y",
    "mle_coefs",
    "logLik",
    "vcov",
    "dispersion",
    "extended_in_neg_plaus",
    "extended_in_neg_betas",
    "extended_in_pos_plaus",
    "extended_in_pos_betas"
  )
  expect_true(all(expected %in% names(gd)))

  expect_equal(gd$family, "poisson")
  expect_equal(gd$dispersion, 1)
  expect_equal(length(gd$possibilities), nrow(gd$betas_evaluated))
  expect_equal(colnames(gd$betas_evaluated), colnames(gd$X))
  expect_equal(dim(gd$vcov), c(2, 2))
  expect_length(gd$mle_coefs, 2)
  expect_true(is.numeric(gd$logLik) && length(gd$logLik) == 1)
  # grid path has no elliptical information
  expect_null(gd$samples)
  expect_null(gd$xi)
})

test_that("default grid path stores one extension (matrix + plausibilities) per coefficient and direction", {
  gd <- fit_grid_pois()[["grid_data"]]
  p <- 2
  for (nm in c(
    "extended_in_neg_plaus",
    "extended_in_pos_plaus",
    "extended_in_neg_betas",
    "extended_in_pos_betas"
  )) {
    expect_length(gd[[nm]], p)
  }
  for (col in seq_len(p)) {
    for (side in c("neg", "pos")) {
      b <- gd[[paste0("extended_in_", side, "_betas")]][[col]]
      pl <- gd[[paste0("extended_in_", side, "_plaus")]][[col]]
      expect_true(is.matrix(b))
      expect_equal(ncol(b), p)
      expect_equal(nrow(b), length(pl))
      expect_true(all(pl >= 0 & pl <= 1))
    }
  }
})

test_that("extensions start beyond the grid edge on the side they extend", {
  gd <- fit_grid_pois()[["grid_data"]]
  for (col in 1:2) {
    pos <- gd$extended_in_pos_betas[[col]]
    neg <- gd$extended_in_neg_betas[[col]]
    if (nrow(pos) > 0) {
      expect_gt(min(pos[, col]), max(gd$betas_evaluated[, col]))
    }
    if (nrow(neg) > 0) expect_lt(max(neg[, col]), min(gd$betas_evaluated[, col]))
  }
})

test_that("directly_calc_marginals = TRUE returns list(marginals, grid_data)", {
  d <- make_data("poisson", n = 40)
  set.seed(5)
  fit <- glim(
    y ~ x,
    data = d$dat,
    family = "poisson",
    m = 30,
    n_grid_evals = 5,
    directly_calc_marginals = TRUE,
    parallel = FALSE
  )
  expect_s3_class(fit, "glim_object")
  expect_named(fit, c("marginals", "grid_data"))
  expect_s3_class(fit[["marginals"]], "glim_object")
  expect_marginals_object(fit[["marginals"]], p = 2, family = "poisson")
  # the grid still exists but has no extensions (marginals replace them)
  expect_false("extended_in_pos_betas" %in% names(fit[["grid_data"]]))
  expect_true("possibilities" %in% names(fit[["grid_data"]]))
})

test_that("custom betas return the flat grid object (no wrapper)", {
  d <- make_data("poisson", n = 40)
  B <- matrix(c(0.5, 0.3, 0.4, 0.2), 2, byrow = TRUE)
  fit <- glim(y ~ x, data = d$dat, family = "poisson", betas = B, m = 20, parallel = FALSE)
  expect_s3_class(fit, "glim_object")
  expect_null(fit[["grid_data"]])
  expect_true(all(
    c(
      "possibilities",
      "betas_evaluated",
      "mle_coefs",
      "X",
      "y",
      "family",
      "logLik",
      "vcov",
      "dispersion"
    ) %in%
      names(fit)
  ))
  expect_equal(unname(fit$betas_evaluated), unname(B))
  expect_equal(length(fit$possibilities), 2)
})

test_that("custom betas + directly_calc_marginals returns list(glim_object, marginals)", {
  d <- make_data("poisson", n = 40)
  fit <- glim(
    y ~ x,
    data = d$dat,
    family = "poisson",
    betas = matrix(c(0.5, 0.3), 1),
    directly_calc_marginals = TRUE,
    m = 20,
    n_grid_evals = 5,
    parallel = FALSE
  )
  expect_named(fit, c("glim_object", "marginals"))
  expect_true("possibilities" %in% names(fit[["glim_object"]]))
  expect_marginals_object(fit[["marginals"]], p = 2)
})

test_that("more than 5 predictors without betas returns a bare marginals object", {
  skip_on_cran()
  set.seed(3)
  n <- 80
  d <- as.data.frame(matrix(rnorm(n * 5), n))
  d$y <- rnorm(n)
  res <- suppressMessages(glim(
    y ~ .,
    data = d,
    family = "gaussian",
    m = 20,
    n_grid_evals = 7,
    parallel = FALSE
  ))
  expect_s3_class(res, "glim_object")
  expect_named(res, c("margins", "extra_info"))
  expect_marginals_object(res, p = 6, family = "gaussian")
})

test_that("approx = TRUE returns list(margin, elliptical_data)", {
  skip_on_cran()
  fit <- fit_approx_pois()
  expect_s3_class(fit, "glim_object")
  expect_named(fit, c("margin", "elliptical_data"))
  expect_marginals_object(fit[["margin"]], p = 2, family = "poisson")

  ed <- fit[["elliptical_data"]]
  expect_named(ed, c("samples", "xi", "alphas", "coordinate_marginal_bounds"))
  expect_equal(dim(ed$samples), c(2, 20)) # p x m
  expect_length(ed$xi, 99)
  expect_length(ed$alphas, 99)
  expect_length(ed$coordinate_marginal_bounds, 99)
  expect_true(all(vapply(ed$xi, length, integer(1)) == 2))
  expect_true(all(vapply(ed$coordinate_marginal_bounds, length, integer(1)) == 2))
  expect_equal(ed$alphas, seq(0.01, 0.99, length.out = 99))
  expect_true(all(is.finite(ed$samples)))
})

test_that("radial = TRUE returns list(margin, radial_data)", {
  skip_on_cran()
  d <- make_data("poisson", n = 40)
  set.seed(5)
  fit <- glim(
    y ~ x,
    data = d$dat,
    family = "poisson",
    radial = TRUE,
    m = 20,
    n_grid_evals = 5,
    num_samps = 20,
    parallel = FALSE
  )
  expect_s3_class(fit, "glim_object")
  expect_named(fit, c("margin", "radial_data"))
  expect_marginals_object(fit[["margin"]], p = 2, family = "poisson")

  rd <- fit[["radial_data"]]
  expect_true(all(
    c(
      "samples",
      "betas_evaluated",
      "family",
      "X",
      "y",
      "mle_coefs",
      "logLik",
      "vcov",
      "dispersion"
    ) %in%
      names(rd)
  ))
  expect_equal(rd$family, "poisson")
  # radial() hands back the whole generate_grid() result here
  expect_true(is.list(rd$betas_evaluated))
  expect_true(all(c("betas", "approx_grid_resolution") %in% names(rd$betas_evaluated)))
})

test_that("fit_parts() / marginals_of() find the right sub-object on every layout", {
  g <- fit_grid_pois()
  expect_identical(fit_parts(g), g[["grid_data"]])
  expect_null(marginals_of(g))

  a <- fit_approx_pois()
  expect_identical(fit_parts(a), a[["margin"]][["extra_info"]])
  expect_identical(marginals_of(a), a[["margin"]])
})
