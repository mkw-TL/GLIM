# generate_grid() and the end-to-end grid path of glim().

generate_grid <- internal("generate_grid")

setup_grid <- function(fam = "poisson", ...) {
  d <- make_data(fam, n = 40)
  fit <- glm(y ~ x, data = d$dat, family = poisson)
  J <- crossprod(d$X, d$X * as.vector(exp(d$X %*% coef(fit))))
  eJ <- eigen(J)
  list(d = d, fit = fit, eJ = eJ)
}

test_that("generate_grid() contains the MLE and has the right shape", {
  s <- setup_grid()
  set.seed(1)
  g <- generate_grid(
    s$d$X, s$d$y, "poisson", coef(s$fit), s$eJ$vectors, s$eJ$values, 1,
    as.numeric(logLik(s$fit)), trim = FALSE, n_grid_evals = 7, m = 20
  )
  expect_equal(ncol(g$betas), 2)
  expect_equal(nrow(g$betas), 7^2)
  expect_length(g$approx_grid_resolution, 2)
  mle_row <- apply(g$betas, 1, function(r) all(abs(r - coef(s$fit)) < 1e-12))
  expect_true(any(mle_row))
})

test_that("generate_grid() has exactly n_grid_evals distinct values per coordinate (odd and even)", {
  s <- setup_grid()
  for (n_ev in c(6, 7)) {
    set.seed(1)
    g <- generate_grid(
      s$d$X, s$d$y, "poisson", coef(s$fit), s$eJ$vectors, s$eJ$values, 1,
      as.numeric(logLik(s$fit)), trim = FALSE, n_grid_evals = n_ev, m = 20
    )
    expect_equal(length(unique(g$betas[, 1])), n_ev, info = paste("n =", n_ev))
    expect_equal(length(unique(g$betas[, 2])), n_ev, info = paste("n =", n_ev))
  }
})

test_that("generate_grid() trim = TRUE only keeps points inside the ellipse", {
  s <- setup_grid()
  set.seed(1)
  full <- generate_grid(
    s$d$X, s$d$y, "poisson", coef(s$fit), s$eJ$vectors, s$eJ$values, 1,
    as.numeric(logLik(s$fit)), trim = FALSE, n_grid_evals = 9, m = 20
  )
  set.seed(1)
  trimmed <- generate_grid(
    s$d$X, s$d$y, "poisson", coef(s$fit), s$eJ$vectors, s$eJ$values, 1,
    as.numeric(logLik(s$fit)), trim = TRUE, n_grid_evals = 9, m = 20
  )
  expect_lt(nrow(trimmed$betas), nrow(full$betas))
  expect_gt(nrow(trimmed$betas), 0)
})

test_that("glim() grid result has consistent structure", {
  d <- make_data("poisson", n = 40)
  set.seed(5)
  fit <- glim(y ~ x, data = d$dat, family = "poisson", m = 50, n_grid_evals = 7, parallel = FALSE)
  expect_s3_class(fit, "glim_object")
  expect_equal(length(fit$possibilities), nrow(fit$betas_evaluated))
  expect_equal(ncol(fit$betas_evaluated), 2)
  expect_equal(colnames(fit$betas_evaluated), colnames(fit$X))
  expect_true(all(fit$possibilities >= 0 & fit$possibilities <= 1))
  for (nm in c("extended_in_neg_plaus", "extended_in_pos_plaus",
               "extended_in_neg_betas", "extended_in_pos_betas")) {
    expect_length(fit[[nm]], 2)
  }
})

test_that("glim() possibility is (near) 1 at the MLE and smaller far away", {
  d <- make_data("poisson", n = 40)
  set.seed(5)
  fit <- glim(y ~ x, data = d$dat, family = "poisson", m = 100, n_grid_evals = 7, parallel = FALSE)
  mle_row <- which(apply(fit$betas_evaluated, 1, function(r) all(abs(r - fit$mle_coefs) < 1e-10)))
  expect_length(mle_row, 1)
  expect_gt(fit$possibilities[mle_row], 0.9)
  corner <- which.max(rowSums(abs(sweep(fit$betas_evaluated, 2, fit$mle_coefs))))
  expect_lt(fit$possibilities[corner], fit$possibilities[mle_row])
})

test_that("glim() MLE agrees with glm()/lm() for every family", {
  refs <- list(
    gaussian = function(d) coef(lm(y ~ x, data = d$dat)),
    poisson = function(d) coef(glm(y ~ x, data = d$dat, family = poisson)),
    binomial = function(d) coef(glm(y ~ x, data = d$dat, family = binomial)),
    gamma = function(d) coef(glm(y ~ x, data = d$dat, family = Gamma("log"))),
    inverse.gaussian = function(d) coef(glm(y ~ x, data = d$dat, family = inverse.gaussian()))
  )
  for (fam in all_families) {
    d <- make_data(fam)
    fit <- glim(y ~ x, data = d$dat, family = fam, betas = matrix(d$beta, nrow = 1), m = 10, parallel = FALSE)
    expect_equal(unname(coef(fit)), unname(refs[[fam]](d)), tolerance = 1e-4, label = fam)
  }
})

test_that("formula interface: factors, no-intercept and explicit intercept column", {
  set.seed(1)
  d <- data.frame(y = rpois(60, 3), g = factor(rep(c("a", "b", "c"), 20)))
  fit <- glim(y ~ g, data = d, family = "poisson", betas = matrix(0, 1, 3), m = 5, parallel = FALSE)
  expect_equal(colnames(fit$X), c("(Intercept)", "gb", "gc"))

  fit0 <- glim(y ~ 0 + g, data = d, family = "poisson", betas = matrix(1, 1, 3), m = 5, parallel = FALSE)
  expect_equal(ncol(fit0$X), 3)
  expect_false("(Intercept)" %in% colnames(fit0$X))
})

test_that("results are reproducible under set.seed()", {
  d <- make_data("poisson", n = 30)
  run <- function() {
    set.seed(99)
    glim(y ~ x, data = d$dat, family = "poisson", m = 30, n_grid_evals = 5, parallel = FALSE)
  }
  a <- run()
  b <- run()
  expect_equal(a$possibilities, b$possibilities)
})

test_that("parallel and serial evaluation give the same possibilities for a fixed seed", {
  d <- make_data("poisson", n = 30)
  B <- matrix(c(0.5, 0.3, 0.4, 0.2), 2, byrow = TRUE)
  set.seed(7)
  a <- glim(y ~ x, data = d$dat, family = "poisson", betas = B, m = 40, parallel = FALSE)
  set.seed(7)
  b <- glim(y ~ x, data = d$dat, family = "poisson", betas = B, m = 40, parallel = TRUE)
  expect_equal(a$possibilities, b$possibilities)
})

test_that("coefficient-count branching: >5 predictors without betas does not fall through silently", {
  set.seed(3)
  n <- 80
  d <- as.data.frame(matrix(rnorm(n * 6), n))
  d$y <- rnorm(n)
  # Currently routes to generate_marginal_object(), which is an empty stub.
  # This test pins the intended contract: result must be a glim_object with
  # something usable in it, not NULL.
  res <- glim(y ~ ., data = d, family = "gaussian", m = 5, parallel = FALSE)
  expect_s3_class(res, "glim_object")
  expect_false(is.null(unclass(res)[[1]]))
})
