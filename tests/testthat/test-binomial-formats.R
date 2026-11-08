# Binomial responses can be 0/1 vectors, 2-column (success, failure) matrices or factors.
# glim() expands them to one row per Bernoulli trial. With custom betas the
# result is the flat grid object, read here through fit_parts().

mk <- function() {
  set.seed(11)
  n <- 30
  x <- rnorm(n)
  trials <- 3
  succ <- rbinom(n, trials, plogis(0.2 + 0.8 * x))
  list(x = x, succ = succ, fail = trials - succ, n = n)
}

test_that("2-column matrix response expands to the same MLE as glm()", {
  d <- mk()
  ym <- cbind(d$succ, d$fail)
  fit <- glim(
    ym ~ d$x,
    family = "binomial",
    betas = matrix(c(0.2, 0.8), nrow = 1),
    m = 10,
    parallel = FALSE
  )
  ref <- glm(ym ~ d$x, family = binomial)
  parts <- fit_parts(fit)
  expect_equal(unname(parts$mle_coefs), unname(coef(ref)), tolerance = 1e-6)
  expect_equal(length(parts$y), sum(ym)) # one row per Bernoulli trial
  expect_equal(nrow(parts$X), sum(ym))
})

test_that("factor response treats the first level as failure", {
  set.seed(2)
  x <- rnorm(40)
  y <- factor(rbinom(40, 1, plogis(0.5 * x)), labels = c("no", "yes"))
  fit <- glim(
    y ~ x,
    family = "binomial",
    betas = matrix(c(0, 0.5), nrow = 1),
    m = 10,
    parallel = FALSE
  )
  ref <- glm(y ~ x, family = binomial)
  expect_equal(unname(fit_parts(fit)$mle_coefs), unname(coef(ref)), tolerance = 1e-6)
})

test_that("factor response with an expanded design keeps X and y aligned", {
  set.seed(3)
  x <- rnorm(25)
  y <- factor(rbinom(25, 1, 0.5), labels = c("a", "b"))
  fit <- glim(
    y ~ x,
    family = "binomial",
    betas = matrix(c(0, 0), nrow = 1),
    m = 5,
    parallel = FALSE
  )
  parts <- fit_parts(fit)
  expect_equal(nrow(parts$X), 25)
  expect_equal(length(parts$y), 25)
  expect_true(all(parts$y %in% c(0, 1)))
})

test_that("0/1 response is used unchanged", {
  d <- make_data("binomial")
  fit <- glim(
    y ~ x,
    data = d$dat,
    family = "binomial",
    betas = matrix(c(0.5, 0.3), nrow = 1),
    m = 10,
    parallel = FALSE
  )
  expect_equal(unname(fit_parts(fit)$y), d$y)
})

test_that("binomial default grid path also returns the grid_data layout", {
  skip_on_cran()
  d <- make_data("binomial", n = 60)
  set.seed(4)
  fit <- glim(y ~ x, data = d$dat, family = "binomial", m = 20, n_grid_evals = 5, parallel = FALSE)
  expect_named(fit, "grid_data")
  expect_equal(fit[["grid_data"]]$family, "binomial")
})
