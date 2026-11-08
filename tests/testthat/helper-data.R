# Shared fixtures for the GLIM test suite.
# Everything here is deterministic (fixed seeds) and small, so tests stay fast.

# Internal functions are reached through the namespace so tests don't depend on exports.
internal <- function(name) getFromNamespace(name, "GLIM")

# ---------------------------------------------------------------------------
# Output-structure helpers
#
# glim() now returns different list layouts depending on the path taken:
#
#   default grid (p <= 5, no betas)        list(grid_data = <grid>)
#   grid + directly_calc_marginals = TRUE  list(marginals = <marginals>, grid_data = <grid>)
#   custom betas                           <grid>  (flat, no wrapper)
#   custom betas + direct marginals        list(glim_object = <grid>, marginals = <marginals>)
#   p > 5 and no betas                     <marginals>  (list(margins, extra_info))
#   approx = TRUE                          list(margin = <marginals>, elliptical_data = ...)
#   radial = TRUE                          list(margin = <marginals>, radial_data = ...)
#
# where <grid> has possibilities, betas_evaluated, family, X, y, mle_coefs,
# logLik, vcov, dispersion (+ extended_in_* for the default grid path), and
# <marginals> has `margins` (one list per coefficient) and `extra_info`.
#
# These helpers always use exact `[[` matching so that `$` partial matching
# (e.g. "margin" -> "margins") can't hide a structural change.
# ---------------------------------------------------------------------------

# The "bag of fit information" (X, y, mle_coefs, family, ...) wherever it lives.
fit_parts <- function(fit) {
  if (!is.null(fit[["grid_data"]])) {
    fit[["grid_data"]]
  } else if (!is.null(fit[["glim_object"]])) {
    fit[["glim_object"]]
  } else if (!is.null(fit[["margin"]])) {
    fit[["margin"]][["extra_info"]]
  } else if (!is.null(fit[["margins"]])) {
    fit[["extra_info"]]
  } else {
    fit
  }
}

# The marginals object (list(margins, extra_info)) if the fit has one, else NULL.
marginals_of <- function(fit) {
  if (!is.null(fit[["margin"]])) {
    fit[["margin"]]
  } else if (!is.null(fit[["marginals"]])) {
    fit[["marginals"]]
  } else if (!is.null(fit[["margins"]])) {
    fit
  } else {
    NULL
  }
}

# Tiny memoiser so expensive fits are shared between tests in a file.
.fit_cache <- new.env()
cached <- function(key, expr) {
  if (is.null(.fit_cache[[key]])) {
    .fit_cache[[key]] <- expr
  }
  .fit_cache[[key]]
}

# Standard poisson fits used by several files.
fit_grid_pois <- function() {
  cached("grid_pois", {
    d <- make_data("poisson", n = 40)
    set.seed(5)
    glim(y ~ x, data = d$dat, family = "poisson", m = 50, n_grid_evals = 7, parallel = FALSE)
  })
}

fit_approx_pois <- function() {
  cached("approx_pois", {
    d <- make_data("poisson", n = 40)
    set.seed(5)
    glim(
      y ~ x,
      data = d$dat,
      family = "poisson",
      m = 20,
      n_grid_evals = 5,
      approx = TRUE,
      parallel = FALSE
    )
  })
}

# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

make_data <- function(family, n = 60, seed = 1) {
  set.seed(seed)
  x <- rnorm(n)
  X <- cbind(`(Intercept)` = 1, x = x)
  beta <- c(0.5, 0.3)
  eta <- as.vector(X %*% beta)
  y <- switch(
    family,
    gaussian = eta + rnorm(n),
    poisson = rpois(n, exp(eta)),
    binomial = rbinom(n, 1, 1 / (1 + exp(-eta))),
    gamma = rgamma(n, shape = 4, rate = 4 / exp(eta)),
    inverse.gaussian = {
      # link is 1/mu^2, so keep eta strictly positive
      eta_ig <- 2 + 0.2 * x
      mu <- eta_ig^(-1 / 2)
      # simple positive draw; exact IG sampling is not needed for these tests
      rgamma(n, shape = 8, rate = 8 / mu)
    },
    stop("unknown family")
  )
  beta_eval <- if (family == "inverse.gaussian") c(2, 0.2) else beta
  list(X = X, y = y, x = x, dat = data.frame(y = y, x = x), beta = beta_eval, n = n)
}

generate_mock_data <- function(n = 100, p = 3, family = "gaussian") {
  set.seed(42)
  X <- cbind(1, matrix(rnorm(n * (p - 1)), nrow = n)) # Explicit intercept

  # Small true_beta to prevent explosion with exponential link functions
  true_beta <- c(0.5, -0.2, 0.1)
  eta <- X %*% true_beta

  if (family == "gaussian") {
    y <- eta + rnorm(n, sd = 0.5)
  } else if (family == "binomial") {
    y <- rbinom(n, 1, 1 / (1 + exp(-eta)))
  } else if (family == "gamma") {
    mu <- exp(eta)
    y <- rgamma(n, shape = 2, rate = 2 / mu)
  } else if (family == "poisson") {
    y <- rpois(n, lambda = exp(eta))
  } else if (family == "inverse.gaussian") {
    mu <- exp(eta)
    if (requireNamespace("statmod", quietly = TRUE)) {
      y <- statmod::rinvgauss(n, mean = mu, shape = 1)
    } else {
      y <- rgamma(n, shape = 2, rate = 2 / mu) # fallback, only dimensions matter
    }
  }

  list(X = X, y = as.numeric(y), true_beta = true_beta)
}

make_gaussian_data <- function(n = 60, p = 2) {
  set.seed(42)
  X <- cbind(intercept = 1, matrix(rnorm(n * (p - 1)), n, p - 1))
  colnames(X) <- c("intercept", paste0("x", seq_len(p - 1)))
  beta <- seq(0.5, 1.5, length.out = p)
  y <- as.vector(X %*% beta + rnorm(n, sd = 0.5))
  list(X = X, y = y, beta = beta)
}

make_poisson_data <- function(n = 80, p = 2) {
  set.seed(42)
  X <- cbind(intercept = 1, matrix(rnorm(n * (p - 1), sd = 0.3), n, p - 1))
  colnames(X) <- c("intercept", paste0("x", seq_len(p - 1)))
  beta <- seq(0.1, 0.3, length.out = p)
  y <- rpois(n, exp(X %*% beta))
  list(X = X, y = y, beta = beta)
}

make_gamma_data <- function(n = 80, p = 2) {
  set.seed(42)
  X <- cbind(intercept = 1, matrix(rnorm(n * (p - 1), sd = 0.3), n, p - 1))
  colnames(X) <- c("intercept", paste0("x", seq_len(p - 1)))
  beta <- seq(0.1, 0.3, length.out = p)
  mu <- exp(X %*% beta)
  y <- rgamma(n, shape = 5, rate = 5 / mu)
  list(X = X, y = y, beta = beta)
}

make_invgauss_data <- function(n = 80, p = 2) {
  set.seed(42)
  X <- cbind(intercept = 1, matrix(rnorm(n * (p - 1), sd = 0.2), n, p - 1))
  colnames(X) <- c("intercept", paste0("x", seq_len(p - 1)))
  beta <- seq(1, 1.5, length.out = p)
  eta <- X %*% beta
  mu <- as.vector(sqrt(1 / eta))
  y <- pmax(mu + rnorm(n, sd = 0.05), 0.05)
  list(X = X, y = y, beta = beta)
}

make_binomial_data <- function(n = 100, p = 2, well_separated = FALSE) {
  set.seed(42)
  X <- cbind(intercept = 1, matrix(rnorm(n * (p - 1)), n, p - 1))
  colnames(X) <- c("intercept", paste0("x", seq_len(p - 1)))
  beta <- if (well_separated) c(0, rep(50, p - 1)) else seq(-0.5, 0.5, length.out = p)
  prob <- 1 / (1 + exp(-X %*% beta))
  y <- rbinom(n, 1, prob)
  list(X = X, y = y, beta = beta)
}

# Log-likelihood written independently from R's own densities, used as an oracle
# for gradient checks. Dispersion phi is held fixed.
oracle_ll <- function(family, X, y, beta, dispersion = 1) {
  eta <- as.vector(X %*% beta)
  switch(
    family,
    gaussian = sum(dnorm(y, eta, sqrt(dispersion), log = TRUE)),
    poisson = sum(dpois(y, exp(eta), log = TRUE)),
    binomial = sum(dbinom(y, 1, 1 / (1 + exp(-eta)), log = TRUE)),
    gamma = sum(dgamma(y, shape = 1 / dispersion, scale = exp(eta) * dispersion, log = TRUE)),
    inverse.gaussian = {
      mu <- eta^(-1 / 2)
      lambda <- 1 / dispersion
      sum(0.5 * log(lambda / (2 * pi * y^3)) - lambda * (y - mu)^2 / (2 * mu^2 * y))
    }
  )
}

# Central finite-difference gradient (no numDeriv dependency)
fd_grad <- function(f, beta, h = 1e-6) {
  vapply(
    seq_along(beta),
    function(j) {
      e <- replace(numeric(length(beta)), j, h)
      (f(beta + e) - f(beta - e)) / (2 * h)
    },
    numeric(1)
  )
}

all_families <- c("gaussian", "poisson", "binomial", "gamma", "inverse.gaussian")
