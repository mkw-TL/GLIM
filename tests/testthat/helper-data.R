# Shared fixtures for the GLIM test suite.
# Everything here is deterministic (fixed seeds) and small, so tests stay fast.

# Internal functions are reached with GLIM::: so tests don't depend on exports.
internal <- function(name) getFromNamespace(name, "GLIM")

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
