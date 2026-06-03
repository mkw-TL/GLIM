# Implementation of GLMs in Ryan Martin's Inferential Model framework.
# Author: Joe Harrison (jrharr25@ncsu.edu)

# Logistic approximation case is broken
# The gamma case works though!

library(GLIM)
# Data setup

X <- matrix(
  c(.0794, .1, .1259, .1413, .15, .1558, .1778, .1995, .2239, .2512, .2818, .3162),
  ncol = 1
)
successes <- c(1, 2, 1, 0, 1, 2, 4, 6, 4, 5, 5, 8)
failures <- 10 - successes
y <- cbind(successes, failures) # from ?glm
# Need to implement this binomial functionality? ^

# rep(vector, other_vector) will repeat each element of the vector the corresponding other_vector element no of times
X_binary <- rep(X, times = successes)
X_binary <- rbind(as.matrix(X_binary), as.matrix(rep(X, times = failures)))
X_binary <- cbind(X_binary, rep(1, sum(successes + failures)))
y_binary <- c(rep(1, times = sum(successes)), rep(0, times = sum(failures)))

# Need to have a way to automatically create a grid around the mle
beta_0 <- seq(from = -5, to = -1, by = .1)
beta_1 <- seq(0, 25, by = .2)
betas <- expand.grid(beta_1, beta_0)
betas <- as.matrix(betas)
fit <- glm(y_binary ~ X_binary - 1, family = "binomial")
ll_mle_original_data <- logLik(fit)
mle_coefs <- fit$coefficients

output <- glim(
  X_binary,
  y_binary,
  family = "binomial",
  betas,
  dispersions = 1,
  m = 100,
  parallel = TRUE,
  approx = FALSE
)

z_matrix <- matrix(output, nrow = length(beta_1), ncol = length(beta_0))
contour(x = beta_1, y = beta_0, z = z_matrix)

# URL <- 'http://static.lib.virginia.edu/statlab/materials/data/alb_homes.csv'
# homes <- read.csv(file = URL)
# y_gamma <- homes$totalvalue
# X_gamma <- cbind(rep(1, length(y_gamma)), homes$finsqft)
# New_X_gamma <- X_gamma
# New_X_gamma[, 2] <- scale(X_gamma[, 2])
# fit <- glm(y_gamma ~ New_X_gamma - 1, family = Gamma(link = "log"))
# mle_coefs <- fit$coefficients
# mle_dispersion <- .1225424 #somewhere around here
# # mle_dispersion <- mle_estimate_dispersion_gamma(
# #   y_gamma,
# #   exp(New_X_gamma %*% mle_coefs),
# #   length(mle_coefs)
# # )
# y <- y_gamma
# eta <- New_X_gamma %*% mle_coefs
# mle_val <- compute_gamma_ll_r(y, eta, 1 / mle_dispersion)

# beta_0_grid <- seq(12.70, 12.90, by = .01)
# beta_1_grid <- seq(.40, .53, by = .01)
# beta_grid <- expand.grid(beta_0_grid, beta_1_grid)
# beta_grid <- as.matrix(beta_grid)

# # Maximum that I could have gotten with my profiling resolution is .9
# # Confirmed that at MLE for the dispersion param, get 1.

# dispersions <- mle_dispersion
# output <- glim(New_X_gamma, y_gamma, "gamma", beta_grid, mle_dispersion, m = 100, parallel = TRUE)
# profiled_mat_glim <- matrix(output, nrow = length(beta_0_grid), ncol = length(beta_1_grid))
# contour(
#   beta_0_grid,
#   beta_1_grid,
#   profiled_mat_glim,
#   main = "Gamma GLM possibility",
#   xlab = "beta_0",
#   ylab = "beta_1"
# )
