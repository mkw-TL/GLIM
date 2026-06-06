# Implementation of GLMs in Ryan Martin's Inferential Model framework.
# Author: Joe Harrison (jrharr25@ncsu.edu)

# IMVAR approximation is broken. Gamma and Logistic fast, though!

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

# Need to have a way to automatically create a grid around the mle if doing non-approx
# beta_0 <- seq(from = -8, to = 1, by = .1)
# beta_1 <- seq(5, 20, by = .2)
# betas <- expand.grid(beta_1, beta_0)
# betas <- as.matrix(betas)

mle_coefs <- glm(y_binary ~ X_binary - 1, family = "binomial")
mle_coefs <- as.matrix(mle_coefs)
mle_coefs <- c(14.606939, -3.560164)

start_time <- Sys.time()
output <- glim(
  X_binary,
  y_binary,
  family = "binomial",
  betas,
  m = 100,
  parallel = TRUE,
  approx = TRUE
)

end_time <- Sys.time()
end_time - start_time

beta1_grid <- seq(min(output[1, ]), max(output[1, ]), length.out = 30)
beta2_grid <- seq(min(output[2, ]), max(output[2, ]), length.out = 30)
beta_p2p_grid <- expand.grid(beta1_grid, beta2_grid)
beta_p2p_grid <- t(as.matrix(beta_p2p_grid))

possibils <- prob2poss_logis(X_binary, y_binary, output, beta_p2p_grid)

z_matrix <- matrix(possibils, nrow = length(beta1_grid), ncol = length(beta2_grid))
contour(
  x = beta1_grid,
  y = beta2_grid,
  z = z_matrix,
  xlab = "beta_0",
  ylab = "beta_1",
  main = "Logistic GLM"
)


##### Gamma case:
#
#
#
#
#

library(GLIM)

URL <- 'http://static.lib.virginia.edu/statlab/materials/data/alb_homes.csv'
homes <- read.csv(file = URL)
y_gamma <- homes$totalvalue[1:100]
X_gamma <- cbind(rep(1, length(y_gamma)), homes$finsqft[1:100])
New_X_gamma <- X_gamma
New_X_gamma[, 2] <- scale(X_gamma[, 2]) # standardize the x-predictor


start_time <- Sys.time()
Rprof()
output <- glim(
  New_X_gamma,
  y_gamma,
  "gamma",
  beta_p2p_grid,
  m = 100,
  approx = TRUE,
  parallel = TRUE
)
Rprof(NULL)
end_time <- Sys.time()
end_time - start_time

summaryRprof()


# Only if grid is needed
beta_0_grid <- seq(12.70, 12.90, by = .01)
beta_1_grid <- seq(.40, .53, by = .01)
beta_grid <- expand.grid(beta_0_grid, beta_1_grid)
beta_grid <- as.matrix(beta_grid)

# Maximum that I could have gotten with my profiling resolution is .9
# Confirmed that at MLE for the dispersion param, get 1.

# debugging
# X <- New_X_gamma
# y <- y_gamma
# res <- glm(y_gamma ~ New_X_gamma - 1, family = Gamma("log"))
# mle_coefs <- res$coefficients
# mle_val <- logLik(res)

# J <- crossprod(X, X)
# J <- eigen(J)

# parallel <- FALSE
# tol <- .01
# a <- 1
# b <- .65
# max.it <- 25
# m <- 100
# family <- "gamma"

# # Rcpp::sourceCpp("src/possibilistic_computations.cpp")

# alpha <- .40
# xi <- c(1, .36787)
# mle <- mle_coefs
# pl <- function(z) {
#   betas_matrix <- if (is.matrix(z)) z else matrix(z, nrow = 1)
#   glim_raw(X, y, family, betas_matrix, mle_coefs, mle_val = mle_val, m, parallel)
# }
# pl(c(12.82, .48))
# a <- 1
# b <- .65
# d <- 2

# pl(mle + c(1, 1))
# pl(mle - c(1, 1))

# pl(mle - c(-.01723379, 0.0000001))
# # imvar(c(1, 1), alpha, pl, mle_coefs, J, .01, 1, 1, 25)

beta1_grid <- seq(min(output[1, ]), max(output[1, ]), length.out = 30)
beta2_grid <- seq(min(output[2, ]), max(output[2, ]), length.out = 30)
beta_p2p_grid <- expand.grid(beta1_grid, beta2_grid)
beta_p2p_grid <- t(as.matrix(beta_p2p_grid))

possibils <- prob2poss_gamma(
  X = X_gamma,
  y = y_gamma,
  samples = output,
  the_compared_theta = beta_p2p_grid
)

profiled_mat_glim <- matrix(possibils, nrow = length(beta1_grid), ncol = length(beta2_grid))
contour(
  beta_0_grid,
  beta_1_grid,
  profiled_mat_glim,
  main = "Gamma GLM possibility",
  xlab = "beta_0",
  ylab = "beta_1"
)


y <- y_gamma
X <- X_gamma
