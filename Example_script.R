# Implementation of GLMs in Ryan Martin's Inferential Model framework.
# Author: Joe Harrison (jrharr25@ncsu.edu)

# IMVAR approximation is broken. Gamma and Logistic fast, though!

library(GLIM)
library(progress)
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
beta_0 <- seq(from = -8, to = 1, by = .1)
beta_1 <- seq(5, 20, by = .2)
betas <- expand.grid(beta_1, beta_0)
betas <- as.matrix(betas)

mle_coefs <- coef(glm(y_binary ~ X_binary - 1, family = "binomial"))
mle_coefs <- as.matrix(mle_coefs)


Rprof()

start_time <- Sys.time()
output <- glim(
  X_binary,
  y_binary,
  family = "binomial",
  betas,
  m = 100,
  parallel = TRUE,
  approx = FALSE
)

end_time <- Sys.time()
end_time - start_time
z_matrix <- matrix(output, nrow = length(beta_1), ncol = length(beta_0))
contour(
  x = beta_1,
  y = beta_0,
  z = z_matrix,
  xlab = "beta_0",
  ylab = "beta_1",
  main = "Logistic GLM"
)

Rprof(NULL)
summaryRprof()


output <- glim(
  X_binary,
  y_binary,
  family = "binomial",
  betas,
  m = 100,
  parallel = TRUE,
  approx = TRUE
)
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


# Rcpp::sourceCpp("src/possibilistic_computations.cpp")
# coef(glm(y_gamma ~ New_X_gamma - 1, family = Gamma("log")))
# fit_gamma_log_cpp(New_X_gamma, t(New_X_gamma) %*% New_X_gamma, y_gamma, c(1, 1))

# y_gamma <- c(10, 40, 30, 60, 50)
# X_gamma <- matrix(c(1, 1, 1, 1, 1, 30, 60, 30, 45, 50), nrow = 5)
# res <- glm(y_gamma ~ X_gamma - 1, family = Gamma("log"))
# mle_coefs <- coef(res)
# disp <- pearson_estimate_dispersion_gamma(y_gamma, exp(X_gamma %*% mle_coefs), length(mle_coefs))

# fit_gamma_log_cpp(X_gamma, t(X_gamma) %*% X_gamma, y_gamma, mle_coefs)
# New_X_gamma <- X_gamma

# Only if grid is needed
dbeta_0_grid <- seq(11.8, 12.2, by = .05)
beta_1_grid <- seq(.48, .53, by = .025)
beta_grid <- expand.grid(beta_0_grid, beta_1_grid)
beta_grid <- as.matrix(beta_grid)
# beta_grid <- matrix(c(12.09, .51), nrow = 1)

res <- glm(y_gamma ~ New_X_gamma - 1, family = Gamma("log"))
coef(res)

start_time <- Sys.time()
Rprof()
output <- glim(New_X_gamma, y_gamma, "gamma", beta_grid, m = 100, approx = FALSE, parallel = TRUE)
Rprof(NULL)
end_time <- Sys.time()
end_time - start_time

summaryRprof()


profiled_mat_glim <- matrix(output, nrow = length(beta_0_grid), ncol = length(beta_1_grid))
contour(
  beta_0_grid,
  beta_1_grid,
  profiled_mat_glim,
  main = "Gamma GLM possibility",
  xlab = "beta_0",
  ylab = "beta_1"
)

# gamma approx:
start_time <- Sys.time()
Rprof()
output <- glim(New_X_gamma, y_gamma, "gamma", beta_grid, m = 100, approx = TRUE, parallel = TRUE)
Rprof(NULL)
end_time <- Sys.time()
end_time - start_time

summaryRprof()

beta1_grid <- seq(min(output[1, ]), max(output[1, ]), length.out = 30)
beta2_grid <- seq(min(output[2, ]), max(output[2, ]), length.out = 30)


beta_p2p_grid <- expand.grid(beta1_grid, beta2_grid)
beta_p2p_grid <- t(as.matrix(beta_p2p_grid))

possibils <- prob2poss_gamma(
  X = New_X_gamma,
  y = y_gamma,
  samples = output,
  the_compared_theta = beta_p2p_grid
)

profiled_mat_glim <- matrix(possibils, nrow = length(beta1_grid), ncol = length(beta2_grid))
contour(
  beta1_grid,
  beta2_grid,
  profiled_mat_glim,
  main = "Gamma GLM possibility",
  xlab = "beta_0",
  ylab = "beta_1"
)

## Poisson case:
#
#
#
#
#
#

beta_grid[c(40, 100), ]

Rcpp::sourceCpp("src/possibilistic_computations.cpp")


pois_pos(X_poisson, y_poisson, as.matrix(c(-1.1, .25), nrow = 1), beta_grid[100, ], 100, FALSE)

fit_glm_omp_r(
  X_poisson,
  y_poisson,
  as.matrix(c(-1.1, .25), nrow = 1),
  beta_grid[1:2, ],
  "poisson",
  FALSE,
  1,
  100,
  FALSE
)


library(GLIM)
library(dplyr)
library(Lahman)
data(BattingPost)
liveball_ws <- BattingPost |> filter(yearID >= 1920) |> filter(round == "WS")

y_poisson <- liveball_ws$R
X_poisson <- liveball_ws$H
X_poisson <- cbind(rep(1, length(X_poisson)), X_poisson)
X_poisson <- as.matrix(X_poisson)

#mle coefs are -1.075, .295

# Only if grid is needed
beta_0_grid <- seq(-1.2, -1, by = .007)
beta_1_grid <- seq(.27, .33, by = .01)
beta_grid <- expand.grid(beta_0_grid, beta_1_grid)
beta_grid <- as.matrix(beta_grid)

start_time <- Sys.time()
Rprof()
output <- glim(
  X_poisson,
  y_poisson,
  "poisson",
  beta_grid,
  m = 100,
  approx = FALSE,
  parallel = FALSE
)
Rprof(NULL)
end_time <- Sys.time()
end_time - start_time

summaryRprof()

profiled_mat_glim <- matrix(output, nrow = length(beta_0_grid), ncol = length(beta_1_grid))
contour(
  beta_0_grid,
  beta_1_grid,
  profiled_mat_glim,
  main = "Poisson GLM possibility",
  xlab = "beta_0",
  ylab = "beta_1"
)


res <- glm(y_poisson ~ X_poisson - 1, family = "poisson")
mle_coefs <- coef(res)
eta <- X_poisson %*% mle_coefs
mu <- exp(eta)

y_poisson %*% eta - sum(mu) - sum(log(gamma(y_poisson + 1)))

Rcpp::sourceCpp("src/possibilistic_computations.cpp")
fit_poisson_log_cpp(as.matrix(X_poisson), y_poisson, as.matrix(c(1, 1), nrow = 2))


y_poisson %*% eta - sum(mu) - sum(log(gamma(y_poisson + 1)))


# gamma approx:
start_time <- Sys.time()
Rprof()
output <- glim(X_poisson, y_poisson, "poisson", beta_grid, m = 100, approx = TRUE, parallel = TRUE)
Rprof(NULL)
end_time <- Sys.time()
end_time - start_time

summaryRprof()


possibils <- prob2poss_poisson(
  X = X_poisson,
  y = y_poisson,
  samples = output,
  the_compared_theta = beta_p2p_grid
)

output <- matrix(possibils, nrow = length(beta1_grid), ncol = length(beta2_grid))

beta1_grid <- seq(min(output[1, ]), max(output[1, ]), length.out = 30)
beta2_grid <- seq(min(output[2, ]), max(output[2, ]), length.out = 30)
beta_p2p_grid <- expand.grid(beta1_grid, beta2_grid)
beta_p2p_grid <- t(as.matrix(beta_p2p_grid))

plot(output[1, ], output[2, ])


contour(
  beta1_grid,
  beta2_grid,
  output,
  main = "Poisson GLM possibility",
  xlab = "beta_0",
  ylab = "beta_1"
)
