# Implementation of GLMs in Ryan Martin's Inferential Model framework.
# Author: Joe Harrison (jrharr25@ncsu.edu)

library(GLIM)

X <- matrix(
  c(.0794, .1, .1259, .1413, .15, .1558, .1778, .1995, .2239, .2512, .2818, .3162),
  ncol = 1
)
successes <- c(1, 2, 1, 0, 1, 2, 4, 6, 4, 5, 5, 8)
failures <- 10 - successes
y <- cbind(successes, failures) # from ?glm

# Need to have a way to automatically create a grid around the mle if doing non-approx
# Related to the eigenvalues? Imvar for one iteration? Think through this...
beta_0 <- seq(from = -8, to = 1, by = .1)
beta_1 <- seq(5, 20, by = .2)
betas <- expand.grid(beta_0, beta_1)
betas <- as.matrix(betas)

Rprof()
start_time <- Sys.time()
output <- glim(X, y, family = "binomial", betas, m = 100, parallel = TRUE, approx = FALSE)
end_time <- Sys.time()
end_time - start_time
z_matrix <- matrix(output, nrow = length(beta_0), ncol = length(beta_1))
contour(
  x = beta_0,
  y = beta_1,
  z = z_matrix,
  xlab = "beta_0",
  ylab = "beta_1",
  main = "Logistic GLIM"
)
Rprof(NULL)
summaryRprof()

# Output most practicioners care about
get_CI(.95, betas, output)

output <- glim(X, y, family = "binomial", betas, m = 100, parallel = TRUE, approx = TRUE)

# Automatically should do this? (yes)
beta1_grid <- seq(min(output[1, ]), max(output[1, ]), length.out = 30)
beta2_grid <- seq(min(output[2, ]), max(output[2, ]), length.out = 30)
beta_p2p_grid <- expand.grid(beta1_grid, beta2_grid)
beta_p2p_grid <- t(as.matrix(beta_p2p_grid))
possibils <- prob2poss_logis(X_binary, y_binary, output, beta_p2p_grid)
# TODO marginalization
z_matrix <- matrix(possibils, nrow = length(beta1_grid), ncol = length(beta2_grid))
contour(
  x = beta1_grid,
  y = beta2_grid,
  z = z_matrix,
  xlab = "beta_0",
  ylab = "beta_1",
  main = "Logistic GLIM"
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

# Only if grid is needed
beta_0_grid <- seq(11.8, 12.2, by = .05)
beta_1_grid <- seq(.48, .53, by = .025)
beta_grid <- expand.grid(beta_0_grid, beta_1_grid)
beta_grid <- as.matrix(beta_grid)

start_time <- Sys.time()
Rprof()
output <- glim(X_gamma, y_gamma, "gamma", beta_grid, m = 100, approx = FALSE, parallel = TRUE)
Rprof(NULL)
end_time <- Sys.time()
end_time - start_time

summaryRprof()


profiled_mat_glim <- matrix(output, nrow = length(beta_0_grid), ncol = length(beta_1_grid))
contour(
  beta_0_grid,
  beta_1_grid,
  profiled_mat_glim,
  main = "Gamma GLIM possibility",
  xlab = "beta_0",
  ylab = "beta_1"
)

# gamma approx:
start_time <- Sys.time()
Rprof()
output <- glim(X_gamma, y_gamma, "gamma", beta_grid, m = 100, approx = TRUE, parallel = TRUE)
Rprof(NULL)
end_time <- Sys.time()
end_time - start_time

summaryRprof()

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
  beta1_grid,
  beta2_grid,
  profiled_mat_glim,
  main = "Gamma GLIM possibility",
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

library(GLIM)
library(dplyr)
library(Lahman)
data(BattingPost)
liveball_ws <- BattingPost |> filter(yearID >= 1920) |> filter(round == "WS")

y_poisson <- liveball_ws$R
X_poisson <- liveball_ws$H
X_poisson <- cbind(rep(1, length(X_poisson)), X_poisson)
X_poisson <- as.matrix(X_poisson)

# Only if grid is needed
beta_0_grid <- seq(-1.15, -1, by = .002)
beta_1_grid <- seq(.28, .31, by = .002)
beta_grid <- expand.grid(beta_0_grid, beta_1_grid)
beta_grid <- as.matrix(beta_grid)

start_time <- Sys.time()
Rprof()
output <- glim(X_poisson, y_poisson, "poisson", beta_grid, m = 100, approx = FALSE, parallel = TRUE)
Rprof(NULL)
end_time <- Sys.time()
end_time - start_time

summaryRprof()

profiled_mat_glim <- matrix(output, nrow = length(beta_0_grid), ncol = length(beta_1_grid))
contour(
  beta_0_grid,
  beta_1_grid,
  profiled_mat_glim,
  main = "Poisson GLIM possibility",
  xlab = "beta_0",
  ylab = "beta_1"
)


# gamma approx:
start_time <- Sys.time()
Rprof()
output <- glim(X_poisson, y_poisson, "poisson", beta_grid, m = 100, approx = TRUE, parallel = TRUE)
Rprof(NULL)
end_time <- Sys.time()
end_time - start_time

summaryRprof()

beta1_grid <- seq(min(output[1, ]), max(output[1, ]), length.out = 30)
beta2_grid <- seq(min(output[2, ]), max(output[2, ]), length.out = 30)
beta_p2p_grid <- expand.grid(beta1_grid, beta2_grid)
beta_p2p_grid <- t(as.matrix(beta_p2p_grid))


possibils <- prob2poss_poisson(
  X = X_poisson,
  y = y_poisson,
  samples = output,
  the_compared_theta = beta_p2p_grid
)

profiled_mat_glim <- matrix(possibils, nrow = length(beta1_grid), ncol = length(beta2_grid))

contour(
  beta1_grid,
  beta2_grid,
  profiled_mat_glim,
  main = "Poisson GLIM possibility",
  xlab = "beta_0",
  ylab = "beta_1"
)


### Gaussian case:
library(GLIM)
X <- matrix(c(1, 1, 1, .3, .5, .73), ncol = 2)
y <- c(4, 5, 7)
GLIM::glim(X, y, "gaussian", c(1.1, 6), 100, FALSE, FALSE)
