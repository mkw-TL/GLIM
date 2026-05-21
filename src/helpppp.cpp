// Written by Gemini
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// IRLS logistic regression solver. Canonical link
// This completely replaces fastglm for the inner simulation loop
arma::vec fit_logistic_cpp(const arma::mat& X, const arma::vec& y) {
  arma::vec beta = arma::zeros(X.n_cols);

  for(int i = 0; i < 15; ++i) { // 15 iterations is usually plenty for convergence
    arma::vec eta = X * beta;
    arma::vec p = 1.0 / (1.0 + arma::exp(-eta));

    // Weights for IRLS
    arma::vec w = p % (1.0 - p);

    // Prevent strictly zero weights to avoid singular matrices
    w.elem(arma::find(w < 1e-6)).fill(1e-6);

    arma::mat XTWX = X.t() * arma::diagmat(w) * X;
    arma::vec grad = X.t() * (y - p);

    arma::vec step;
    // Attempt fast solve, fallback to robust solve if nearly singular
    bool success = arma::solve(step, XTWX, grad, arma::solve_opts::fast);
    if(!success) {
      arma::solve(step, XTWX, grad);
    }

    beta += step;

    // Check for convergence
    if(arma::norm(step) < 1e-6) break;
  }
  return beta;
}

// 2. The main simulation function
// [[Rcpp::export]]
double glm_logis_pl_cpp(const arma::mat& X, const arma::vec& y,
                        const arma::vec& beta_vals, double mle_val, int m) {
  int n = X.n_rows;

  // Compute true probabilities based on proposed betas
  arma::vec eta = X * beta_vals;
  arma::vec p = 1.0 / (1.0 + arma::exp(-eta));

  // Precompute constant scalar for f.x
  double sum_log_term = arma::sum(arma::log1p(arma::exp(eta)));
  double f_x = arma::dot(y, eta) - sum_log_term - mle_val;

  // 1. Generate an n x m matrix of Uniform(0,1) numbers in one shot
  arma::mat U = arma::randu<arma::mat>(n, m);
  arma::mat Y(n, m);

  // 2. Loop over columns and apply vectorized thresholding
  for(int j = 0; j < m; ++j) {
    // U.col(j) < p creates a boolean/unsigned vector (0s and 1s)
    // arma::conv_to converts it back to doubles to fit inside Y
    Y.col(j) = arma::conv_to<arma::vec>::from( U.col(j) < p );
  }
  // Fast cross-product for all M simulations
  arma::rowvec llX = eta.t() * Y - sum_log_term;

  int count_less = 0;

  // Inner loop: fit models entirely in C++
  for(int j = 0; j < m; ++j) {
    arma::vec y_sim = Y.col(j);

    arma::vec coefs = fit_logistic_cpp(X, y_sim);
    arma::vec eta_hat = X * coefs;

    double mle_sim = arma::dot(y_sim, eta_hat) - arma::sum(arma::log1p(arma::exp(eta_hat)));
    double f_X_j = llX(j) - mle_sim;

    if(f_X_j < f_x) {
      count_less++;
    }
  }

  return (double)count_less / m;
}

// Gamma regression (log-link)
//
//
//

// 1. IRLS Gamma regression solver (Log link), fisher weights so just OLS as W = 1.
arma::vec fit_gamma_log_cpp(const arma::mat& X, const arma::vec& y) {
  arma::vec beta = arma::zeros(X.n_cols);

  // Precompute XTWX because W is an identity matrix (weights = 1)
  arma::mat XTX = X.t() * X;

  for(int i = 0; i < 20; ++i) {
    arma::vec eta = X * beta;
    arma::vec mu = arma::exp(eta);

    // Gradient remains the same: X^T * (y/mu - 1)
    arma::vec grad = X.t() * (y / mu - 1.0);

    arma::vec step;
    bool success = arma::solve(step, XTX, grad, arma::solve_opts::fast);
    if(!success) {
      if(!arma::solve(step, XTX, grad)) {
        break;
      }
    }

    beta += step;
    if(arma::norm(step) < 1e-6) break;
  }
  return beta;
}

// Helper function to compute full Gamma log-likelihood matching R's logLik()
double compute_gamma_ll(const arma::vec& y, const arma::vec& eta, double shape, int n) {
  double term1 = n * (shape * std::log(shape) - std::lgamma(shape));
  double term2 = (shape - 1.0) * arma::sum(arma::log(y));
  double term3 = -shape * (arma::dot(y, arma::exp(-eta)) + arma::sum(eta));
  return term1 + term2 + term3;
}

// 2. The main simulation function
// [[Rcpp::export]]
double glm_gamma_pl_cpp(const arma::mat& X, const arma::vec& y,
                        const arma::vec& beta_vals, double mle_val,
                        double shape, int m) {
  int n = X.n_rows;

  // Compute true expected values based on proposed betas
  arma::vec eta = X * beta_vals;
  arma::vec mu = arma::exp(eta);

  // Compute full log-likelihood for the observed data under proposed beta
  double true_ll = compute_gamma_ll(y, eta, shape, n);
  double f_x = true_ll - mle_val;

  // Simulate Y matrix inline using R's Gamma RNG
  // R::rgamma uses (shape, scale) parameterization -> scale = mu / shape
  arma::mat Y(n, m);
  arma::vec scale = mu / shape;
  for(int j = 0; j < m; ++j) {
    for(int i = 0; i < n; ++i) {
      Y(i, j) = R::rgamma(shape, scale(i));
    }
  }

  // Precompute constant pieces of log-likelihood across all M simulations
  double constant_ll_term = n * (shape * std::log(shape) - std::lgamma(shape));

  // Vectorized cross-product step for part of the log-likelihood evaluation
  arma::rowvec term3_all = -shape * (arma::exp(-eta).t() * Y + arma::sum(eta));

  int count_less = 0;

  // Inner loop: fit models entirely in C++
  for(int j = 0; j < m; ++j) {
    arma::vec y_sim = Y.col(j);

    // Calculate the simulated dependent term: (shape - 1) * sum(log(y_sim))
    double term2_j = (shape - 1.0) * arma::sum(arma::log(y_sim));
    double llX_j = constant_ll_term + term2_j + term3_all(j);

    // Fit the simulated data
    arma::vec coefs = fit_gamma_log_cpp(X, y_sim);
    arma::vec eta_hat = X * coefs;

    // Evaluate simulated MLE log-likelihood
    double mle_sim = compute_gamma_ll(y_sim, eta_hat, shape, n);
    double f_X_j = llX_j - mle_sim;

    if(f_X_j < f_x) {
      count_less++;
    }
  }

  return (double)count_less / m;
}


// =====================================================================
// GAUSSIAN REGRESSION (IDENTITY LINK)
// =====================================================================

// Identity link Gaussian regression is just Ordinary Least Squares (OLS)
// No IRLS loop is required.
arma::vec fit_gaussian_cpp(const arma::mat& X, const arma::vec& y) {
  arma::vec beta;
  bool success = arma::solve(beta, X, y, arma::solve_opts::fast);
  if(!success) {
    arma::solve(beta, X, y);
  }
  return beta;
}

// Helper function to compute Gaussian log-likelihood
double compute_gaussian_ll(const arma::vec& y, const arma::vec& mu, double sigma, int n) {
  double ll = -(n / 2.0) * std::log(2.0 * M_PI * sigma * sigma)
  - arma::sum(arma::pow(y - mu, 2)) / (2.0 * sigma * sigma);
  return ll;
}

// Main simulation function for Gaussian
// [[Rcpp::export]]
double glm_gaussian_pl_cpp(const arma::mat& X, const arma::vec& y,
                           const arma::vec& beta_vals, double mle_val,
                           double sigma, int m) {
  int n = X.n_rows;

  arma::vec mu = X * beta_vals; // Identity link

  double true_ll = compute_gaussian_ll(y, mu, sigma, n);
  double f_x = true_ll - mle_val;

  // 1. Batch generate an n x m matrix of Standard Normals N(0,1)
  arma::mat Y = arma::randn<arma::mat>(n, m);

  // 2. Scale by standard deviation
  Y *= sigma;

  // 3. Add the mean vector (mu) to every column simultaneously
  Y.each_col() += mu;

  int count_less = 0;

  for(int j = 0; j < m; ++j) {
    arma::vec y_sim = Y.col(j);

    arma::vec coefs = fit_gaussian_cpp(X, y_sim);
    arma::vec mu_hat = X * coefs;

    double mle_sim = compute_gaussian_ll(y_sim, mu_hat, sigma, n);
    double llX_j = compute_gaussian_ll(y_sim, mu, sigma, n);

    double f_X_j = llX_j - mle_sim;

    if(f_X_j < f_x) {
      count_less++;
    }
  }

  return (double)count_less / m;
}

// =====================================================================
// POISSON REGRESSION (LOG LINK)
// =====================================================================

// IRLS Poisson regression solver (Log link)
arma::vec fit_poisson_log_cpp(const arma::mat& X, const arma::vec& y) {
  arma::vec beta = arma::zeros(X.n_cols);

  for(int i = 0; i < 20; ++i) {
    arma::vec eta = X * beta;

    // Cap eta to prevent overflow when calculating expected values
    eta.elem(arma::find(eta > 20)).fill(20);
    arma::vec mu = arma::exp(eta);

    // Weights for Poisson log-link
    arma::vec w = mu;
    w.elem(arma::find(w < 1e-8)).fill(1e-8);

    arma::mat XTWX = X.t() * arma::diagmat(w) * X;
    arma::vec grad = X.t() * (y - mu);

    arma::vec step;
    bool success = arma::solve(step, XTWX, grad, arma::solve_opts::fast);
    if(!success) {
      if(!arma::solve(step, XTWX, grad)) break;
    }

    beta += step;
    if(arma::norm(step) < 1e-6) break;
  }
  return beta;
}

double glm_poisson_ll(arma::vec& eta, arma::vec& mu, const arma::vec& y) {

  double term1 = arma::dot(y, eta);

  // Sum of mu
  double term2 = arma::accu(mu);

  // log(y!) is equivalent to lgamma(y + 1)
  double term3 = arma::accu(arma::lgamma(y + 1.0));

  return term1 - term2 - term3;
}

// Main simulation function for Poisson
// [[Rcpp::export]]
double glm_poisson_pl_cpp(const arma::mat& X, const arma::vec& y,
                          const arma::vec& beta_vals, double mle_val, int m) {
  int n = X.n_rows;

  arma::vec eta = X * beta_vals;
  arma::vec mu = arma::exp(eta);

  double true_ll = glm_poisson_ll(eta, mu, y);

  double f_x = true_ll - mle_val;

  arma::mat Y(n, m);
  for(int j = 0; j < m; ++j) {
    for(int i = 0; i < n; ++i) {
      Y(i, j) = R::rpois(mu(i));
    }
  }

  int count_less = 0;

  for(int j = 0; j < m; ++j) {
    arma::vec y_sim = Y.col(j);

    arma::vec coefs = fit_poisson_log_cpp(X, y_sim);
    arma::vec eta_hat = X * coefs;
    arma::vec mu_hat = arma::exp(eta_hat);

    double mle_sim = 0.0;
    double llX_j = 0.0;
    for(int i = 0; i < n; ++i) {
      mle_sim += R::dpois(y_sim(i), mu_hat(i), 1);
      llX_j += R::dpois(y_sim(i), mu(i), 1);
    }

    double f_X_j = llX_j - mle_sim;

    if(f_X_j < f_x) {
      count_less++;
    }
  }

  return (double)count_less / m;
}

// =====================================================================
// INVERSE GAUSSIAN REGRESSION (1/MU^2 LINK)
// =====================================================================

// R does not have a native rinvgauss. Implemented is the Michael,
// Schucany, and Haas (1976) algorithm to simulate it via C++.
double rinvgauss_single(double mu, double lambda) {
  double v = R::rnorm(0, 1);
  double y_sq = v * v;
  double x = mu + (mu * mu * y_sq) / (2.0 * lambda) -
    (mu / (2.0 * lambda)) * std::sqrt(4.0 * mu * lambda * y_sq + mu * mu * y_sq * y_sq);

  double u = R::runif(0, 1);
  if (u <= mu / (mu + x)) {
    return x;
  } else {
    return (mu * mu) / x;
  }
}

// IRLS Inverse Gaussian regression solver (1/mu^2 link)
arma::vec fit_invgauss_cpp(const arma::mat& X, const arma::vec& y) {
  arma::vec beta = arma::zeros(X.n_cols);

  // Initialization: For 1/mu^2 link, eta = X*beta must be strictly > 0
  double mean_y = arma::mean(y);
  beta(0) = 1.0 / (mean_y * mean_y);

  for(int i = 0; i < 30; ++i) {
    arma::vec eta = X * beta;
    // Enforce strict positivity constraint for the link function
    eta.elem(arma::find(eta < 1e-6)).fill(1e-6);

    arma::vec mu = arma::pow(eta, -0.5);

    // For Inverse Gaussian 1/mu^2 link:
    // W = diag(mu^3) as opposed to diag(mu^3 / 4) for canonical link
    // Modified gradient step RHS = -2 * X^T (y - mu)
    arma::vec w = arma::pow(mu, 3.0);

    // Clamp extreme weights to preserve numerical stability
    w.elem(arma::find(w < 1e-8)).fill(1e-8);
    w.elem(arma::find(w > 1e8)).fill(1e8);
    // -2 is pulled out front to make computations nicer
    arma::mat XTWX = X.t() * arma::diagmat(w) * X;
    arma::vec grad = -2.0 * X.t() * (y - mu);

    arma::vec step;
    bool success = arma::solve(step, XTWX, grad, arma::solve_opts::fast);
    if(!success) {
      if(!arma::solve(step, XTWX, grad)) break;
    }

    // Line search (step halving) to ensure eta remains > 0
    arma::vec new_beta = beta + step;
    arma::vec new_eta = X * new_beta;
    int iter_halve = 0;
    while (arma::any(new_eta <= 0.0) && iter_halve < 10) {
      step /= 2.0;
      new_beta = beta + step;
      new_eta = X * new_beta;
      iter_halve++;
    }

    beta = new_beta;
    if(arma::norm(step) < 1e-6) break;
  }
  return beta;
}

// Helper function to compute Inverse Gaussian log-likelihood
double compute_invgauss_ll(const arma::vec& y, const arma::vec& mu, double lambda, int n) {
  double term1 = 0.5 * n * std::log(lambda / (2.0 * M_PI));
  double term2 = -1.5 * arma::sum(arma::log(y));
  double term3 = -lambda * arma::sum(arma::pow(y - mu, 2) / (2.0 * arma::pow(mu, 2) % y));
  return term1 + term2 + term3;
}

// Main simulation function for Inverse Gaussian
// [[Rcpp::export]]
double glm_invgauss_pl_cpp(const arma::mat& X, const arma::vec& y,
                           const arma::vec& beta_vals, double mle_val,
                           double lambda, int m) {
  int n = X.n_rows;

  arma::vec eta = X * beta_vals;
  arma::vec mu = arma::pow(eta, -0.5);

  double true_ll = compute_invgauss_ll(y, mu, lambda, n);
  double f_x = true_ll - mle_val;

  arma::mat Y(n, m);
  for(int j = 0; j < m; ++j) {
    for(int i = 0; i < n; ++i) {
      Y(i, j) = rinvgauss_single(mu(i), lambda);
    }
  }

  int count_less = 0;

  for(int j = 0; j < m; ++j) {
    arma::vec y_sim = Y.col(j);

    arma::vec coefs = fit_invgauss_cpp(X, y_sim);
    arma::vec eta_hat = X * coefs;

    // Validate eta_hat to compute simulated MLE likelihood
    eta_hat.elem(arma::find(eta_hat < 1e-6)).fill(1e-6);
    arma::vec mu_hat = arma::pow(eta_hat, -0.5);

    double mle_sim = compute_invgauss_ll(y_sim, mu_hat, lambda, n);
    double llX_j = compute_invgauss_ll(y_sim, mu, lambda, n);

    double f_X_j = llX_j - mle_sim;

    if(f_X_j < f_x) {
      count_less++;
    }
  }

  return (double)count_less / m;
}


