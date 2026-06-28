#include "headers.h"
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace arma;

// IRLS logistic regression solver. Canonical link
// This completely replaces fastglm for the inner simulation loop
// Does not need to estimate the dispersion parameter since a function of the
// mean

// Define struct
struct LogisticResult {
  arma::vec beta;
  bool separated;
};

// Regular solver that outputs a list
LogisticResult fit_logistic_inner(const arma::mat &X, const arma::vec &y,
                                  const arma::vec &initial_beta) {
  arma::vec proposed_beta = initial_beta;
  bool separated = false;

  for (int i = 0; i < 15; i++) {
    arma::vec eta = X * proposed_beta;
    arma::vec p = 1.0 / (1.0 + arma::exp(-eta));

    // Check for complete separation
    if (arma::any(p < 1e-8) || arma::any(p > (1.0 - 1e-8))) {
      separated = true;
    }

    arma::vec w = p % (1.0 - p);

    // Prevent strictly zero weights
    w.elem(arma::find(w < 1e-6)).fill(1e-6);

    arma::mat XTWX = X.t() * (X.each_col() % w);
    arma::vec grad = X.t() * (y - p);

    arma::vec step;
    bool success = arma::solve(step, XTWX, grad, arma::solve_opts::fast);
    if (!success) {
      success = arma::solve(step, XTWX, grad);
    }
    if (!success) {
      break;
    }

    proposed_beta += step;

    if (arma::norm(step) < 1e-6)
      break;
  }

  return {proposed_beta, separated};
}

// Rcpp wrapper for the solver
// [[Rcpp::export]]
arma::vec fit_logistic_cpp(const arma::mat &X, const arma::vec &y,
                           const arma::vec &initial_beta, bool approx) {
  LogisticResult res = fit_logistic_inner(X, y, initial_beta);
  return res.beta;
}

// Written by gemini. Uses the trick to seperate into cases
inline double sum_softplus(const arma::vec &eta) {
  double sum_val = 0.0;
  for (int i = 0; i < eta.n_elem; ++i) {
    sum_val += (eta(i) > 0) ? (eta(i) + std::log1p(std::exp(-eta(i))))
                            : std::log1p(std::exp(eta(i)));
  }
  return sum_val;
}

// The main simulation function
// [[Rcpp::export]]
double glm_logis_pl_cpp(const arma::mat &X, const arma::vec &y,
                        const arma::vec &mle_coefs, const arma::vec &beta_vals,
                        int m, bool approx) {
  int n = X.n_rows;

  // Compute true probabilities based on proposed betas
  arma::vec eta = X * beta_vals;
  arma::vec p = 1.0 / (1.0 + arma::exp(-eta));

  // Calculate MLE log-likelihood for the original data
  arma::vec eta_hat = X * mle_coefs;
  arma::vec p_hat = 1.0 / (1.0 + arma::exp(-eta_hat));
  bool orig_separated =
      arma::any(p_hat < 1e-8) || arma::any(p_hat > (1.0 - 1e-8));

  // If original data is separated, max log-likelihood is 0
  double mle_val =
      orig_separated ? 0.0 : (arma::dot(y, eta_hat) - sum_softplus(eta_hat));

  // Precompute constant scalar for f.x
  double sum_log_term = sum_softplus(eta);
  double f_x = arma::dot(y, eta) - sum_log_term - mle_val;

  // Computing new random binomial data:
  thread_local std::random_device rd;
  thread_local std::mt19937 gen(rd());
  std::uniform_real_distribution<double> runif(0.0, 1.0);
  arma::mat Y(n, m);

  for (int j = 0; j < m; ++j) {
    for (int i = 0; i < n; i++) {
      Y(i, j) = (runif(gen) < p(i)) ? 1.0 : 0.0;
    }
  }

  // Fast cross-product for all M simulations
  arma::rowvec llX = eta.t() * Y - sum_log_term;

  int count_less = 0;

// Inner loop: fit models entirely in C++
#pragma omp parallel for schedule(static)                                      \
    reduction(+ : count_less) if (approx == true)
  for (int j = 0; j < m; ++j) {
    arma::vec y_sim = Y.col(j);

    // Call the inner thread-safe solver
    LogisticResult sim_res = fit_logistic_inner(X, y_sim, beta_vals);
    arma::vec eta_sim_hat = X * sim_res.beta;

    // Force MLE to 0.0 if the simulated dataset resulted in complete separation
    double mle_sim = 0.0;
    if (!sim_res.separated) {
      mle_sim = arma::dot(y_sim, eta_sim_hat) - sum_softplus(eta_sim_hat);
    }

    double f_X_j = llX(j) - mle_sim;

    if (f_X_j <= f_x) {
      count_less++;
    }
  }

  return (double)1.0 * count_less / m;
}
