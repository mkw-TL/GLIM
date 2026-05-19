// Written by Gemini
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// IRLS logistic regression solver
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

  // Simulate Y matrix inline using R's RNG
  arma::mat Y(n, m);
  for(int j = 0; j < m; ++j) {
    for(int i = 0; i < n; ++i) {
      Y(i, j) = R::rbinom(1, p(i));
    }
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

// Now I want to write a function that does Gamma regression.



