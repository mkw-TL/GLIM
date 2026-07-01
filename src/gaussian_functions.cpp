#include "headers.h"
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace arma;

// Identity link Gaussian regression is just Ordinary Least Squares (OLS)
// No IRLS loop is required.
// [[Rcpp::export]]
arma::vec fit_gaussian_cpp(const arma::mat &X, const arma::vec &y) {
  arma::vec proposed_beta;
  bool success = arma::solve(proposed_beta, X, y, arma::solve_opts::fast);
  if (!success) {
    success = arma::solve(proposed_beta, X, y);
  }
  if (!success) {
    proposed_beta.zeros(X.n_cols);
  }
  return proposed_beta;
}

// Helper function to compute Gaussian log-likelihood
// [[Rcpp::export]]
double compute_gaussian_ll(const arma::vec &y, const arma::vec &mu,
                           double sigma) {
  double ll = -(y.n_elem / 2.0) * std::log(2.0 * M_PI * sigma * sigma) -
              arma::accu((y - mu) % (y - mu)) / (2 * sigma * sigma);
  return ll;
}

// [[Rcpp::export]]
arma::vec compute_gaussian_ll_mat(const arma::vec &y, const arma::mat &mu,
                                  double sigma) {
  arma::vec ll(mu.n_cols);
  for (int i = 0; i < mu.n_cols; i++) {

    ll(i) = compute_gaussian_ll(y, mu.col(i), sigma);
  }
  return ll;
}

// [[Rcpp::export]]
double est_dispersion(const arma::vec &y, const arma::vec &mu, int &p) {
  return arma::accu((y - mu) % (y - mu)) / (y.n_elem - p);
}

// Main simulation function for Gaussian
// [[Rcpp::export]]
double glm_gaussian_pl_cpp(const arma::mat &X, const arma::vec &y,
                           const arma::vec &mle_coefs,
                           const arma::vec &beta_vals, int m, bool approx) {
  int n = X.n_rows;

  arma::vec mu = X * beta_vals; // Identity link
  arma::vec mu_hat = X * mle_coefs;
  // Estimated dispersion
  int p = beta_vals.n_elem;

  double sigma = std::sqrt(est_dispersion(y, mu_hat, p));

  double true_ll = compute_gaussian_ll(y, mu, sigma);
  double mle_val = compute_gaussian_ll(y, mu_hat, sigma);
  double f_x = true_ll - mle_val;

  thread_local std::random_device rd;
  thread_local std::mt19937 gen(rd());
  std::normal_distribution<double> rnorm(0.0, 1.0);

  arma::mat Y(n, m);
  for (int j = 0; j < m; ++j) {
    for (int i = 0; i < n; i++) {
      Y(i, j) = mu(i) + sigma * rnorm(gen);
    }
  }

  int count_less = 0;

  for (int j = 0; j < m; ++j) {
    arma::vec y_sim = Y.col(j);

    arma::vec sim_coefs = fit_gaussian_cpp(X, y_sim);
    arma::vec mu_hat_sim = X * sim_coefs;
    double mle_sim = compute_gaussian_ll(y_sim, mu_hat, sigma);
    double llX_j = compute_gaussian_ll(y_sim, mu, sigma);

    double f_X_j = llX_j - mle_sim;

    if (f_X_j <= f_x) {
      count_less++;
    }
  }

  return (double)1.0 * count_less / m;
}
