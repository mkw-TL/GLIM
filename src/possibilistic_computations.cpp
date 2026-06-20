// Translated to c++ by Gemini
// Thoroughly vetted

#include "headers.h"
#define ARMA_BOUNDS_CHECK
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace arma;

// =====================================================================
// POISSON REGRESSION (LOG LINK)
// =====================================================================

double calculate_deviance_poisson(const arma::vec &y,
                                  const arma::vec &proposed_mu) {
  arma::vec safe_mu = arma::clamp(proposed_mu, 1e-10, double(arma::datum::inf));

  arma::vec log_term = arma::zeros<arma::vec>(y.n_elem);

  // 3. Only calculate log for elements where y > 0. y*log(y) where y goes to
  // zero is taken as zero
  arma::uvec positive_y_indices = arma::find(y > 0);

  if (!positive_y_indices.is_empty()) {
    // Use element-wise multiplication (%) and division (/) for the safe indices
    log_term.elem(positive_y_indices) =
        y.elem(positive_y_indices) %
        arma::log(y.elem(positive_y_indices) /
                  safe_mu.elem(positive_y_indices));
  }
  arma::vec dev_elements = log_term - (y - safe_mu);
  return 2.0 * arma::accu(dev_elements);
}

// IRLS Poisson regression solver (Log link)
// [[Rcpp::export]]
arma::vec fit_poisson_log_cpp(const arma::mat &X, const arma::vec &y,
                              const arma::vec &mle_coefs) {
  arma::vec proposed_beta = mle_coefs;
  arma::vec initial_mu = arma::exp(arma::clamp(X * proposed_beta, -30.0, 30.0));

  double current_dev = calculate_deviance_poisson(y, initial_mu);
  arma::vec step(X.n_cols);

  for (int i = 0; i < 20; i++) {
    arma::vec eta = X * proposed_beta;

    arma::vec mu = arma::exp(arma::clamp(eta, -30.0, 30.0));
    mu = arma::clamp(mu, std::numeric_limits<double>::epsilon(), 1e15);

    // Weights for Poisson log-link
    arma::vec w = mu;
    w.elem(arma::find(w < 1e-8)).fill(1e-8);

    arma::mat XtWX = X.t() * arma::diagmat(w) * X;
    arma::vec grad = X.t() * (y - mu);

    bool success = arma::solve(step, XtWX, grad, arma::solve_opts::fast);
    if (!success) {
      success = arma::solve(step, XtWX, grad);
      if (!success) {
        Rcpp::Rcout << "[fit_poisson_log_cpp] iter " << i
                    << ": solve failed, breaking\n";
        break;
      }
    }
    if (!step.is_finite()) {
      Rcpp::Rcout << "[fit_poisson_log_cpp] iter " << i
                  << ": step has non-finite values\n";
      break;
    }

    // Propose a new beta
    arma::vec temp_beta = proposed_beta + step;
    arma::vec proposed_mu = arma::exp(arma::clamp(X * temp_beta, -30.0, 30.0));
    proposed_mu =
        arma::clamp(proposed_mu, std::numeric_limits<double>::epsilon(), 1e15);
    double proposed_dev = calculate_deviance_poisson(y, proposed_mu);
    // Step-Halving Loop
    int half_iter = 0;
    while ((!std::isfinite(proposed_dev) || proposed_dev > current_dev) &&
           half_iter < 25) {
      step = step / 2.0;                // Cut the step in half
      temp_beta = proposed_beta + step; // Try again
      proposed_mu = arma::exp(arma::clamp(X * temp_beta, -30.0, 30.0));
      proposed_mu = arma::clamp(proposed_mu,
                                std::numeric_limits<double>::epsilon(), 1e15);
      proposed_dev = calculate_deviance_poisson(y, proposed_mu);
      half_iter++;
    }
    proposed_beta = temp_beta;
    current_dev = proposed_dev;

    if (arma::norm(step) < 1e-6) {
      break;
    }
  }
  return proposed_beta;
}

// [[Rcpp::export]]
double glm_poisson_ll(arma::vec &eta, arma::vec &mu, const arma::vec &y) {

  double term1 = arma::dot(y, eta);

  // Sum of mu
  double term2 = arma::accu(mu);

  // log(y!) is equivalent to lgamma(y + 1)
  double term3 = arma::accu(arma::lgamma(y + 1.0));

  return term1 - term2 - term3;
}

// Main simulation function for Poisson
// [[Rcpp::export]]
double glm_poisson_pl_cpp(const arma::mat &X, const arma::vec &y,
                          const arma::vec &mle_coefs,
                          const arma::vec &beta_vals, int m, bool approx) {

  // Will get me into issues when parallel is true....
  // Need to check first.
  if (y.n_elem != X.n_rows || mle_coefs.n_elem != X.n_cols ||
      beta_vals.n_elem != X.n_cols) {
    Rcpp::stop("Dimension mismatch: X is %d x %d, y has %d, mle_coefs has %d, "
               "beta_vals has %d",
               X.n_rows, X.n_cols, y.n_elem, mle_coefs.n_elem,
               beta_vals.n_elem);
  }
  int n = X.n_rows;

  arma::vec eta = X * beta_vals;
  eta = arma::clamp(eta, -30.0, 30.0);
  arma::vec mu = arma::exp(eta);
  mu = arma::clamp(mu, std::numeric_limits<double>::epsilon(), 1e15);

  double true_ll = glm_poisson_ll(eta, mu, y);
  arma::vec eta_hat = X * mle_coefs;
  eta_hat = arma::clamp(eta_hat, -30.0, 30.0);
  arma::vec mu_hat = arma::exp(eta_hat);
  mu_hat = arma::clamp(mu_hat, std::numeric_limits<double>::epsilon(), 1e15);
  double mle_ll = glm_poisson_ll(eta_hat, mu_hat, y);

  double f_x = true_ll - mle_ll;

  if (!mu.is_finite()) {
    Rcpp::Rcout << "mu contains NaN/Inf!\n";
    arma::uvec bad_idx = arma::find_nonfinite(mu);
    if (!bad_idx.is_empty()) {
      Rcpp::Rcout << "Found " << bad_idx.n_elem
                  << " non-finite mu values at indices: " << bad_idx.t()
                  << "\n";
      Rcpp::Rcout << "Offending mu values: " << mu.elem(bad_idx).t() << "\n";
    }
  }
  if (mu.max() > 1e7) {
    Rcpp::Rcout << "[WARNING] mu max is " << mu.max()
                << " — poisson_distribution may misbehave\n";
  }

  arma::mat Y(n, m);

  // Constructor
  thread_local std::random_device rd;
  // rd() is a non-deterministic random number
  thread_local std::mt19937 gen(rd());
  // Is using the random import to get access to these distributions
  for (int j = 0; j < m; ++j) {
    for (int i = 0; i < n; i++) {
      if (!std::isfinite(mu(i)) || mu(i) > 1e7) {
        Rcpp::Rcout << "[j=" << j << ", i=" << i << "] bad mu: " << mu(i)
                    << "\n";
        Rcpp::stop("mu out of safe range for poisson_distribution");
      }
      std::poisson_distribution<int> rpois(mu(i));
      Y(i, j) = rpois(gen);
    }
  }

  int count_less = 0;

  for (int j = 0; j < m; ++j) {
    arma::vec y_sim = Y.col(j);

    arma::vec coefs_sim = fit_poisson_log_cpp(X, y_sim, beta_vals);
    if (!coefs_sim.is_finite()) {
      Rcpp::Rcout << "[j=" << j << "] coefs_sim is non-finite!\n";
      Rcpp::Rcout << coefs_sim.t() << "\n";
    }
    arma::vec eta_hat_sim = X * coefs_sim;
    eta_hat_sim = arma::clamp(eta_hat_sim, -30.0, 30.0);
    arma::vec mu_hat_sim = arma::exp(eta_hat_sim);
    mu_hat_sim =
        arma::clamp(mu_hat_sim, std::numeric_limits<double>::epsilon(), 1e15);

    double mle_sim = 0.0;
    double llX_j = 0.0;
    // Not using the dedicated log-likelihood function, as can remove the
    // factorial this way Compare log likelihoods. Gamma y+1 is y!. Can remove
    // the factorial computation since is subtracted away
    for (int i = 0; i < n; i++) {
      mle_sim += y_sim(i) * eta_hat_sim(i) - mu_hat_sim(i);
      llX_j += y_sim(i) * eta(i) - mu(i);
    }

    // Rcpp::Rcout << "coefs_sim - mle_coefs: " << coefs_sim - mle_coefs <<
    // "\n";
    double f_X_j = llX_j - mle_sim;
    // Rcpp::Rcout << "f_X_j: " << f_X_j << "\n";

    if (f_X_j < f_x) {
      count_less++;
    }
  }

  return (double)count_less / m;
}
