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
arma::vec fit_poisson_log_cpp(const arma::mat &X, const arma::vec &y) {
  // Don't need to worry about an initial value, since this comes from the data
  // Prevent log(0) by capping minimum values to 0.1
  arma::vec safe_y = y;
  safe_y.elem(arma::find(safe_y < 0.1)).fill(0.1);
  arma::vec eta_init = arma::log(safe_y);

  arma::vec proposed_beta;
  // Get rough starting point. Fallback to zeros if X is ill-conditioned.
  if (!arma::solve(proposed_beta, X, eta_init)) {
    proposed_beta = arma::zeros<arma::vec>(X.n_cols);
  }

  arma::vec eta = X * proposed_beta;
  arma::vec mu = arma::exp(arma::clamp(eta, -50.0, 50.0));
  double current_dev = calculate_deviance_poisson(y, mu);
  arma::vec step(X.n_cols);

  for (int i = 0; i < 25; i++) {
    arma::vec w = mu;
    w.elem(arma::find(w < 1e-8)).fill(1e-8); // Stabilize tiny weights

    arma::mat XtWX = X.t() * arma::diagmat(w) * X;
    arma::vec grad = X.t() * (y - mu);

    bool success = arma::solve(step, XtWX, grad, arma::solve_opts::fast);
    if (!success) {
      success = arma::solve(step, XtWX, grad);
      if (!success) {
        break; // Matrix is singular, accept current beta
      }
    }

    if (!step.is_finite())
      break;

    // cap the norm(step-size) to be 20. Since the mean and the variance of the
    // poisson are linked, we are preventing drastic oversteps
    double max_step = 3.0;
    double step_norm = arma::norm(step);
    if (step_norm > max_step) {
      step = step * (max_step / step_norm);
    }

    arma::vec temp_beta = proposed_beta + step;
    arma::vec temp_eta = X * temp_beta;
    arma::vec proposed_mu = arma::exp(arma::clamp(temp_eta, -50.0, 50.0));
    double proposed_dev = calculate_deviance_poisson(y, proposed_mu);

    // step-half
    int half_iter = 0;
    while ((!std::isfinite(proposed_dev) || proposed_dev > current_dev) &&
           half_iter < 10) {
      step = step / 2.0;
      temp_beta = proposed_beta + step;
      temp_eta = X * temp_beta;
      proposed_mu = arma::exp(arma::clamp(temp_eta, -50.0, 50.0));
      proposed_dev = calculate_deviance_poisson(y, proposed_mu);
      half_iter++;
    }

    // Update state
    proposed_beta = temp_beta;
    mu = proposed_mu;
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
  int n = X.n_rows;

  arma::vec eta = X * beta_vals;
  eta = arma::clamp(eta, -50.0, 50.0);
  arma::vec mu = arma::exp(eta);

  // If the value of mu is too large, then the parameter is not possible
  if (!mu.is_finite() || mu.max() > 1e7) {
    return 0.0;
  }

  arma::vec eta_hat = X * mle_coefs;
  eta_hat = arma::clamp(eta_hat, -50.0, 50.0);
  arma::vec mu_hat = arma::exp(eta_hat);
  mu_hat = arma::clamp(mu_hat, std::numeric_limits<double>::epsilon(), 1e15);

  double true_ll = glm_poisson_ll(eta, mu, y);
  double mle_ll = glm_poisson_ll(eta_hat, mu_hat, y);
  double f_x = true_ll - mle_ll;

  arma::mat Y_sim(n, m);

  // Constructor
  thread_local std::random_device rd;
  // rd() is a non-deterministic random number
  thread_local std::mt19937 gen(rd());
  // Is using the random import to get access to these distributions
  for (int i = 0; i < n; i++) {
    std::poisson_distribution<int> rpois(mu(i));
    for (int j = 0; j < m; j++) {
      Y_sim(i, j) = rpois(gen);
    }
  }

  int count_less = 0;

  for (int j = 0; j < m; j++) {
    arma::vec y_sim = Y_sim.col(j);

    arma::vec coefs_sim = fit_poisson_log_cpp(X, y_sim);
    if (!coefs_sim.is_finite()) {
      Rcpp::Rcout << "[j=" << j << "] coefs_sim is non-finite!\n";
      Rcpp::Rcout << coefs_sim.t() << "\n";
    }
    arma::vec eta_hat_sim = X * coefs_sim;
    eta_hat_sim = arma::clamp(eta_hat_sim, -50.0, 50.0);
    arma::vec mu_hat_sim = arma::exp(eta_hat_sim);

    double mle_sim = 0.0;
    double llX_j = 0.0;
    // Not using the dedicated log-likelihood function, as can remove the
    // factorial this way Compare log likelihoods. Gamma y+1 is y!. Can remove
    // the factorial computation since is subtracted away
    for (int i = 0; i < n; i++) {
      mle_sim += y_sim(i) * eta_hat_sim(i) - mu_hat_sim(i);
      llX_j += y_sim(i) * eta(i) - mu(i);
    }

    double f_X_j = llX_j - mle_sim;

    if (f_X_j <= f_x) {
      count_less++;
    }
  }

  return (double)count_less / m;
}
