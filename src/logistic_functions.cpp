#include "headers.h"
#include <cmath>
#include <limits>
// #include <chrono>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace arma;

// IRLS logistic regression solver. Canonical link
// This completely replaces fastglm for the inner simulation loop
// Does not need to estimate the dispersion parameter since a function of the
// mean

// Highly optimized, allocation-free inner solver
LogisticResult fit_logistic(const arma::mat &X, const arma::vec &y,
                            const arma::vec &initial_beta) {
  int N = X.n_rows;
  int P = X.n_cols;
  arma::vec proposed_beta = initial_beta;
  bool separated = false;

  // Pre-allocate all memory exactly once per fit
  arma::vec eta(N), p(N), w(N), grad(P);
  arma::mat XTWX(P, P);
  arma::mat XW(N, P);

  for (int i = 0; i < 15; i++) {
    eta = X * proposed_beta;
    separated = false;

    // Combine probability, separation check, and weights into one pass
    for (int k = 0; k < N; ++k) {
      double p_k = 1.0 / (1.0 + std::exp(-eta[k]));
      p[k] = p_k;

      if (p_k < 1e-8 || p_k > (1.0 - 1e-8)) {
        separated = true;
      }

      double w_k = p_k * (1.0 - p_k);
      w[k] = (w_k < 1e-6) ? 1e-6 : w_k;
    }

    if (separated)
      break;

    // Guarantee ZERO temporary matrix allocations during weight multiplication
    for (int j = 0; j < P; ++j) {
      for (int k = 0; k < N; ++k) {
        XW(k, j) = X(k, j) * w[k];
      }
    }

    XTWX = X.t() * XW;
    grad = X.t() * (y - p);

    arma::vec step;
    bool success = arma::solve(step, XTWX, grad, arma::solve_opts::fast);
    if (!success) {
      success = arma::solve(step, XTWX, grad);
    }
    if (!success)
      break;

    proposed_beta += step;

    if (arma::norm(step) < 1e-6)
      break;
  }

  return {proposed_beta, separated};
}
// Highly optimized, allocation-free inner solver
LogisticResult1D fit_logistic_1d(const arma::vec &X, const arma::vec &y,
                                 double initial_beta) {
  int N = X.n_rows;
  double proposed_beta = initial_beta;
  bool separated = false;

  // Pre-allocate vector memory
  arma::vec eta(N), p(N), w(N);

  for (int i = 0; i < 15; i++) {
    eta = X * proposed_beta; // Vector-scalar multiplication
    separated = false;

    // Combined loop for probabilities, separation check, and weights
    for (int k = 0; k < N; ++k) {
      double p_k = 1.0 / (1.0 + std::exp(-eta[k]));
      p[k] = p_k;

      if (p_k < 1e-8 || p_k > (1.0 - 1e-8)) {
        separated = true;
      }

      double w_k = p_k * (1.0 - p_k);
      w[k] = (w_k < 1e-6) ? 1e-6 : w_k;
    }

    if (separated)
      break;

    // In 1D, XTWX and grad are pure scalars.
    // We compute them in a single, cache-friendly pass.
    double XTWX = 0.0;
    double grad = 0.0;

    for (int k = 0; k < N; ++k) {
      double x_k = X[k];
      XTWX += x_k * x_k * w[k];    // Equivalent to X.t() * W * X
      grad += x_k * (y[k] - p[k]); // Equivalent to X.t() * (y - p)
    }

    // Safety check for zero-variance/empty steps
    if (std::abs(XTWX) < 1e-9) {
      break;
    }

    // Newton-Raphson update: step = grad / Hessian
    double step = grad / XTWX;
    proposed_beta += step;

    // Convergence check using absolute scalar value
    if (std::abs(step) < 1e-6)
      break;
  }

  return {proposed_beta, separated};
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
LogisticPlResult glm_logis_pl_cpp(const arma::mat &X, const arma::vec &y,
                                  const arma::vec &mle_coefs,
                                  const arma::vec &beta_vals, int m,
                                  bool approx, bool appendix) {
  // auto t_start = std::chrono::high_resolution_clock::now();

  int n = X.n_rows;

  // Compute true probabilities based on proposed betas
  arma::vec eta = X * beta_vals;
  arma::vec p = 1.0 / (1.0 + arma::exp(-eta));

  // Calculate MLE log-likelihood for the original data
  arma::vec eta_hat = X * mle_coefs;
  arma::vec p_hat = 1.0 / (1.0 + arma::exp(-eta_hat));
  bool orig_separated =
      arma::any(p_hat < 1e-8) || arma::any(p_hat > (1.0 - 1e-8));

  double mle_val = (arma::dot(y, eta_hat) - sum_softplus(eta_hat));

  // Precompute constant scalar for f.x
  double sum_log_term = sum_softplus(eta);
  double f_x = arma::dot(y, eta) - sum_log_term - mle_val;
  if (orig_separated) {
    f_x = std::numeric_limits<double>::infinity();
  }

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

  // auto t_sim_end = std::chrono::high_resolution_clock::now();

  int count_less = 0;
  double prop_sep = 0;

// Inner loop: fit models entirely in C++
#pragma omp parallel for schedule(static) reduction(                           \
        + : count_less, prop_sep) if (approx == true && appendix == false)
  for (int j = 0; j < m; ++j) {
    arma::vec y_sim = Y.col(j);

    // auto s_start = std::chrono::high_resolution_clock::now();

    // Call the inner thread-safe solver
    LogisticResult sim_res = fit_logistic(X, y_sim, beta_vals);
    if (sim_res.seperated == true) {
      prop_sep++;
    }

    arma::vec eta_sim_hat = X * sim_res.beta;

    double mle_sim = arma::dot(y_sim, eta_sim_hat) - sum_softplus(eta_sim_hat);

    // Calculate the simulated test statistic
    // llX[j] is the log-likelihood of the simulated data under the proposed
    // betas
    double f_x_sim = llX[j] - mle_sim;

    // Increment count_less if the simulated statistic is more extreme than the
    // observed
    if (f_x_sim <= f_x) {
      count_less++;
    }
  }

  prop_sep = prop_sep / m;
  double poss = 1.0 * count_less / m;
  return {poss, orig_separated, prop_sep};
}

// [[Rcpp::export]]
double logistic_ll(const arma::mat &X, const arma::vec &y,
                   const arma::vec &beta_vals) {
  LogisticResult res = fit_logistic(X, y, beta_vals);
  double mle_sim = -10000000000; // Sooooo right now if I get seperation, I
                                 // return a value of zero.
                                 // TODO. This is 100% going to lead to issues
  if (!res.seperated) {
    mle_sim = arma::dot(y, X * beta_vals) - sum_softplus(X * beta_vals);
  }
  return mle_sim;
}
// [[Rcpp::export]]
double logistic_ll_1d(const arma::vec &X, const arma::mat &y,
                      const double beta_vals) {
  LogisticResult1D res = fit_logistic_1d(X, y, beta_vals);
  double mle_sim =
      0.0; // Sooooo right now if I get seperation, I return a value of zero.
           // TODO. This is 100% going to lead to issues
  mle_sim = arma::dot(y, X * beta_vals) - sum_softplus(X * beta_vals);
  return mle_sim;
}