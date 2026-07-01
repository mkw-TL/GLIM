#include "headers.h"
#include <chrono>
#include <random>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace arma;

// Define struct to track solver results cleanly
struct PoissonResult {
  arma::vec beta;
  bool success;
};

// Highly optimized, allocation-free deviance calculation
inline double calculate_deviance_poisson(const arma::vec &y,
                                         const arma::vec &mu) {
  double dev = 0.0;
  int n = y.n_elem;

  for (int i = 0; i < n; i++) {
    double y_val = y[i];
    double mu_val = mu[i];
    if (mu_val < 1e-10) {
      mu_val = 1e-10; // Safe clamp for mu
    }

    double log_term = 0.0;
    if (y_val > 0.0) {
      log_term = y_val * std::log(y_val / mu_val);
    }
    dev += log_term - (y_val - mu_val);
  }
  return 2.0 * dev;
}

// Allocation-free inner Poisson solver
PoissonResult fit_poisson_inner(const arma::mat &X, const arma::vec &y,
                                const arma::vec &initial_beta) {
  int N = X.n_rows;
  int P = X.n_cols;
  arma::vec proposed_beta = initial_beta;

  // Pre-allocated tracking vectors
  arma::vec eta(N), mu(N), grad(P), step(P);
  arma::mat XTWX(P, P);
  arma::mat XW(N, P);

  bool solver_success = true;

  // Fixed 10 iterations max. Well-conditioned systems converge in 4-6 steps.
  for (int i = 0; i < 10; i++) {
    eta = X * proposed_beta;

    // Aggressive, tight clamping prevents the matrix from becoming singular
    for (int k = 0; k < N; ++k) {
      double e = eta[k];
      if (e < -10.0)
        e = -10.0; // Floor prevents weight from dropping below 0.000045
      if (e > 10.0)
        e = 10.0; // Ceiling prevents exponential explosion
      mu[k] = std::exp(e);
    }

    // High-speed column scaling (matrix multiplication bypass)
    XW = X;
    XW.each_col() %= mu; // mu acts directly as our diagonal weight vector

    XTWX = X.t() * XW;
    grad = X.t() * (y - mu);

    // With tight clamping, the fast-path solver succeeds reliably
    solver_success = arma::solve(step, XTWX, grad, arma::solve_opts::fast);
    if (!solver_success) {
      solver_success = arma::solve(step, XTWX, grad); // Rare safety fallback
    }

    if (!solver_success || !step.is_finite()) {
      break;
    }

    // Unconditional step update (relies on eta clamping for absolute stability)
    proposed_beta += step;

    // Rapid convergence exit
    if (arma::norm(step) < 1e-5) {
      break;
    }
  }

  return {proposed_beta, solver_success};
}

double compute_poisson_ll(const arma::vec &eta, const arma::vec &y) {
  return arma::dot(y, eta) - arma::accu(exp(eta)) -
         arma::accu(arma::lgamma(y + 1.0));
}

// [[Rcpp::export]]
arma::vec compute_poisson_ll_mat(const arma::mat &eta, const arma::vec y) {
  arma::vec ll(eta.n_cols);
  for (int i = 0; i < eta.n_cols; i++) {
    ll(i) = compute_poisson_ll(eta.col(i), y);
  }
  return ll;
}

// Exported wrapper for diagnostic verification
// [[Rcpp::export]]
arma::vec fit_poisson_log_cpp(const arma::mat &X, const arma::vec &y,
                              const arma::vec &initial_beta) {
  PoissonResult res = fit_poisson_inner(X, y, initial_beta);
  return res.beta;
}

// Main Parallelized Simulation Function for Poisson
// [[Rcpp::export]]
double glm_poisson_pl_cpp(const arma::mat &X, const arma::vec &y,
                          const arma::vec &mle_coefs,
                          const arma::vec &beta_vals, int m, bool approx) {
  auto t_start = std::chrono::high_resolution_clock::now();
  int n = X.n_rows;

  arma::vec eta = X * beta_vals;
  eta = arma::clamp(eta, -10.0, 10.0);
  arma::vec mu = arma::exp(eta);

  if (!mu.is_finite() || mu.max() > 1e5) {
    return 0.0;
  }

  arma::vec eta_hat = X * mle_coefs;
  eta_hat = arma::clamp(eta_hat, -10.0, 10.0);

  double true_ll = compute_poisson_ll(eta, y);
  double mle_ll = compute_poisson_ll(eta_hat, y);
  double f_x = true_ll - mle_ll;

  // ---------------------------------------------------------
  // PHASE 1: LOOP-INVERTED PARALLEL DATA GENERATION
  // ---------------------------------------------------------
  arma::mat Y_sim(n, m); // Pre-allocate full simulation matrix

#pragma omp parallel if (approx)
  {
    std::random_device rd;
    std::mt19937 gen(rd());

    // By iterating over 'n' on the outside, the distribution stays hot in the
    // cache
#pragma omp for schedule(static)
    for (int i = 0; i < n; i++) {
      std::poisson_distribution<int> rpois(mu(i));
      for (int j = 0; j < m; j++) {
        Y_sim(i, j) = rpois(gen);
      }
    }
  }
  auto t_sim_end = std::chrono::high_resolution_clock::now();

  // ---------------------------------------------------------
  // PHASE 2: PARALLEL SOLVER
  // ---------------------------------------------------------
  int count_less = 0;
  double total_solver_time = 0.0;
  double total_likelihood_time = 0.0;

#pragma omp parallel reduction(+ : count_less, total_solver_time,              \
                                   total_likelihood_time) if (approx)
  {
    // Thread-local workspaces (Zero heap allocations inside the j-loop)
    arma::vec y_sim_local(n);
    arma::vec eta_hat_sim(n);

#pragma omp for schedule(static)
    for (int j = 0; j < m; j++) {

      // Pull the pre-generated column from cache directly
      y_sim_local = Y_sim.col(j);

      // Start the solver directly using beta_vals
      auto s_start = std::chrono::high_resolution_clock::now();
      PoissonResult sim_res = fit_poisson_inner(X, y_sim_local, beta_vals);
      auto s_end = std::chrono::high_resolution_clock::now();
      total_solver_time +=
          std::chrono::duration<double>(s_end - s_start).count();

      auto l_start = std::chrono::high_resolution_clock::now();
      eta_hat_sim = X * sim_res.beta;

      double mle_sim = 0.0;
      double llX_j = 0.0;

      // Fast inline calculation of log-likelihood differences without
      // allocations
      for (int i = 0; i < n; i++) {
        double ehs = eta_hat_sim(i);
        if (ehs < -10.0)
          ehs = -10.0;
        if (ehs > 10.0)
          ehs = 10.0;

        mle_sim += y_sim_local(i) * ehs - std::exp(ehs);
        llX_j += y_sim_local(i) * eta(i) - mu(i);
      }
      auto l_end = std::chrono::high_resolution_clock::now();
      total_likelihood_time +=
          std::chrono::duration<double>(l_end - l_start).count();

      double f_X_j = llX_j - mle_sim;

      if (f_X_j <= f_x) {
        count_less++;
      }
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();

  // Print results
  // Rcpp::Rcout << "\n--- POISSON PROFILE ---" << "\n";
  // Rcpp::Rcout << "Total Time:      "
  // << std::chrono::duration<double>(t_end - t_start).count()
  // << "s\n";
  // Rcpp::Rcout << "Data Gen Time:   "
  //             << std::chrono::duration<double>(t_sim_end - t_start).count()
  //             << "s\n";
  // Rcpp::Rcout << "Sum Solver Time: " << total_solver_time
  //             << "s (Combined across threads)\n";
  // Rcpp::Rcout << "Sum LL Time:     " << total_likelihood_time
  //             << "s (Combined across threads)\n";

  return (double)count_less / m;
}