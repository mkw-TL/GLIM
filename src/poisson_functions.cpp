#include "headers.h"
#include <atomic>
// #include <chrono>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace arma;

// Since not doing step-halving, don't need this
// inline double calculate_deviance_poisson(const arma::vec &y,
//                                          const arma::vec &mu) {
//   double dev = 0.0;
//   int n = y.n_elem;

//   for (int i = 0; i < n; i++) {
//     double y_val = y[i];
//     double mu_val = mu[i];
//     if (mu_val < 1e-10) {
//       mu_val = 1e-10; // Safe clamp for mu
//     }

//     double log_term = 0.0;
//     if (y_val > 0.0) {
//       log_term = y_val * std::log(y_val / mu_val);
//     }
//     dev += log_term - (y_val - mu_val);
//   }
//   return 2.0 * dev;
// }

PoissonResult fit_poisson_inner(const arma::mat &X, const arma::vec &y,
                                const arma::vec &initial_beta) {
  int N = X.n_rows;
  int P = X.n_cols;
  arma::vec proposed_beta = initial_beta;

  arma::vec eta(N), mu(N), grad(P), step(P);
  arma::mat XTWX(P, P);
  arma::mat XW(N, P);

  bool solver_success = true;
  bool converged = false;

  for (int i = 0; i < 12; i++) {
    eta = X * proposed_beta;

    for (int k = 0; k < N; ++k) {
      double e = eta[k];
      if (e < -10.0)
        e = -10.0;
      if (e > 10.0)
        e = 10.0;
      mu[k] = std::exp(e);
    }

    XW = X;
    XW.each_col() %= mu;

    XTWX = X.t() * XW;
    grad = X.t() * (y - mu);

    solver_success = arma::solve(step, XTWX, grad, arma::solve_opts::fast);
    if (!solver_success) {
      solver_success = arma::solve(step, XTWX, grad);
    }

    if (!solver_success || !step.is_finite()) {
      break;
    }

    proposed_beta += step;

    if (arma::norm(step) < 1e-3) {
      converged = true;
      break;
    }
  }

  // Post-estimation check for complete separation
  bool separated = false;
  if (!converged && solver_success) {
    // If the loop timed out but the matrix arithmetic didn't fail, check if
    // any row with a zero-outcome is pinned at or past the lower clamp
    // boundary.
    for (int k = 0; k < N; ++k) {
      if (y[k] == 0 && eta[k] <= -10.0) {
        separated = true;
        break;
      }
    }
  }

  return {proposed_beta, solver_success, separated};
}

// [[Rcpp::export]]
double compute_poisson_ll(const arma::vec &eta, const arma::vec &y) {
  return arma::dot(y, eta) - arma::accu(exp(eta)) -
         arma::accu(arma::lgamma(y + 1.0));
}

// [[Rcpp::export]]
arma::vec compute_poisson_ll_mat(const arma::mat &eta, const arma::vec y) {
  arma::vec ll(eta.n_cols);
  for (arma::uword i = 0; i < eta.n_cols; i++) {
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

PoissonPlResult glm_poisson_pl_cpp(const arma::mat &X, const arma::vec &y,
                                   const arma::vec &mle_coefs,
                                   const arma::vec &beta_vals, int m,
                                   bool approx, bool radial,
                                   std::atomic<bool> &singular_warning,
                                   uint32_t base_seed = 0, int eval_index = 0) {
  int n = X.n_rows;

  arma::vec eta = X * beta_vals;
  eta = arma::clamp(eta, -10.0, 10.0);
  arma::vec mu = arma::exp(eta);

  bool orig_separated = false;
  if (!mu.is_finite() || mu.max() > 1e5) {
    orig_separated = true;
  }

  arma::vec eta_hat = X * mle_coefs;
  eta_hat = arma::clamp(eta_hat, -10.0, 10.0);

  double true_ll = compute_poisson_ll(eta, y);
  double mle_ll = compute_poisson_ll(eta_hat, y);
  double f_x = true_ll - mle_ll;

  arma::mat Y_sim(n, m); // Pre-allocate full simulation matrix

  // Check if we are already inside an active outer OpenMP loop
  bool is_already_parallel = false;
#ifdef _OPENMP
  is_already_parallel = omp_in_parallel();
#endif

  // Only enable inner parallelization if requested AND not already inside an
  // outer parallel region
  bool run_inner_parallel = (radial || approx) && !is_already_parallel;

  // Derive a unique base seed for this specific grid evaluation point
  uint32_t eval_seed = base_seed + static_cast<uint32_t>(eval_index * 10007);

  // --- 1. Deterministic Parallel Data Generation ---
#pragma omp parallel for schedule(static) if (run_inner_parallel)
  for (int j = 0; j < m; ++j) {
    // Each simulation j gets a unique, deterministic seed
    std::mt19937 gen(eval_seed + j);

    for (int i = 0; i < n; ++i) {
      std::poisson_distribution<int> rpois(mu(i));
      Y_sim(i, j) = rpois(gen);
    }
  }

  // Parallel Model Fitting & Likelihood Evaluation
  int count_less = 0;
  double prop_separated = 0.0;

#pragma omp parallel for schedule(guided)                                      \
    reduction(+ : count_less, prop_separated) if (run_inner_parallel)
  for (int j = 0; j < m; ++j) {
    arma::vec y_sim_local = Y_sim.col(j);

    PoissonResult sim_res = fit_poisson_inner(X, y_sim_local, beta_vals);
    if (sim_res.separated) {
      prop_separated += 1.0;
    }

    arma::vec eta_hat_sim = X * sim_res.beta;

    double mle_sim = 0.0;
    double llX_j = 0.0;

    for (int i = 0; i < n; ++i) {
      double eta_hat_sim_clamped =
          std::max(-10.0, std::min(10.0, eta_hat_sim(i)));

      // In-place log-likelihood calculation
      mle_sim +=
          y_sim_local(i) * eta_hat_sim_clamped - std::exp(eta_hat_sim_clamped);
      llX_j += y_sim_local(i) * eta(i) - mu(i);
    }

    double f_X_j = llX_j - mle_sim;

    if (f_X_j <= f_x) {
      count_less++;
    }
  }

  double prop_sep = prop_separated / m;
  double poss = 1.0 * count_less / m;

  return {poss, orig_separated, prop_sep};
}
