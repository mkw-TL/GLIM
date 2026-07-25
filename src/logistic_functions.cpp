#include "headers.h"
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

    // No temporary matrix allocations during weight multiplication
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

// Written by gemini. Uses the trick to seperate into cases
inline double sum_softplus(const arma::vec &eta) {
  double sum_val = 0.0;
  for (arma::uword i = 0; i < eta.n_elem; ++i) {
    sum_val += (eta(i) > 0) ? (eta(i) + std::log1p(std::exp(-eta(i))))
                            : std::log1p(std::exp(eta(i)));
  }
  return sum_val;
}

// The main simulation function
LogisticPlResult glm_logis_pl_cpp(const arma::mat &X, const arma::vec &y,
                                  const arma::vec &mle_coefs,
                                  const arma::vec &beta_vals, int m,
                                  bool approx, bool radial, uint32_t base_seed,
                                  int eval_index) {
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
  bool is_already_parallel = false;
#ifdef _OPENMP
  is_already_parallel = omp_in_parallel();
#endif

  // Only enable inner parallelization if requested AND not already inside an
  // outer parallel region
  bool run_inner_parallel = (radial || approx) && !is_already_parallel;

  // Derive a unique base seed for this specific grid evaluation point
  uint32_t eval_seed = base_seed + static_cast<uint32_t>(eval_index * 10007);

  arma::mat Y_sim(n, m);
  // --- 1. Deterministic Parallel Data Generation ---
#pragma omp parallel for schedule(static) if (run_inner_parallel)
  for (int j = 0; j < m; ++j) {
    std::mt19937 gen(eval_seed + j);
    for (int i = 0; i < n; ++i) {
      std::bernoulli_distribution rbern(p(i));
      Y_sim(i, j) = rbern(gen) ? 1.0 : 0.0;
    }
  }

  // Fast cross-product for all M simulations
  arma::rowvec llX = eta.t() * Y_sim - sum_log_term;

  // auto t_sim_end = std::chrono::high_resolution_clock::now();

  int count_less = 0;
  double prop_sep = 0;

// Inner loop: fit models entirely in C++
#pragma omp parallel for schedule(guided)                                      \
    reduction(+ : count_less, prop_sep) if (run_inner_parallel)
  for (int j = 0; j < m; ++j) {
    arma::vec y_sim = Y_sim.col(j);

    // Call inner solver returning LogisticResult struct
    LogisticResult sim_res = fit_logistic(X, y_sim, beta_vals);
    if (sim_res.separated) {
      prop_sep += 1.0;
    }

    arma::vec eta_sim_hat = X * sim_res.beta;

    // Fast softplus evaluation for simulated MLE log-likelihood
    double mle_sim = arma::dot(y_sim, eta_sim_hat) - sum_softplus(eta_sim_hat);

    // Use pre-computed llX[j] instead of recomputing inside the loop
    double f_x_sim = llX[j] - mle_sim;

    if (f_x_sim <= f_x) {
      count_less++;
    }
  }

  prop_sep /= m;
  double poss = static_cast<double>(count_less) / m;

  return {poss, orig_separated, prop_sep};
}

// [[Rcpp::export]]
double compute_logistic_ll(const arma::mat &X, const arma::vec &y,
                           const arma::vec &beta_vals) {
  LogisticResult res = fit_logistic(X, y, beta_vals);
  double ll = -10000000000;
  if (!res.separated) {
    ll = arma::dot(y, X * beta_vals) - sum_softplus(X * beta_vals);
  }
  return ll;
}

// [[Rcpp::export]]
arma::vec compute_logistic_ll_mat(const arma::mat &X, const arma::vec &y,
                                  const arma::mat &beta_vals) {
  arma::vec ll(X.n_cols);
  for (arma::uword i = 0; i < X.n_cols; i++) {
    arma::vec beta_val = beta_vals.col(i);
    ll(i) = compute_logistic_ll(X, y, beta_val);
  }
  return ll;
}
