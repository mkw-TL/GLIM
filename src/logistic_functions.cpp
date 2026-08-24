#include "armadillo"
#include "headers.h"
#include <random>
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
                            const arma::vec &initial_beta,
                            std::atomic<bool> &singular_warning) {
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

    // 1. Vectorized probability calculation
    p = 1.0 / (1.0 + arma::exp(-eta));

    // 2. Fast logical separation check
    separated = arma::any(p < 1e-8) || arma::any(p > (1.0 - 1e-8));
    if (separated)
      break;

    // 3. Vectorized weights and thresholding
    w = p % (1.0 - p); // '%' is element-wise multiplication in Armadillo
    w.elem(arma::find(w < 1e-6)).fill(1e-6);

    // 4. Vectorized column weighting (Replaces the nested loop)
    XW = X.each_col() % w;

    XTWX = X.t() * XW;
    grad = X.t() * (y - p);

    arma::vec step;
    bool success = arma::solve(step, XTWX, grad, arma::solve_opts::no_approx);

    if (!success) {
      singular_warning = true;
      step = arma::pinv(XTWX) * grad;
    }

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
// 1. Core Engine (Accepts precomputed scalars AND workspace)
LogisticPlResult
glm_logis_pl_cpp(const arma::mat &X, const arma::vec &y,
                 const arma::vec &mle_coefs, const arma::vec &beta_vals, int m,
                 bool approx, bool radial, std::atomic<bool> &singular_warning,
                 uint32_t base_seed, int eval_index, double mle_val,
                 bool orig_separated,
                 arma::mat &Y_sim_workspace) { // <-- Passed by reference!

  int n = X.n_rows;

  arma::vec eta = X * beta_vals;

  // Vectorized probability calculation (faster than element-wise division
  // object creation)
  arma::vec p(eta.n_elem);
  p = 1.0 / (1.0 + arma::exp(-eta));

  double sum_log_term = sum_softplus(eta);
  double f_x = orig_separated ? std::numeric_limits<double>::infinity()
                              : (arma::dot(y, eta) - sum_log_term - mle_val);

  bool is_already_parallel = false;
#ifdef _OPENMP
  is_already_parallel = omp_in_parallel();
#endif

  bool run_inner_parallel = (radial || approx) && !is_already_parallel;
  uint32_t eval_seed = base_seed + static_cast<uint32_t>(eval_index * 10007);

  // Use the pre-allocated workspace instead of allocating a new matrix
#pragma omp parallel for schedule(static) if (run_inner_parallel)
  for (int j = 0; j < m; ++j) {
    // Ultra-fast RNG initialization
    std::minstd_rand gen(eval_seed + j);
    for (int i = 0; i < n; ++i) {
      std::bernoulli_distribution rbern(p[i]);
      Y_sim_workspace(i, j) = rbern(gen) ? 1.0 : 0.0;
    }
  }

  arma::rowvec llX = eta.t() * Y_sim_workspace - sum_log_term;

  int count_less = 0;
  double prop_sep = 0;

#pragma omp parallel for schedule(guided)                                      \
    reduction(+ : count_less, prop_sep) if (run_inner_parallel)
  for (int j = 0; j < m; ++j) {
    arma::vec y_sim = Y_sim_workspace.col(j);
    LogisticResult sim_res =
        fit_logistic(X, y_sim, beta_vals, singular_warning);

    if (sim_res.separated) {
      prop_sep += 1.0;
    }

    arma::vec eta_sim_hat = X * sim_res.beta;
    double mle_sim = arma::dot(y_sim, eta_sim_hat) - sum_softplus(eta_sim_hat);
    double f_x_sim = llX[j] - mle_sim;

    if (f_x_sim <= f_x) {
      count_less++;
    }
  }

  return {static_cast<double>(count_less) / m, orig_separated,
          prop_sep /
              m}; // static_cast ensures we are not doing integer division
}

// 2. Convenience Wrapper (Calculates MLE scalars, still requires workspace)
LogisticPlResult glm_logis_pl_cpp(const arma::mat &X, const arma::vec &y,
                                  const arma::vec &mle_coefs,
                                  const arma::vec &beta_vals, int m,
                                  bool approx, bool radial,
                                  std::atomic<bool> &singular_warning,
                                  uint32_t base_seed, int eval_index,
                                  arma::mat &Y_sim_workspace) {

  arma::vec eta_hat = X * mle_coefs;
  arma::vec p_hat = 1.0 / (1.0 + arma::exp(-eta_hat));
  bool orig_separated =
      arma::any(p_hat < 1e-8) || arma::any(p_hat > (1.0 - 1e-8));
  double mle_val = arma::dot(y, eta_hat) - sum_softplus(eta_hat);

  return glm_logis_pl_cpp(X, y, mle_coefs, beta_vals, m, approx, radial,
                          singular_warning, base_seed, eval_index, mle_val,
                          orig_separated, Y_sim_workspace);
}

// [[Rcpp::export]]
double compute_logistic_ll(const arma::mat &X, const arma::vec &y,
                           const arma::vec &beta_vals) {
  std::atomic<bool> singular_warning(false);
  LogisticResult res = fit_logistic(X, y, beta_vals, singular_warning);
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
