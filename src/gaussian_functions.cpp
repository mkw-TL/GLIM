#include "headers.h"
#include <atomic>
#include <cmath>
#include <random>
#include <stdexcept>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace arma;

// Helper function to compute Gaussian log-likelihood
// [[Rcpp::export]]
double compute_gaussian_ll(const arma::vec &y, const arma::vec &mu,
                           double sigma) {
  double ll = -(y.n_elem / 2.0) * std::log(2.0 * M_PI * sigma * sigma) -
              arma::accu((y - mu) % (y - mu)) / (2 * sigma * sigma);
  return ll;
}

double glm_gaussian_pl_cpp(const arma::mat &X, const arma::vec &y,
                           const arma::vec &mle_coefs, const double mle_val,
                           const arma::vec &beta_vals, int m, bool approx,
                           bool radial, uint32_t base_seed, int eval_index,
                           double sigma, arma::mat &Y_sim_workspace,
                           GaussianWorkspace &ws) {
  const int n = X.n_rows;
  const int P = X.n_cols;

  // Exceptions thrown here are caught by the try/catch around the OpenMP loop
  // in fit_glm_omp_cpp and re-thrown as R errors after the region ends.
  if (!std::isfinite(sigma) || sigma <= 0.0) {
    throw std::runtime_error(
        "Gaussian: estimated sigma is not a positive finite number "
        "(is p >= n, or is the fit exact?)");
  }
  if (static_cast<int>(Y_sim_workspace.n_rows) != n ||
      static_cast<int>(Y_sim_workspace.n_cols) < m) {
    throw std::runtime_error("Gaussian: Y_sim workspace has wrong dimensions");
  }

  // TODO (speed): this SVD runs once per grid point. Compute it once in
  // fit_glm_omp_cpp and pass it in if this shows up in profiling.
  const arma::mat X_pinv = arma::pinv(X);

  // Hypothesized mean at the candidate beta, and the observed LR-type
  // statistic. Both log-likelihoods use the SAME sigma as the simulated
  // statistics below, so the comparison is consistent. (mle_val comes from
  // R's logLik(lm), which uses sigma^2 = SSE/n and would not cancel.)
  const arma::vec mu = X * beta_vals;
  const double ll_hat = compute_gaussian_ll(y, X * mle_coefs, sigma);
  const double f_x = compute_gaussian_ll(y, mu, sigma) - ll_hat;
  (void)mle_val; // kept in the signature for parity with the other families

  bool is_already_parallel = false;
#ifdef _OPENMP
  is_already_parallel = omp_in_parallel();
#endif
  const bool run_inner_parallel = (radial || approx) && !is_already_parallel;
  const uint32_t eval_seed =
      base_seed + static_cast<uint32_t>(eval_index * 10007);

  // Generation loop: simulate m responses under the hypothesized beta.
  // Serial on purpose: one generator, deterministic given eval_seed, and
  // thread-safe (unlike arma's / R's global RNG).
  {
    std::mt19937 gen(eval_seed);
    std::normal_distribution<double> nd(0.0, sigma);
    for (int j = 0; j < m; ++j) {
      for (int i = 0; i < n; ++i) {
        Y_sim_workspace(i, j) = mu(i) + nd(gen);
      }
    }
  }

  int count_less = 0;

#pragma omp parallel if (run_inner_parallel) reduction(+ : count_less)
  {
    GaussianWorkspace local_ws;
    if (run_inner_parallel)
      local_ws = GaussianWorkspace(n, P);
    GaussianWorkspace &fit_ws = run_inner_parallel ? local_ws : ws;

#pragma omp for schedule(guided)
    for (int j = 0; j < m; ++j) {
      const arma::vec y_sim(Y_sim_workspace.colptr(j), n, false, true);

      fit_ws.beta = X_pinv * y_sim; // OLS fit: O(nd), no QR
      fit_ws.mu_hat = X * fit_ws.beta;

      const double mle_sim = compute_gaussian_ll(y_sim, fit_ws.mu_hat, sigma);
      const double llX_j = compute_gaussian_ll(y_sim, mu, sigma);

      if (llX_j - mle_sim <= f_x)
        count_less++;
    }
  }
  return static_cast<double>(count_less) / m;
}

// Identity link Gaussian regression is just Ordinary Least Squares (OLS)
// No IRLS loop is required.
arma::vec fit_gaussian_cpp(const arma::mat &X, const arma::vec &y,
                           std::atomic<bool> &singular_warning) {
  arma::vec proposed_beta;
  bool success =
      arma::solve(proposed_beta, X, y,
                  arma::solve_opts::fast + arma::solve_opts::no_approx);
  if (!success) {
    // Fast solver failed (e.g. non-positive definite / rank deficient).
    success = arma::solve(proposed_beta, X, y, arma::solve_opts::no_approx);
  }

  if (!success) {
    // Both exact solvers failed (matrix is genuinely singular/collinear).
    // Flag the main thread to emit the R warning safely after the loop, then
    // fall back to the pseudoinverse (SVD) solution.
    singular_warning = true;
    proposed_beta = arma::pinv(X) * y;
  }
  return proposed_beta;
}

// [[Rcpp::export]]
arma::vec fit_gaussian_r(const arma::mat &X, const arma::vec &y) {
  std::atomic<bool> singular_warning = false;
  return (fit_gaussian_cpp(X, y, singular_warning));
}

// [[Rcpp::export]]
arma::vec compute_gaussian_ll_mat(const arma::vec &y, const arma::mat &mu,
                                  double sigma) {
  arma::vec ll(mu.n_cols);
  for (arma::uword i = 0; i < mu.n_cols; i++) {
    ll(i) = compute_gaussian_ll(y, mu.col(i).eval(), sigma);
  }
  return ll;
}

// [[Rcpp::export]]
double est_dispersion_normal(const arma::vec &y, const arma::vec &mu, int p) {
  return arma::accu((y - mu) % (y - mu)) / y.n_rows;
}