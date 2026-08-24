#include "headers.h"
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace arma;

// Identity link Gaussian regression is just Ordinary Least Squares (OLS)
// No IRLS loop is required.
arma::vec fit_gaussian_cpp(const arma::mat &X, const arma::vec &y,
                           std::atomic<bool> &singular_warning) {
  arma::vec proposed_beta;
  // Now we do step-halfing. Ensures that any improvement we make does
  // decrease the varaince. Avoids wild steps. Calculate the proposed step
  bool success =
      arma::solve(proposed_beta, X, y,
                  arma::solve_opts::fast + arma::solve_opts::no_approx);
  if (!success) {
    // Fast solver failed (e.g. non-positive definite / rank deficient).
    // Try the general solver without automatic SVD fallback or console
    // prints.
    success = arma::solve(proposed_beta, X, y, arma::solve_opts::no_approx);
  }

  if (!success) {
    // Both exact solvers failed (matrix is genuinely singular/collinear).
    // 1. Flag the main thread to emit the R warning safely after the loop
    singular_warning = true;

    // 2. Compute the approximate solution via pseudoinverse (SVD)
    proposed_beta = arma::pinv(X) * y;
    success = true; // Mark as resolved via approximation
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
  for (arma::uword i = 0; i < mu.n_cols; i++) {
    ll(i) = compute_gaussian_ll(y, mu.col(i).eval(), sigma);
  }
  return ll;
}

// [[Rcpp::export]]
double est_dispersion_normal(const arma::vec &y, const arma::vec &mu, int p) {
  return arma::accu((y - mu) % (y - mu)) / (y.n_elem - p);
}

// 1. Core Engine (Accepts precomputed scalars AND workspace)
double glm_gaussian_pl_cpp(
    const arma::mat &X, const arma::vec &y, const arma::vec &mle_coefs,
    const arma::vec &beta_vals, int m, bool approx, bool radial,
    std::atomic<bool> &singular_warning, uint32_t base_seed, int eval_index,
    double sigma, double mle_val, // <-- Precomputed passed in
    arma::mat &Y_sim_workspace) { // <-- Passed by reference!
  int n = X.n_rows;

  arma::vec mu = X * beta_vals; // Identity link
  double true_ll = compute_gaussian_ll(y, mu, sigma);
  double f_x = true_ll - mle_val;

  // Check if we are already inside an active outer OpenMP loop
  bool is_already_parallel = false;
#ifdef _OPENMP
  is_already_parallel = omp_in_parallel();
#endif
  bool run_inner_parallel = (radial || approx) && !is_already_parallel;
  uint32_t eval_seed = base_seed + static_cast<uint32_t>(eval_index * 10007);

  // --- 1. Deterministic Parallel Data Generation ---
  // Use the pre-allocated workspace
#pragma omp parallel for schedule(static) if (run_inner_parallel)
  for (int j = 0; j < m; ++j) {
    std::mt19937 gen(eval_seed + j);
    std::normal_distribution<double> rnorm(0.0, 1.0);
    for (int i = 0; i < n; i++) {
      Y_sim_workspace(i, j) = mu(i) + sigma * rnorm(gen);
    }
  }

  int count_less = 0;

  // --- 2. Parallel Model Fitting & Likelihood Evaluation ---
#pragma omp parallel for schedule(guided)                                      \
    reduction(+ : count_less) if (run_inner_parallel)
  for (int j = 0; j < m; ++j) {
    arma::vec y_sim = Y_sim_workspace.col(j);

    arma::vec sim_coefs = fit_gaussian_cpp(X, y_sim, singular_warning);
    arma::vec mu_hat_sim = X * sim_coefs;
    double mle_sim = compute_gaussian_ll(y_sim, mu_hat_sim, sigma);
    double llX_j = compute_gaussian_ll(y_sim, mu, sigma);

    double f_X_j = llX_j - mle_sim;

    if (f_X_j <= f_x) {
      count_less++;
    }
  }

  return static_cast<double>(count_less) / m;
}

// 2. Convenience Wrapper (Calculates MLE scalars, requires workspace)
double glm_gaussian_pl_cpp(const arma::mat &X, const arma::vec &y,
                           const arma::vec &mle_coefs,
                           const arma::vec &beta_vals, int m, bool approx,
                           bool radial, std::atomic<bool> &singular_warning,
                           uint32_t base_seed, int eval_index,
                           arma::mat &Y_sim_workspace) {

  // Precompute MLE invariants once per evaluation
  int p = mle_coefs.n_elem;
  arma::vec mu_hat = X * mle_coefs;
  double sigma = std::sqrt(est_dispersion_normal(y, mu_hat, p));
  double mle_val = compute_gaussian_ll(y, mu_hat, sigma);

  return glm_gaussian_pl_cpp(X, y, mle_coefs, beta_vals, m, approx, radial,
                             singular_warning, base_seed, eval_index, sigma,
                             mle_val, Y_sim_workspace);
}