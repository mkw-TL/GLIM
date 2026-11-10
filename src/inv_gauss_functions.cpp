#include "headers.h"
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace arma;

// =====================================================================
// INVERSE GAUSSIAN REGRESSION (1/MU^2 LINK)
// =====================================================================

// R does not have a native rinvgauss. Implemented is the Michael,
// Schucany, and Haas (1976) algorithm to simulate it via C++.
// Pass the pointer to the random number generator
// Don't need to export it as an R object, because R doesn't know what the
// mercene twister generator is
double rinvgauss_single(double mu, double lambda, std::mt19937 &gen) {
  // <> denotes the type expected out of this template
  // Loosely, templates can be thought of an instance of a class
  std::normal_distribution<double> rnorm(0.0, 1.0);
  std::uniform_real_distribution<double> runif(0.0, 1.0);
  double v = rnorm(gen);

  double y_sq = v * v;
  double x = mu + (mu * mu * y_sq) / (2.0 * lambda) -
             (mu / (2.0 * lambda)) *
                 std::sqrt(4.0 * mu * lambda * y_sq + mu * mu * y_sq * y_sq);

  double u = runif(gen);
  if (u <= mu / (mu + x)) {
    return x;
  } else {
    return (mu * mu) / x;
  }
}

// Non-allocating log-likelihood (the exported version just calls this)
inline double invgauss_ll_raw(const arma::vec &y, const arma::vec &mu,
                              double gamma_val) {
  double slog = 0.0, sq = 0.0;
  for (arma::uword i = 0; i < y.n_elem; ++i) {
    const double d = y[i] - mu[i];
    slog += std::log(y[i]);
    sq += d * d / (2.0 * mu[i] * mu[i] * y[i]);
  }
  return 0.5 * y.n_elem * std::log(gamma_val / (2.0 * M_PI)) - 1.5 * slog -
         gamma_val * sq;
}

// [[Rcpp::export]]
double mle_estimate_dispersion_inv_gauss(const arma::vec &y,
                                         const double ybar) {
  // Taking the mle wrt mu yields ybar (presuming mu =/= 0)
  return y.n_elem /
         (arma::accu(ybar * ybar * y / arma::dot((y - ybar), (y - ybar))));
}

// IRLS Inverse Gaussian regression solver (1/mu^2 link)
// [[Rcpp::export]]
arma::vec fit_invgauss_cpp(const arma::mat &X, const arma::vec &y,
                           const arma::vec &initial_beta, bool approx) {
  arma::vec proposed_beta = initial_beta;
  for (int i = 0; i < 30; i++) {
    arma::vec eta = X * proposed_beta;
    // Enforce strict positivity constraint for the link function
    eta.elem(arma::find(eta < 1e-6)).fill(1e-6);

    arma::vec mu = arma::pow(eta, -0.5);

    // For Inverse Gaussian 1/mu^2 link:
    // W = diag(mu^3) as opposed to diag(mu^3 / 4) for canonical link
    // Modified gradient step RHS = -2 * X^T (y - mu)
    arma::vec w = arma::pow(mu, 3.0);

    // Clamp extreme weights to preserve numerical stability
    w.elem(arma::find(w < 1e-8)).fill(1e-8);
    w.elem(arma::find(w > 1e8)).fill(1e8);
    // -2 is pulled out front to make computations nicer
    arma::mat XTWX = X.t() * arma::diagmat(w) * X;
    arma::vec grad = -2.0 * X.t() * (y - mu);

    arma::vec step;
    bool success = arma::solve(step, XTWX, grad, arma::solve_opts::no_approx);
    if (!success) {
      step = arma::pinv(XTWX) * grad;
    }

    // Line search (step halving) to ensure eta remains > 0
    arma::vec new_beta = proposed_beta + step;
    arma::vec new_eta = X * new_beta;
    int iter_halve = 0;
    while (arma::any(new_eta <= 0.0) && iter_halve < 10) {
      step /= 2.0;
      new_beta = proposed_beta + step;
      new_eta = X * new_beta;
      iter_halve++;
    }

    proposed_beta = new_beta;
    if (arma::norm(step) < 1e-6)
      break;
  }
  return proposed_beta;
}

// [[Rcpp::export]]
double compute_invgauss_ll(const arma::vec &y, const arma::vec &mu,
                           double gamma_val) {
  return invgauss_ll_raw(y, mu, gamma_val);
}

arma::vec fit_invgauss_cpp(const arma::mat &X, const arma::vec &y,
                           const arma::vec &initial_beta, bool approx,

                           InvGaussWorkspace &ws) {
  arma::vec beta = initial_beta;
  ws.eta = X * beta;

  for (int i = 0; i < 30; i++) {
    ws.eta = arma::clamp(ws.eta, 1e-6, arma::datum::inf);
    ws.mu = arma::pow(ws.eta, -0.5);
    ws.w = arma::clamp(ws.mu % ws.mu % ws.mu, 1e-8, 1e8);

    ws.XW = X;
    ws.XW.each_col() %= ws.w;
    ws.XTWX = X.t() * ws.XW;
    ws.grad = -2.0 * (X.t() * (y - ws.mu));

    if (!arma::solve(ws.step, ws.XTWX, ws.grad, arma::solve_opts::no_approx)) {
      ws.step = arma::pinv(ws.XTWX) * ws.grad;
    }

    ws.new_beta = beta + ws.step;
    ws.new_eta = X * ws.new_beta;
    int iter_halve = 0;
    while (arma::any(ws.new_eta <= 0.0) && iter_halve < 10) {
      ws.step *= 0.5;
      ws.new_beta = beta + ws.step;
      ws.new_eta = X * ws.new_beta;
      iter_halve++;
    }

    beta = ws.new_beta;
    ws.eta.swap(ws.new_eta); // next iteration's X*beta is already computed
    if (arma::norm(ws.step) < 1e-6)
      break;
  }
  return beta;
}

double glm_invgauss_pl_cpp(const arma::mat &X, const arma::vec &y,
                           const arma::vec &mle_coefs, double mle_val,
                           const arma::vec &beta_vals, int m, bool approx,
                           bool radial, uint32_t base_seed, int eval_index,
                           const arma::vec &mu_hat, arma::mat &Y_sim_workspace,
                           InvGaussWorkspace &ws) {
  const int n = X.n_rows;
  const int P = X.n_cols;

  // Local (not ws.*): the fitter overwrites ws.eta / ws.mu in the serial path
  const arma::vec eta = X * beta_vals;
  const arma::vec mu = arma::pow(eta, -0.5);

  double sbar = arma::accu(2 * mu % (y - mu) / (mu % mu)) / y.n_elem;
  double estimated_dispersion = arma::accu(
      ((y - mu) % (y - mu)) /
      (((mu % mu % mu) * (y.n_elem - beta_vals.n_elem)) * (1 + sbar)));
  double gamma = 1.0 / estimated_dispersion;

  mle_val = compute_invgauss_ll(y, mu_hat, gamma);
  const double f_x =
      invgauss_ll_raw(y, mu, gamma) - invgauss_ll_raw(y, mu_hat, gamma);

  bool is_already_parallel = false;
#ifdef _OPENMP
  is_already_parallel = omp_in_parallel();
#endif
  bool run_inner_parallel = (radial || approx) && !is_already_parallel;

  // is_already_parallel / run_inner_parallel / eval_seed as before

  // generation loop as before (uses mu, gamma, eval_seed)

  int count_less = 0;

#pragma omp parallel if (run_inner_parallel) reduction(+ : count_less)
  {
    InvGaussWorkspace local_ws;
    if (run_inner_parallel)
      local_ws = InvGaussWorkspace(n, P);
    InvGaussWorkspace &fit_ws = run_inner_parallel ? local_ws : ws;

#pragma omp for schedule(guided)
    for (int j = 0; j < m; ++j) {
      const arma::vec y_sim(Y_sim_workspace.colptr(j), n, false, true);

      arma::vec coefs = fit_invgauss_cpp(X, y_sim, beta_vals, approx, fit_ws);

      fit_ws.eta = arma::clamp(X * coefs, 1e-6, arma::datum::inf);
      fit_ws.mu = arma::pow(fit_ws.eta, -0.5);

      const double mle_sim = invgauss_ll_raw(y_sim, fit_ws.mu, gamma);
      const double llX_j = invgauss_ll_raw(y_sim, mu, gamma);

      if (llX_j - mle_sim <= f_x)
        count_less++;
    }
  }
  return static_cast<double>(count_less) / m;
}

// Wrapper: add `InvGaussWorkspace ws(X.n_rows, X.n_cols);` and pass it through.

// [[Rcpp::export]]
arma::vec compute_invgauss_ll_mat(const arma::vec &y, const arma::mat &mu,
                                  double gamma_val) {
  arma::vec ll(mu.n_cols);
  for (arma::uword i = 0; i < mu.n_cols; i++) {
    ll(i) = compute_invgauss_ll(y, mu.col(i).eval(), gamma_val);
  }
  return ll;
}

// 1. Core Engine (Accepts precomputed mu_hat AND workspace)
double
glm_invgauss_pl_cpp(const arma::mat &X, const arma::vec &y,
                    const arma::vec &mle_coefs, const double mle_val,
                    const arma::vec &beta_vals, int m, bool approx, bool radial,
                    uint32_t base_seed, int eval_index,
                    const arma::vec &mu_hat,      // <-- Precomputed passed in
                    arma::mat &Y_sim_workspace) { // <-- Passed by reference!
  int n = X.n_rows;

  arma::vec eta = X * beta_vals;
  arma::vec mu = arma::pow(eta, -0.5);

  double sbar = arma::accu(2 * mu % (y - mu) / (mu % mu)) / y.n_elem;
  double estimated_dispersion = arma::accu(
      ((y - mu) % (y - mu)) /
      (((mu % mu % mu) * (y.n_elem - beta_vals.n_elem)) * (1 + sbar)));
  double gamma = 1.0 / estimated_dispersion;

  double true_ll = compute_invgauss_ll(y, mu, gamma);
  double f_x = true_ll - mle_val;

  bool is_already_parallel = false;
#ifdef _OPENMP
  is_already_parallel = omp_in_parallel();
#endif
  bool run_inner_parallel = (radial || approx) && !is_already_parallel;
  uint32_t eval_seed = base_seed + static_cast<uint32_t>(eval_index * 10007);

  // --- 1. Deterministic Parallel Data Generation ---
  // Reuses thread workspace instead of allocating a new matrix
#pragma omp parallel for schedule(static) if (run_inner_parallel)
  for (int j = 0; j < m; ++j) {
    std::mt19937 gen(eval_seed + j);
    for (int i = 0; i < n; i++) {
      Y_sim_workspace(i, j) = rinvgauss_single(mu(i), gamma, gen);
    }
  }

  int count_less = 0;

  // --- 2. Parallel Model Fitting & Likelihood Evaluation ---
#pragma omp parallel for schedule(guided)                                      \
    reduction(+ : count_less) if (run_inner_parallel)
  for (int j = 0; j < m; ++j) {
    arma::vec y_sim = Y_sim_workspace.col(j);

    arma::vec coefs = fit_invgauss_cpp(X, y_sim, beta_vals, approx);
    arma::vec eta_hat_sim = X * coefs;

    // Validate eta_hat to compute simulated MLE likelihood
    eta_hat_sim.elem(arma::find(eta_hat_sim < 1e-6)).fill(1e-6);
    arma::vec mu_hat_sim = arma::pow(eta_hat_sim, -0.5);

    double mle_sim = compute_invgauss_ll(y_sim, mu_hat_sim, gamma);
    double llX_j = compute_invgauss_ll(y_sim, mu, gamma);

    double f_X_j = llX_j - mle_sim;

    if (f_X_j <= f_x) {
      count_less++;
    }
  }

  return static_cast<double>(count_less) / m;
}

// 2. Convenience Wrapper (Calculates MLE vectors, requires workspace)
double glm_invgauss_pl_cpp(const arma::mat &X, const arma::vec &y,
                           const arma::vec &mle_coefs, const double mle_val,
                           const arma::vec &beta_vals, int m, bool approx,
                           bool radial, uint32_t base_seed, int eval_index,
                           arma::mat &Y_sim_workspace, InvGaussWorkspace &ws) {

  // Precompute mu_hat once per evaluation
  arma::vec eta_hat = X * mle_coefs;
  arma::vec mu_hat = arma::pow(eta_hat, -0.5);

  return glm_invgauss_pl_cpp(X, y, mle_coefs, mle_val, beta_vals, m, approx,
                             radial, base_seed, eval_index, mu_hat,
                             Y_sim_workspace, ws);
}