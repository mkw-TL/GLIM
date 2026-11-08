#include "headers.h"
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace arma;

double calculate_deviance_gamma(const arma::vec &y, const arma::vec &mu) {
  return 2.0 * arma::accu(y / mu - arma::log(y / mu) - 1.0);
}

arma::vec fit_gamma_log_cpp(const arma::mat &X, const arma::mat &XtX,
                            const arma::vec &y, const arma::vec &initial_beta,
                            bool approx, GammaWorkspace &ws) {
  const double eps = std::numeric_limits<double>::epsilon();
  arma::vec beta = initial_beta;

  ws.mu = arma::clamp(arma::exp(arma::clamp(X * beta, -30.0, 30.0)), eps, 1e15);
  double current_dev = calculate_deviance_gamma(y, ws.mu);

  for (int i = 0; i < 20; i++) {
    // ws.mu already holds mu at the current beta (initially above, later
    // from the accepted step), so there is no need to recompute eta/mu here.
    ws.grad = X.t() * (y / ws.mu - 1.0);

    bool success =
        arma::solve(ws.step, XtX, ws.grad,
                    arma::solve_opts::fast + arma::solve_opts::no_approx);
    if (!success)
      success = arma::solve(ws.step, XtX, ws.grad, arma::solve_opts::no_approx);
    if (!success) {
      ws.step = arma::pinv(XtX) * ws.grad;
    }

    ws.temp_beta = beta + ws.step;
    ws.mu_try = arma::clamp(
        arma::exp(arma::clamp(X * ws.temp_beta, -30.0, 30.0)), eps, 1e15);
    double proposed_dev = calculate_deviance_gamma(y, ws.mu_try);

    int half_iter = 0;
    while ((!std::isfinite(proposed_dev) || proposed_dev > current_dev) &&
           half_iter < 25) {
      ws.step *= 0.5;
      ws.temp_beta = beta + ws.step;
      ws.mu_try = arma::clamp(
          arma::exp(arma::clamp(X * ws.temp_beta, -30.0, 30.0)), eps, 1e15);
      proposed_dev = calculate_deviance_gamma(y, ws.mu_try);
      half_iter++;
    }

    beta = ws.temp_beta;
    ws.mu.swap(ws.mu_try); // accepted mu becomes current mu, no copy
    current_dev = proposed_dev;

    if (arma::norm(ws.step) < 1e-6)
      break;
  }
  return beta;
}

// Helper function to compute full Gamma log-likelihood matching R's logLik()
// [[Rcpp::export]]
double compute_gamma_ll(const arma::vec &y, const arma::vec &eta,
                        double shape) {
  arma::vec eta_clamped = arma::clamp(eta, -50.0, 50.0);
  double term1 = y.n_elem * (shape * std::log(shape) - std::lgamma(shape));
  double term2 = (shape - 1.0) * arma::sum(arma::log(y));
  double term3 =
      -shape * (arma::dot(y, arma::exp(-eta_clamped)) + arma::sum(eta_clamped));
  return term1 + term2 + term3;
}

// [[Rcpp::export]]
arma::vec compute_gamma_ll_mat(const arma::vec &y, const arma::mat &eta,
                               double shape) {
  arma::vec ll(eta.n_cols);
  for (arma::uword i = 0; i < eta.n_cols; i++) {
    ll(i) = compute_gamma_ll(y, eta.col(i), shape);
  }
  return ll;
}

// Uses Fletcher's correction to Pearson's MoM estimator. (Not yet -- keeping it
// simple)
// [[Rcpp::export]]
double pearson_estimate_dispersion_gamma(arma::vec y, arma::vec mu_hat,
                                         double p) {
  // // Have canceled out a mu on top and bottom
  // double sbar = arma::accu((y - mu_hat)/ (2*mu_hat))/y.n_elem;
  return (double)arma::accu(arma::square(y - mu_hat) / arma::square(mu_hat)) /
         ((y.n_elem - p));
}

// [[Rcpp::export]]
double mle_estimate_dispersion_gamma(arma::vec y, arma::vec &mu_hat, double p) {
  // Safety net against zero or negative values disrupting logs
  arma::vec ratio = y / mu_hat;
  ratio.elem(arma::find(ratio < 1e-8)).fill(1e-8);

  // D_mean is the mean deviance-like component: mean( y/mu - log(y/mu) - 1 )
  // Note that this is the unscaled deviance. The estimated dispersion parameter
  // are the same for a particular dataset.
  double D_mean = arma::mean(ratio - arma::log(ratio) - 1.0);

  // Edge case: If the model fits perfectly, D_mean hits 0, implying infinite
  // shape.
  if (D_mean <= 1e-10) {
    return 99999.0;
  }

  // Compute an initial guess using Thom's Approximation
  // (Uses a laurent series to approximate digamma, and then solve the resulting
  // quadratic) We want to taylor expand nu around zero, but we have 1/nu. There
  // is a pole there, and thus we can't use taylor approx Laurent expansion
  // still valid, though.
  double nu = (1.0 + std::sqrt(1.0 + (4.0 / 3.0) * D_mean)) / (4.0 * D_mean);

  // 1D Newton-Raphson Loop to solve: log(nu) - digamma(nu) - D_mean = 0
  int max_iter = 100;
  double tol = 1e-8;

  for (int i = 0; i < max_iter; i++) {
    // psi is the digamma function
    double f = std::log(nu) - boost::math::digamma(nu) - D_mean;
    double f_prime = (1.0 / nu) - boost::math::polygamma(1, nu);

    double step = f / f_prime;
    double next_nu = nu - step;

    // Safety measure: Newton steps can occasionally swing negative
    // if the curve is exceptionally steep. If so, reduce the current dispersion
    // by half.
    if (next_nu <= 0) {
      next_nu = nu * 0.5;
    }

    // Check for convergence
    if (std::abs(next_nu - nu) < tol) {
      nu = next_nu;
      break;
    }

    nu = next_nu;
  }
  if (nu < 1e-5) {
    return 1e-5;
  }
  return (1 / nu);
}

// 1. Core Engine (Accepts precomputed eta_hat AND workspace)
double glm_gamma_pl_cpp(const arma::mat &X, const arma::mat &XtX,
                        const arma::vec &y, const arma::vec &mle_coefs,
                        double mle_val, const arma::vec &beta_vals, int m,
                        bool approx, bool radial, uint32_t base_seed,
                        int eval_index, const arma::vec &eta_hat,
                        arma::mat &Y_sim_workspace, GammaWorkspace &ws) {
  const int n = X.n_rows;
  const int P = X.n_cols;

  ws.eta = arma::clamp(X * beta_vals, -50.0, 50.0); // avoids D_mean explosions
  ws.mu = arma::exp(ws.eta);
  ws.mu.elem(arma::find(ws.mu < 1e-8)).fill(1e-8);

  const double dispersion =
      mle_estimate_dispersion_gamma(y, ws.mu, beta_vals.n_elem);
  const double shape = 1.0 / dispersion;
  const double true_ll = compute_gamma_ll(y, ws.eta, shape);
  const double mle_ll = compute_gamma_ll(y, eta_hat, shape);
  const double f_x = true_ll - mle_ll;

  bool is_already_parallel = false;
#ifdef _OPENMP
  is_already_parallel = omp_in_parallel();
#endif
  const bool run_inner_parallel = (radial || approx) && !is_already_parallel;
  const uint32_t eval_seed =
      base_seed + static_cast<uint32_t>(eval_index * 10007);

  // ws.mu is overwritten per-thread below only when run serially, so take
  // the scale vector now.
  const arma::vec scale = ws.mu * dispersion;

#pragma omp parallel for schedule(static) if (run_inner_parallel)
  for (int j = 0; j < m; ++j) {
    std::mt19937 gen(eval_seed + j);
    for (int i = 0; i < n; i++) {
      std::gamma_distribution<double> rgamma(shape, scale(i));
      double sim_val = rgamma(gen);
      Y_sim_workspace(i, j) = (sim_val < 1e-10) ? 1e-10 : sim_val;
    }
  }

  int count_less = 0;

#pragma omp parallel if (run_inner_parallel) reduction(+ : count_less)
  {
    // Threads get private scratch; the serial path reuses ws (no allocation).
    GammaWorkspace local_ws;
    if (run_inner_parallel)
      local_ws = GammaWorkspace(n, P);
    GammaWorkspace &fit_ws = run_inner_parallel ? local_ws : ws;

#pragma omp for schedule(guided)
    for (int j = 0; j < m; ++j) {
      // Zero-copy view of column j (no allocation, no copy)
      const arma::vec y_sim(Y_sim_workspace.colptr(j), n, false, true);

      arma::vec beta_sim_hat =
          fit_gamma_log_cpp(X, XtX, y_sim, beta_vals, approx, fit_ws);

      arma::vec eta_sim_hat = X * beta_sim_hat;
      arma::vec mu_sim_hat = arma::exp(eta_sim_hat);

      double shape_sim_hat = 1.0 / mle_estimate_dispersion_gamma(
                                       y_sim, mu_sim_hat, beta_sim_hat.n_elem);

      double llX_j = compute_gamma_ll(y_sim, ws.eta, shape_sim_hat);
      double mle_sim = compute_gamma_ll(y_sim, eta_sim_hat, shape_sim_hat);

      if (llX_j - mle_sim < f_x)
        count_less++;
    }
  }

  return static_cast<double>(count_less) / m;
}

// 2. Convenience Wrapper (Calculates MLE vectors, requires workspace)
double glm_gamma_pl_cpp(const arma::mat &X, const arma::mat &XtX,
                        const arma::vec &y, const arma::vec &mle_coefs,
                        const double mle_val, const arma::vec &beta_vals, int m,
                        bool approx, bool radial, uint32_t base_seed,
                        int eval_index, arma::mat &Y_sim_workspace,
                        GammaWorkspace &ws) {
  arma::vec eta_hat = arma::clamp(X * mle_coefs, -50.0, 50.0);
  return glm_gamma_pl_cpp(X, XtX, y, mle_coefs, mle_val, beta_vals, m, approx,
                          radial, base_seed, eval_index, eta_hat,
                          Y_sim_workspace, ws);
}