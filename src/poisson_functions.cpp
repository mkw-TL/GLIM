#include "headers.h"
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

PoissonResult fit_poisson_inner(const arma::mat &X, const arma::vec &y,
                                const arma::vec &initial_beta,

                                PoissonWorkspace &ws) {
  arma::vec beta = initial_beta;
  bool solver_success = true;
  bool converged = false;

  for (int i = 0; i < 12; i++) {
    ws.eta = X * beta;
    ws.mu = arma::exp(arma::clamp(ws.eta, -10.0, 10.0));

    // Copy-assign into same-size ws.XW reuses its buffer (truly
    // allocation-free)
    ws.XW = X;
    ws.XW.each_col() %= ws.mu;
    ws.XTWX = X.t() * ws.XW;
    ws.grad = X.t() * (y - ws.mu);

    solver_success = arma::solve(ws.step, ws.XTWX, ws.grad,
                                 arma::solve_opts::no_approx +
                                     arma::solve_opts::likely_sympd);
    if (!solver_success) {
      ws.step = arma::pinv(ws.XTWX) * ws.grad;
      solver_success = ws.step.is_finite();
    }
    if (!solver_success || !ws.step.is_finite())
      break;

    beta += ws.step;
    if (arma::norm(ws.step) < 1e-3) {
      converged = true;
      break;
    }
  }

  bool separated = false;
  if (!converged && solver_success) {
    for (arma::uword k = 0; k < X.n_rows; ++k) {
      if (y[k] == 0 && ws.eta[k] <= -10.0) {
        separated = true;
        break;
      }
    }
  }
  return {beta, separated};
}

// Standalone wrapper for other callers
// [[Rcpp::export]]
arma::vec fit_poisson_log_cpp(const arma::mat &X, const arma::vec &y,
                              const arma::vec &initial_beta) {
  PoissonWorkspace ws(X.n_rows, X.n_cols);
  return fit_poisson_inner(X, y, initial_beta, ws).beta;
}

PoissonPlResult glm_poisson_pl_cpp(const arma::mat &X, const arma::vec &y,
                                   const arma::vec &mle_coefs, double mle_val,
                                   const arma::vec &beta_vals, int m,
                                   bool approx, bool radial, uint32_t base_seed,
                                   int eval_index, arma::mat &Y_sim_workspace,
                                   PoissonWorkspace &ws) {
  const int n = X.n_rows;
  const int P = X.n_cols;

  // Local, not ws: in the serial path fit_ws aliases ws and the fitter
  // overwrites ws.eta / ws.mu while the j-loop still needs these.
  const arma::vec eta = arma::clamp(X * beta_vals, -10.0, 10.0);
  const arma::vec mu = arma::exp(eta);

  const double f_x = compute_poisson_ll(eta, y) - mle_val;
  const double sum_mu = arma::accu(mu); // constant across j, hoisted

  bool is_already_parallel = false;
#ifdef _OPENMP
  is_already_parallel = omp_in_parallel();
#endif
  const bool run_inner_parallel = (radial || approx) && !is_already_parallel;
  const uint32_t eval_seed =
      base_seed + static_cast<uint32_t>(eval_index * 10007);

#pragma omp parallel for schedule(static) if (run_inner_parallel)
  for (int j = 0; j < m; ++j) {
    std::mt19937 gen(eval_seed + j);
    for (int i = 0; i < n; ++i) {
      std::poisson_distribution<int> rpois(mu(i));
      Y_sim_workspace(i, j) = rpois(gen);
    }
  }

  int count_less = 0;
  double prop_separated = 0.0;

#pragma omp parallel if (run_inner_parallel)                                   \
    reduction(+ : count_less, prop_separated)
  {
    PoissonWorkspace local_ws;
    if (run_inner_parallel)
      local_ws = PoissonWorkspace(n, P);
    PoissonWorkspace &fit_ws = run_inner_parallel ? local_ws : ws;

#pragma omp for schedule(guided)
    for (int j = 0; j < m; ++j) {
      const arma::vec y_sim(Y_sim_workspace.colptr(j), n, false, true);

      PoissonResult sim_res = fit_poisson_inner(X, y_sim, beta_vals, fit_ws);
      if (sim_res.separated)
        prop_separated += 1.0;

      fit_ws.eta = arma::clamp(X * sim_res.beta, -10.0, 10.0);
      const double mle_sim =
          arma::dot(y_sim, fit_ws.eta) - arma::accu(arma::exp(fit_ws.eta));
      const double llX_j = arma::dot(y_sim, eta) - sum_mu;

      if (llX_j - mle_sim <= f_x)
        count_less++;
    }
  }

  return {static_cast<double>(count_less) / m, prop_separated / m};
}

// Wrapper
PoissonPlResult glm_poisson_pl_cpp(const arma::mat &X, const arma::vec &y,
                                   const arma::vec &mle_coefs, double mle_val,
                                   const arma::vec &beta_vals, int m,
                                   bool approx, bool radial,

                                   uint32_t base_seed, int eval_index,
                                   arma::mat &Y_sim_workspace) {
  arma::vec eta_hat = arma::clamp(X * mle_coefs, -10.0, 10.0);
  PoissonWorkspace ws(X.n_rows, X.n_cols);
  return glm_poisson_pl_cpp(X, y, mle_coefs, mle_val, beta_vals, m, approx,
                            radial, base_seed, eval_index, Y_sim_workspace, ws);
}
