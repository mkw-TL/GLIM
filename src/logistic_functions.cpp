#include "headers.h"
#include <atomic>
#include <cstdint>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

// NOTE: headers.h must come first (it defines ARMA_NO_DEBUG before Armadillo is
// included). Including "armadillo" before it silently leaves bounds checking
// on.

using namespace Rcpp;
using namespace arma;

// ---------------------------------------------------------------------------
// Portable uniform generator (SplitMix64).
//
// Why not std::minstd_rand + std::bernoulli_distribution?
//  * Seeding minstd_rand with consecutive integers (eval_seed + j) gives
//    strongly correlated streams across columns.
//  * The std:: distributions are implementation-defined, so results differ
//    between compilers/platforms.
// Here every column's start state is hashed from (seed, column), so streams are
// independent, and the output is identical everywhere.
// ---------------------------------------------------------------------------
static inline uint64_t mix64(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

void fill_uniforms(arma::mat &U, uint64_t seed, int m, bool use_threads) {
  const arma::uword n = U.n_rows;
#pragma omp parallel for schedule(static) if (use_threads)
  for (int j = 0; j < m; ++j) {
    uint64_t s =
        mix64(seed ^ mix64(static_cast<uint64_t>(j) + 0x632BE59BD9B4E019ULL));
    double *col = U.colptr(j);
    for (arma::uword i = 0; i < n; ++i) {
      s += 0x9E3779B97F4A7C15ULL;
      // Top 53 bits -> [0, 1)
      col[i] = static_cast<double>(mix64(s) >> 11) * (1.0 / 9007199254740992.0);
    }
  }
}

// ---------------------------------------------------------------------------
// IRLS logistic regression solver (canonical link). Allocation-free once the
// workspace is sized: every temporary lives in `ws`.
//
// ws.beta must hold the starting value on entry; it holds the fit on exit.
// Returns true if separation was detected (beta is then the last iterate).
// ---------------------------------------------------------------------------
bool fit_logistic_inplace(const arma::mat &X, const arma::vec &y,
                          LogisticWorkspace &ws) {
  const arma::uword n = X.n_rows;
  const arma::uword P = X.n_cols;
  ws.ensure(n, P);

  const double *yp = y.memptr();

  for (int it = 0; it < 15; ++it) {
    ws.eta = X * ws.beta;

    // One fused pass over the observations: probability, weight, residual.
    const double *e = ws.eta.memptr();
    double *wp = ws.w.memptr();
    double *rp = ws.resid.memptr();
    for (arma::uword i = 0; i < n; ++i) {
      const double pi = 1.0 / (1.0 + std::exp(-e[i]));
      // Written so that NaN also counts as "separated" (divergence).
      if (!(pi >= 1e-8 && pi <= 1.0 - 1e-8))
        return true;
      const double wi = pi * (1.0 - pi);
      wp[i] = (wi < 1e-6) ? 1e-6 : wi;
      rp[i] = yp[i] - pi;
    }

    // Column-wise assignment writes into the existing buffer (no allocation,
    // unlike `ws.XW = X.each_col() % ws.w`, which builds a temporary).
    for (arma::uword a = 0; a < P; ++a)
      ws.XW.col(a) = X.col(a) % ws.w;

    ws.XTWX = X.t() * ws.XW;
    ws.grad = X.t() * ws.resid;

    // XTWX is symmetric positive definite, so try Cholesky first.
    if (!arma::solve(ws.step, ws.XTWX, ws.grad,
                     arma::solve_opts::likely_sympd +
                         arma::solve_opts::no_approx)) {
      ws.step = arma::pinv(ws.XTWX) * ws.grad;
    }
    ws.beta += ws.step;

    // Newton decrement g' H^-1 g ~ 2 * (remaining log-likelihood gap). 1e-10 is
    // far below the 1/m resolution of the possibility estimate, and it saves
    // roughly one iteration over a ||step|| < 1e-6 test.
    if (arma::dot(ws.grad, ws.step) < 1e-10)
      break;
  }
  return false;
}

LogisticResult fit_logistic(const arma::mat &X, const arma::vec &y,
                            const arma::vec &initial_beta,
                            LogisticWorkspace &ws) {
  ws.beta = initial_beta;
  const bool separated = fit_logistic_inplace(X, y, ws);
  return {ws.beta, separated};
}

// ---------------------------------------------------------------------------
// Core engine. Takes pre-drawn uniforms, so a caller that evaluates the SAME
// replicate set at many betas (the Robbins-Monro loop) draws them once.
//
// Memory note: U (n x m) replaces the old Y_sim matrix, and the 0/1 response is
// built one column at a time into ws.y_sim, so peak memory is unchanged.
// ---------------------------------------------------------------------------
LogisticPlResult glm_logis_pl_from_uniforms(const arma::mat &X,
                                            const arma::vec &y, double mle_val,
                                            const arma::vec &beta_vals,
                                            const arma::mat &U, int m,
                                            bool run_inner_parallel,
                                            LogisticWorkspace &ws) {
  const arma::uword n = X.n_rows;
  const arma::uword P = X.n_cols;
  ws.ensure(n, P);

  // Quantities at the proposal beta. Stored in eta0/p0, which the IRLS fits do
  // NOT touch (important: in the serial path fits reuse `ws` itself).
  ws.eta0 = X * beta_vals;
  {
    const double *e = ws.eta0.memptr();
    double *p0 = ws.p0.memptr();
    for (arma::uword i = 0; i < n; ++i)
      p0[i] = 1.0 / (1.0 + std::exp(-e[i]));
  }
  const double sum_log_term = sum_softplus(ws.eta0);
  const double f_x = arma::dot(y, ws.eta0) - sum_log_term - mle_val;

  int count_less = 0;
  double prop_sep = 0.0;

#pragma omp parallel if (run_inner_parallel) reduction(+ : count_less, prop_sep)
  {
    // Threads get private scratch space; the serial path reuses ws (no
    // allocation).
    LogisticWorkspace local_ws;
    if (run_inner_parallel)
      local_ws = LogisticWorkspace(n, P);
    LogisticWorkspace &fit_ws = run_inner_parallel ? local_ws : ws;

    const double *p0 = ws.p0.memptr();

#pragma omp for schedule(guided)
    for (int j = 0; j < m; ++j) {
      // y_sim = 1{U < p}
      const double *ucol = U.colptr(j);
      double *ys = fit_ws.y_sim.memptr();
      for (arma::uword i = 0; i < n; ++i)
        ys[i] = (ucol[i] < p0[i]) ? 1.0 : 0.0;

      const double ll_x = arma::dot(ws.eta0, fit_ws.y_sim) - sum_log_term;

      fit_ws.beta = beta_vals; // same-size copy, no allocation
      if (fit_logistic_inplace(X, fit_ws.y_sim, fit_ws))
        prop_sep += 1.0;

      fit_ws.eta = X * fit_ws.beta;
      const double mle_sim =
          arma::dot(fit_ws.y_sim, fit_ws.eta) - sum_softplus(fit_ws.eta);
      if (ll_x - mle_sim <= f_x)
        ++count_less;
    }
  }

  return {static_cast<double>(count_less) / m, prop_sep / m};
}

// ---------------------------------------------------------------------------
// Seeded entry point (used by the grid / elliptical paths, where each beta gets
// its own replicate set). Y_sim_workspace now holds the UNIFORMS, not 0/1s.
// ---------------------------------------------------------------------------
LogisticPlResult glm_logis_pl_cpp(const arma::mat &X, const arma::vec &y,
                                  const arma::vec &mle_coefs, double mle_val,
                                  const arma::vec &beta_vals, int m,
                                  bool approx, bool radial, uint32_t base_seed,
                                  int eval_index, arma::mat &Y_sim_workspace,
                                  LogisticWorkspace &ws) {

  const arma::uword n = X.n_rows;

  bool is_already_parallel = false;
#ifdef _OPENMP
  is_already_parallel = omp_in_parallel();
#endif
  const bool run_inner_parallel = (radial || approx) && !is_already_parallel;
  const uint64_t eval_seed = static_cast<uint64_t>(base_seed) +
                             10007ULL * static_cast<uint64_t>(eval_index);

  if (Y_sim_workspace.n_rows != n ||
      Y_sim_workspace.n_cols < static_cast<arma::uword>(m)) {
    Y_sim_workspace.set_size(n, m);
  }
  fill_uniforms(Y_sim_workspace, eval_seed, m, run_inner_parallel);

  return glm_logis_pl_from_uniforms(X, y, mle_val, beta_vals, Y_sim_workspace,
                                    m, run_inner_parallel, ws);
}

// 2. Convenience Wrapper (Calculates MLE scalars, still requires workspace)
LogisticPlResult glm_logis_pl_cpp(const arma::mat &X, const arma::vec &y,
                                  const arma::vec &mle_coefs,
                                  const arma::vec &beta_vals, int m,
                                  bool approx, bool radial,

                                  uint32_t base_seed, int eval_index,
                                  arma::mat &Y_sim_workspace) {

  arma::vec eta_hat = X * mle_coefs;
  double mle_val = arma::dot(y, eta_hat) - sum_softplus(eta_hat);

  LogisticWorkspace ws(X.n_rows, X.n_cols);
  return glm_logis_pl_cpp(X, y, mle_coefs, mle_val, beta_vals, m, approx,
                          radial, base_seed, eval_index, Y_sim_workspace, ws);
}

static double logistic_ll_impl(const arma::mat &X, const arma::vec &y,
                               const arma::vec &beta_vals,
                               std::atomic<bool> &warn, LogisticWorkspace &ws) {
  LogisticResult res = fit_logistic(X, y, beta_vals, ws);
  if (res.separated)
    return -1e10;
  arma::vec eta = X * beta_vals;
  return arma::dot(y, eta) - sum_softplus(eta);
}

// [[Rcpp::export]]
double compute_logistic_ll(const arma::mat &X, const arma::vec &y,
                           const arma::vec &beta_vals) {
  std::atomic<bool> warn(false);
  LogisticWorkspace ws(X.n_rows, X.n_cols);
  return logistic_ll_impl(X, y, beta_vals, warn, ws);
}

// [[Rcpp::export]]
arma::vec compute_logistic_ll_mat(const arma::mat &X, const arma::vec &y,
                                  const arma::mat &beta_vals) {
  arma::vec ll(beta_vals.n_cols);
  std::atomic<bool> warn(false);
  LogisticWorkspace ws(X.n_rows, X.n_cols);
  for (arma::uword i = 0; i < beta_vals.n_cols; i++)
    ll(i) = logistic_ll_impl(X, y, beta_vals.col(i), warn, ws);
  return ll;
}