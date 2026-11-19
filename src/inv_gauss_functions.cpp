#include "glim_prof.h"
#include "headers.h"
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace arma;

// =====================================================================
// INVERSE GAUSSIAN REGRESSION (1/MU^2 LINK)
//
// Key identity used throughout. With eta = 1/mu^2 (so 1/mu = sqrt(eta)) the
// inverse Gaussian deviance is
//     Dev(y; eta) = sum( y/mu^2 - 2/mu + 1/y ) = sum(y*eta) - 2*sum(sqrt(eta))
//     + sum(1/y)
// and the log-likelihood is  const - (gamma/2) * Dev.   Two consequences:
//   * sum(1/y) and gamma both cancel in the comparison the plausibility makes
//     (llX_j - mle_sim <= f_x  <=>  D_j >= D_obs, D = Dev(eta_beta) -
//     Dev(eta_hat)), so no log(y), no 1/y, no per-observation likelihood is
//     needed;
//   * gamma only enters through the simulation.
// =====================================================================

namespace {

// Michael, Schucany & Haas (1976) draw with the normal / uniform generators
// supplied by the caller (constructing std::normal_distribution per draw, as
// before, throws away half of every polar-method pair).
//   x = mu + mu^2 v^2/(2 lambda) - mu/(2 lambda) sqrt(4 mu lambda v^2 + mu^2
//   v^4)
// is rewritten as  x = mu / (1 + a/2 + sqrt(a (a + 4)) / 2),  a = mu v^2 /
// lambda, which is algebraically identical (p^2 - q^2 = 1) but does not cancel
// catastrophically when a is large.
inline double rinvgauss_fast(double mu, double lambda,
                             std::normal_distribution<double> &rnorm,
                             std::uniform_real_distribution<double> &runif,
                             std::mt19937 &gen) {
  const double v = rnorm(gen);
  const double a = mu * v * v / lambda;
  const double x = mu / (1.0 + 0.5 * a + 0.5 * std::sqrt(a * (a + 4.0)));
  const double u = runif(gen);
  return (u <= mu / (mu + x)) ? x : (mu * mu) / x;
}

constexpr double kEtaFloor = 1e-6;  // eta is clamped to >= this inside the fit
constexpr double kStepTol = 1e-6;   // convergence: ||step|| < kStepTol
constexpr double kDevSlack = 1e-10; // relative slack on the descent check
constexpr int kMaxIter = 30;
constexpr int kMaxHalve = 20;

// out = X b  (column passes; no BLAS call, no allocation)
inline void ig_matvec(const arma::mat &X, const double *__restrict b,
                      double *__restrict out) {
  const arma::uword n = X.n_rows, P = X.n_cols;
  const double *__restrict x0 = X.colptr(0);
  const double b0 = b[0];
#pragma omp simd
  for (arma::uword i = 0; i < n; ++i)
    out[i] = x0[i] * b0;
  for (arma::uword a = 1; a < P; ++a) {
    const double *__restrict xa = X.colptr(a);
    const double ba = b[a];
#pragma omp simd
    for (arma::uword i = 0; i < n; ++i)
      out[i] += xa[i] * ba;
  }
}

// sum(y*e) - 2*sum(sqrt(e)),  e = max(eta, kEtaFloor)   (= Dev minus sum(1/y))
// Also returns min(eta) so positivity is checked in the same pass.
inline double ig_dev(const double *__restrict y, const double *__restrict eta,
                     arma::uword n, double &min_eta) {
  double s = 0.0, mn = eta[0];
#pragma omp simd reduction(+ : s) reduction(min : mn)
  for (arma::uword i = 0; i < n; ++i) {
    const double ei = eta[i];
    const double e = ei > kEtaFloor ? ei : kEtaFloor;
    s += y[i] * e - 2.0 * std::sqrt(e);
    mn = ei < mn ? ei : mn;
  }
  min_eta = mn;
  return s;
}

// Fisher scoring for the 1/mu^2 link. `beta` is the start on entry and the
// fit on return. On return ws.eta = X * beta, dev_out = ig_dev at the fit.
// Same update as before:  step = (X'WX)^{-1} * (-2 X'(y - mu)),  W =
// diag(mu^3), but (a) one fused pass builds w and the residual vector, (b) X'WX
// and the gradient are accumulated column-wise (no n x P copy of X per
// iteration), (c) the half-step reuses eta by linearity (no second X*b), and
// (d) a deviance-descent check is added to the positivity check.
void fit_invgauss_core(const arma::mat &X, const double *__restrict y,
                       arma::vec &beta, InvGaussWorkspace &ws, double &dev_out,
                       int &iters, bool &converged) {
  const arma::uword n = X.n_rows, P = X.n_cols;
  ws.eta.set_size(n);
  ws.new_eta.set_size(n);
  ws.w.set_size(n);
  ws.mu.set_size(n); // reused as the scaled residual vector
  ws.XTWX.set_size(P, P);
  ws.grad.set_size(P);
  ws.step.set_size(P);
  ws.new_beta.set_size(P);

  double *eta = ws.eta.memptr();
  double *new_eta = ws.new_eta.memptr();

  ig_matvec(X, beta.memptr(), eta);
  double min_eta;
  double cur_dev = ig_dev(y, eta, n, min_eta);

  converged = false;
  int it = 0;
  for (; it < kMaxIter; ++it) {
    double *__restrict w = ws.w.memptr();
    double *__restrict g = ws.mu.memptr();

#pragma omp simd
    for (arma::uword i = 0; i < n; ++i) {
      const double ei = eta[i] > kEtaFloor ? eta[i] : kEtaFloor;
      const double mu = 1.0 / std::sqrt(ei);
      double wi = mu * mu * mu;
      wi = wi < 1e-8 ? 1e-8 : (wi > 1e8 ? 1e8 : wi);
      w[i] = wi;
      g[i] = -2.0 * (y[i] - mu);
    }

    for (arma::uword a = 0; a < P; ++a) {
      const double *__restrict xa = X.colptr(a);
      double gs = 0.0;
#pragma omp simd reduction(+ : gs)
      for (arma::uword i = 0; i < n; ++i)
        gs += g[i] * xa[i];
      ws.grad[a] = gs;
      for (arma::uword b = 0; b <= a; ++b) {
        const double *__restrict xb = X.colptr(b);
        double s = 0.0;
#pragma omp simd reduction(+ : s)
        for (arma::uword i = 0; i < n; ++i)
          s += w[i] * xa[i] * xb[i];
        ws.XTWX(a, b) = s;
        ws.XTWX(b, a) = s;
      }
    }

    bool ok = arma::solve(ws.step, ws.XTWX, ws.grad,
                          arma::solve_opts::fast + arma::solve_opts::no_approx);
    if (!ok)
      ok = arma::solve(ws.step, ws.XTWX, ws.grad, arma::solve_opts::no_approx);
    if (!ok)
      ws.step = arma::pinv(ws.XTWX) * ws.grad;

    for (arma::uword a = 0; a < P; ++a)
      ws.new_beta[a] = beta[a] + ws.step[a];
    ig_matvec(X, ws.new_beta.memptr(), new_eta);
    double new_dev = ig_dev(y, new_eta, n, min_eta);

    // Step halving: keep eta > 0 (as before) and do not let the deviance rise.
    int halves = 0;
    while ((min_eta <= 0.0 ||
            !(new_dev <= cur_dev + kDevSlack * (1.0 + std::abs(cur_dev)))) &&
           halves < kMaxHalve) {
      ws.step *= 0.5;
      for (arma::uword a = 0; a < P; ++a)
        ws.new_beta[a] = beta[a] + ws.step[a];
      for (arma::uword i = 0; i < n; ++i) // eta is linear in beta
        new_eta[i] = 0.5 * (eta[i] + new_eta[i]);
      new_dev = ig_dev(y, new_eta, n, min_eta);
      ++halves;
    }

    const double step_norm = arma::norm(ws.step);
    beta = ws.new_beta;
    ws.eta.swap(ws.new_eta);
    eta = ws.eta.memptr();
    new_eta = ws.new_eta.memptr();
    cur_dev = new_dev;

    if (step_norm < kStepTol) {
      converged = true;
      ++it;
      break;
    }
  }
  iters = it;
  dev_out = cur_dev;
}

} // namespace

// R does not have a native rinvgauss. Implemented is the Michael,
// Schucany, and Haas (1976) algorithm to simulate it via C++.
// Kept (same signature) for any other callers; the hot loop below uses
// rinvgauss_fast with generators that persist across draws.
double rinvgauss_single(double mu, double lambda, std::mt19937 &gen) {
  std::normal_distribution<double> rnorm(0.0, 1.0);
  std::uniform_real_distribution<double> runif(0.0, 1.0);
  return rinvgauss_fast(mu, lambda, rnorm, runif, gen);
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

// [[Rcpp::export]]
double compute_invgauss_ll(const arma::vec &y, const arma::vec &mu,
                           double gamma_val) {
  return invgauss_ll_raw(y, mu, gamma_val);
}

// IRLS Inverse Gaussian regression solver (1/mu^2 link), workspace version.
arma::vec fit_invgauss_cpp(const arma::mat &X, const arma::vec &y,
                           const arma::vec &initial_beta, bool approx,
                           InvGaussWorkspace &ws, bool inner_parallel) {
  (void)approx;
  (void)inner_parallel;
  arma::vec beta = initial_beta;
  double dev;
  int iters;
  bool converged;
  fit_invgauss_core(X, y.memptr(), beta, ws, dev, iters, converged);
  return beta;
}

// [[Rcpp::export]]
arma::vec fit_invgauss_cpp(const arma::mat &X, const arma::vec &y,
                           const arma::vec &initial_beta, bool approx) {
  InvGaussWorkspace ws(X.n_rows, X.n_cols);
  return fit_invgauss_cpp(X, y, initial_beta, approx, ws, false);
}

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
//    Y_sim_workspace is no longer used: each replicate is simulated into a
//    small per-thread vector right before it is fitted (stays in cache, no
//    n x m matrix). The parameter is kept so callers need not change.
double glm_invgauss_pl_cpp(const arma::mat &X, const arma::vec &y,
                           const arma::vec &mle_coefs, double mle_val,
                           const arma::vec &beta_vals, int m, bool approx,
                           uint32_t base_seed, int eval_index,
                           const arma::vec &mu_hat, arma::mat &Y_sim_workspace,
                           InvGaussWorkspace &ws, bool inner_parallel) {
  (void)mle_coefs;
  (void)mle_val;
  (void)Y_sim_workspace;
  (void)approx;
  const arma::uword n = X.n_rows;
  const arma::uword P = X.n_cols;
  const double *yp = y.memptr();

  // ---- per-evaluation constants (fixed beta): computed once, reused for all m
  arma::vec eta0 = X * beta_vals;
  const double *e0 = eta0.memptr();
  double min_e0 = e0[0];
  for (arma::uword i = 1; i < n; ++i)
    min_e0 = e0[i] < min_e0 ? e0[i] : min_e0;
  // eta must be positive for mu = eta^(-1/2) to exist; such a beta has no
  // likelihood (the old code produced NaNs here and counted 0).
  if (!(min_e0 > 0.0))
    return 0.0;

  arma::vec mu0(n);
  double *mu0p = mu0.memptr();
  double S0 = 0.0, ye0 = 0.0, sbar_sum = 0.0, pearson = 0.0;
  for (arma::uword i = 0; i < n; ++i) {
    const double s = std::sqrt(e0[i]); // 1/mu
    const double r = yp[i] - 1.0 / s;
    mu0p[i] = 1.0 / s;
    S0 += s;
    ye0 += yp[i] * e0[i];
    sbar_sum += yp[i] * s - 1.0;
    pearson += r * r * e0[i] * s; // (y-mu)^2 / mu^3
  }
  // Fletcher-corrected Pearson dispersion (unchanged)
  const double sbar = 2.0 * sbar_sum / static_cast<double>(n);
  const double est_disp =
      pearson /
      ((static_cast<double>(n) - static_cast<double>(P)) * (1.0 + sbar));
  const double gamma = 1.0 / est_disp;
  if (!std::isfinite(gamma) || !(gamma > 0.0))
    return 0.0; // invalid dispersion: the old code produced NaNs and counted 0

  // Observed deviance difference D_obs = Dev(eta_beta) - Dev(eta_hat)
  double S_hat = 0.0, ye_hat = 0.0;
  for (arma::uword i = 0; i < n; ++i) {
    const double s = 1.0 / mu_hat[i];
    S_hat += s;
    ye_hat += yp[i] * s * s;
  }
  const double D_obs = (ye0 - ye_hat) - 2.0 * (S0 - S_hat);

  bool is_already_parallel = false;
#ifdef _OPENMP
  is_already_parallel = omp_in_parallel();
#endif
  const bool run_inner_parallel = inner_parallel && !is_already_parallel;
  const uint32_t eval_seed =
      base_seed + static_cast<uint32_t>(eval_index * 10007);

  const bool prof = glim_prof::enabled.load(std::memory_order_relaxed); // PROF

  int count_less = 0;
  uint64_t sim_ns = 0, fit_ns = 0, ll_ns = 0, iter_sum = 0, n_nonconv = 0;

#pragma omp parallel if (run_inner_parallel)                                   \
    reduction(+ : count_less, sim_ns, fit_ns, ll_ns, iter_sum, n_nonconv)
  {
    // Threads get private scratch; the serial path reuses ws (no allocation).
    InvGaussWorkspace local_ws;
    if (run_inner_parallel)
      local_ws = InvGaussWorkspace(n, P);
    InvGaussWorkspace &fit_ws = run_inner_parallel ? local_ws : ws;

    arma::vec y_sim(n);
    arma::vec beta_hat(P);
    double *ys = y_sim.memptr();
    std::mt19937 gen;
    std::normal_distribution<double> rnorm(0.0, 1.0);
    std::uniform_real_distribution<double> runif(0.0, 1.0);

#pragma omp for schedule(guided)
    for (int j = 0; j < m; ++j) {
      const uint64_t t0 = prof ? glim_prof::now_ns() : 0; // PROF

      // ---- simulate replicate j (deterministic in (eval_seed, j)) ----
      gen.seed(eval_seed + j);
      rnorm.reset(); // drop the cached normal so replicate j depends only on
                     // its seed
      for (arma::uword i = 0; i < n; ++i)
        ys[i] = rinvgauss_fast(mu0p[i], gamma, rnorm, runif, gen);

      const uint64_t t1 = prof ? glim_prof::now_ns() : 0; // PROF

      // ---- refit; also returns Dev(eta_hat_sim) (minus the sum(1/y) term)
      // ----
      beta_hat = beta_vals;
      double dev_hat;
      int iters;
      bool converged;
      fit_invgauss_core(X, ys, beta_hat, fit_ws, dev_hat, iters, converged);

      const uint64_t t2 = prof ? glim_prof::now_ns() : 0; // PROF

      // D_j = Dev(eta_beta) - Dev(eta_hat_sim) >= 0 by construction.
      double ye = 0.0;
#pragma omp simd reduction(+ : ye)
      for (arma::uword i = 0; i < n; ++i)
        ye += ys[i] * e0[i];
      double D_j = (ye - 2.0 * S0) - dev_hat;
      if (D_j < 0.0)
        D_j = 0.0;
      if (D_j >= D_obs)
        count_less++;

      if (prof) { // PROF
        const uint64_t t3 = glim_prof::now_ns();
        sim_ns += (t1 - t0);
        fit_ns += (t2 - t1);
        ll_ns += (t3 - t2);
        iter_sum += static_cast<uint64_t>(iters);
        if (!converged)
          n_nonconv++;
      }
    }
  }
  if (prof) { // PROF
    glim_prof::ns_sim.fetch_add(sim_ns, std::memory_order_relaxed);
    glim_prof::ns_fit.fetch_add(fit_ns, std::memory_order_relaxed);
    glim_prof::ns_ll.fetch_add(ll_ns, std::memory_order_relaxed);
    glim_prof::newton_iters.fetch_add(iter_sum, std::memory_order_relaxed);
    glim_prof::nonconverged.fetch_add(n_nonconv, std::memory_order_relaxed);
    glim_prof::n_fits.fetch_add(m, std::memory_order_relaxed);
  }

  return static_cast<double>(count_less) / m;
}

// 2. Convenience Wrapper (Calculates MLE vectors, requires workspace)
double glm_invgauss_pl_cpp(const arma::mat &X, const arma::vec &y,
                           const arma::vec &mle_coefs, const double mle_val,
                           const arma::vec &beta_vals, int m, bool approx,
                           uint32_t base_seed, int eval_index,
                           arma::mat &Y_sim_workspace, InvGaussWorkspace &ws,
                           bool inner_parallel) {
  // Precompute mu_hat once per evaluation
  arma::vec eta_hat = X * mle_coefs;
  arma::vec mu_hat = arma::pow(eta_hat, -0.5);

  return glm_invgauss_pl_cpp(X, y, mle_coefs, mle_val, beta_vals, m, approx,
                             base_seed, eval_index, mu_hat, Y_sim_workspace, ws,
                             inner_parallel);
}