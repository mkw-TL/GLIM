#include "glim_prof.h"
#include "headers.h"
#include <cstring>
#include <random>
#include <vector>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace arma;

// =====================================================================
// POISSON REGRESSION (LOG LINK)
//
// Where the old code spent its time, per replicate (n x P design, m replicates
// per grid point):
//   * simulation: std::poisson_distribution constructed for EVERY observation
//     (its constructor does exp / and for mean >= 12 also lgamma, sqrt, log);
//   * IRLS: every iteration copied X into ws.XW (n x P), scaled it, then called
//     BLAS twice (X'WX, X'(y-mu)), plus ws.eta = X*beta, clamp, exp as separate
//     Armadillo passes (each of which spawns OpenMP teams when called outside a
//     parallel region);
//   * after the fit: clamp(X*beta), dot, exp and accu as four more passes.
// Below: a table-free inversion sampler, one fused pass per IRLS iteration
// (mu, log-likelihood terms), column-wise X'WX / gradient, the starting
// eta / mu taken from the per-evaluation values instead of recomputed (saves
// one exp pass per replicate), and the final log-likelihood read off the last
// iteration's pass. The statistical procedure is unchanged.
// =====================================================================

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

namespace {

constexpr double kPoisInvMax = 12.0;  // inversion below, std:: sampler at/above
constexpr int kPoisMaxIter = 12;      // as before
constexpr double kPoisStepTol = 1e-3; // as before

// Poisson by sequential inversion with a 53-bit uniform. emu = exp(-mu) is
// supplied (computed once per grid point). Expected work ~ mu + 1 steps, so it
// is only used for mu < kPoisInvMax. The k cap guards against the cumulative
// sum saturating just below u in floating point.
inline double rpois_inv(double mu, double emu, std::mt19937 &gen) {
  const uint32_t a = gen() >> 5;
  const uint32_t b = gen() >> 6;
  const double u = (a * 67108864.0 + b) * (1.0 / 9007199254740992.0); // [0, 1)
  double p = emu, c = p;
  int k = 0;
  while (u > c && k < 128) {
    ++k;
    p *= mu / k;
    c += p;
  }
  return static_cast<double>(k);
}

struct PoisFitInfo {
  double ll; // dot(y, eta_c) - sum(exp(eta_c)), eta_c = clamp(X beta_hat)
  bool separated;
  bool converged;
  int iters; // Newton steps applied
};

// Newton / IRLS (canonical link). `beta` is the start on entry and the fit on
// return. eta_start = X*beta (unclamped) and mu_start = exp(clamp(eta_start))
// at the start point are supplied by the caller (they are the same for every
// replicate of a grid point). Same update, step cap and tolerance as before.
PoisFitInfo fit_poisson_core(const arma::mat &X, const double *__restrict y,
                             arma::vec &beta, const double *eta_start,
                             const double *mu_start, PoissonWorkspace &ws) {
  const arma::uword n = X.n_rows, P = X.n_cols;
  ws.eta.set_size(n);
  ws.mu.set_size(n);
  ws.XTWX.set_size(P, P);
  ws.grad.set_size(P);
  ws.step.set_size(P);

  double *__restrict eta = ws.eta.memptr(); // unclamped X*beta, kept current
  double *__restrict mu = ws.mu.memptr();   // exp(clamp(eta))
  std::memcpy(eta, eta_start, n * sizeof(double));
  std::memcpy(mu, mu_start, n * sizeof(double));

  bool solver_ok = true, converged = false, have_ll = false;
  double ll = 0.0;
  int applied = 0;

  for (int it = 0; it < kPoisMaxIter; ++it) {
    // gradient X'(y - mu) and X'WX with W = diag(mu), column-wise
    for (arma::uword a = 0; a < P; ++a) {
      const double *__restrict xa = X.colptr(a);
      double gs = 0.0;
#pragma omp simd reduction(+ : gs)
      for (arma::uword i = 0; i < n; ++i)
        gs += xa[i] * (y[i] - mu[i]);
      ws.grad[a] = gs;
      for (arma::uword b = 0; b <= a; ++b) {
        const double *__restrict xb = X.colptr(b);
        double s = 0.0;
#pragma omp simd reduction(+ : s)
        for (arma::uword i = 0; i < n; ++i)
          s += mu[i] * xa[i] * xb[i];
        ws.XTWX(a, b) = s;
        ws.XTWX(b, a) = s;
      }
    }

    bool ok = arma::solve(ws.step, ws.XTWX, ws.grad,
                          arma::solve_opts::no_approx +
                              arma::solve_opts::likely_sympd);
    if (!ok) {
      ws.step = arma::pinv(ws.XTWX) * ws.grad;
      ok = ws.step.is_finite();
    }
    if (!ok || !ws.step.is_finite()) {
      solver_ok = false;
      break;
    }

    beta += ws.step;
    ++applied;

    // eta <- eta + X*step  (linear: no second full X*beta)
    for (arma::uword a = 0; a < P; ++a) {
      const double *__restrict xa = X.colptr(a);
      const double sa = ws.step[a];
#pragma omp simd
      for (arma::uword i = 0; i < n; ++i)
        eta[i] += xa[i] * sa;
    }

    // One fused pass: mu at the new beta (needed by the next iteration) AND the
    // log-likelihood terms at the new beta (needed if this is the last one).
    double sm = 0.0, dy = 0.0;
#pragma omp simd reduction(+ : sm, dy)
    for (arma::uword i = 0; i < n; ++i) {
      const double e = eta[i];
      const double ec = e < -10.0 ? -10.0 : (e > 10.0 ? 10.0 : e);
      const double mi = std::exp(ec);
      mu[i] = mi;
      sm += mi;
      dy += y[i] * ec;
    }
    ll = dy - sm;
    have_ll = true;

    if (arma::norm(ws.step) < kPoisStepTol) {
      converged = true;
      break;
    }
  }

  if (!have_ll) { // solver failed on the first iteration: ll at the start point
    double sm = 0.0, dy = 0.0;
    for (arma::uword i = 0; i < n; ++i) {
      const double e = eta[i];
      const double ec = e < -10.0 ? -10.0 : (e > 10.0 ? 10.0 : e);
      sm += mu[i];
      dy += y[i] * ec;
    }
    ll = dy - sm;
  }

  // (Evaluated at the final beta; the old code used the eta from the start of
  // the last iteration. This flag is informational only.)
  bool separated = false;
  if (!converged && solver_ok) {
    for (arma::uword k = 0; k < n; ++k) {
      if (y[k] == 0 && eta[k] <= -10.0) {
        separated = true;
        break;
      }
    }
  }
  return {ll, separated, converged, applied};
}

} // namespace

PoissonResult fit_poisson_inner(const arma::mat &X, const arma::vec &y,
                                const arma::vec &initial_beta,
                                PoissonWorkspace &ws) {
  arma::vec beta = initial_beta;
  const arma::vec eta0 = X * beta;
  const arma::vec mu0 = arma::exp(arma::clamp(eta0, -10.0, 10.0));
  PoisFitInfo info =
      fit_poisson_core(X, y.memptr(), beta, eta0.memptr(), mu0.memptr(), ws);
  return {beta, info.separated};
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
                                   bool approx, uint32_t base_seed,
                                   int eval_index, arma::mat &Y_sim_workspace,
                                   PoissonWorkspace &ws, bool inner_parallel) {
  (void)mle_coefs;
  (void)approx;
  (void)Y_sim_workspace; // replicates are simulated into a per-thread vector
  const int n = X.n_rows;
  const int P = X.n_cols;

  // ---- per-evaluation constants (fixed beta): computed once, reused for all m
  // Local, not ws: in the serial path fit_ws aliases ws and the fitter
  // overwrites ws.eta / ws.mu while the j-loop still needs these.
  const arma::vec eta_raw = X * beta_vals; // fit start state
  const arma::vec eta = arma::clamp(eta_raw, -10.0, 10.0);
  const arma::vec mu = arma::exp(eta);

  const double f_x = compute_poisson_ll(eta, y) - mle_val;
  const double sum_mu = arma::accu(mu);

  // Sampler constants: exp(-mu) for the inversion branch; precomputed parameter
  // objects for observations with mu >= kPoisInvMax (rare; shared, read-only).
  const arma::vec emu = arma::exp(-mu);
  std::vector<std::poisson_distribution<int>::param_type> big;
  bool any_big = false;
  for (int i = 0; i < n; ++i)
    if (mu[i] >= kPoisInvMax) {
      any_big = true;
      break;
    }
  if (any_big) {
    big.reserve(n);
    for (int i = 0; i < n; ++i)
      big.emplace_back(mu[i] >= kPoisInvMax ? mu[i] : 1.0);
  }
  const double *mu_p = mu.memptr();
  const double *emu_p = emu.memptr();
  const double *eta_p = eta.memptr();
  const double *eta_raw_p = eta_raw.memptr();
  const std::poisson_distribution<int>::param_type *big_p = big.data();

  bool is_already_parallel = false;
#ifdef _OPENMP
  is_already_parallel = omp_in_parallel();
#endif
  const bool run_inner_parallel = inner_parallel && !is_already_parallel;
  const uint32_t eval_seed =
      base_seed + static_cast<uint32_t>(eval_index * 10007);

  const bool prof = glim_prof::enabled.load(std::memory_order_relaxed); // PROF

  int count_less = 0;
  double prop_separated = 0.0;
  uint64_t sim_ns = 0, fit_ns = 0, ll_ns = 0, iter_sum = 0, n_nonconv = 0;

#pragma omp parallel if (run_inner_parallel)                                   \
    reduction(+ : count_less, prop_separated, sim_ns, fit_ns, ll_ns, iter_sum, \
                  n_nonconv)
  {
    PoissonWorkspace local_ws;
    if (run_inner_parallel)
      local_ws = PoissonWorkspace(n, P);
    PoissonWorkspace &fit_ws = run_inner_parallel ? local_ws : ws;

    arma::vec y_sim(n);
    arma::vec beta_hat(P);
    double *ys = y_sim.memptr();
    std::mt19937 gen;
    std::poisson_distribution<int> pd; // only used when some mu >= kPoisInvMax

#pragma omp for schedule(guided)
    for (int j = 0; j < m; ++j) {
      const uint64_t t0 = prof ? glim_prof::now_ns() : 0; // PROF

      // ---- simulate replicate j (deterministic in (eval_seed, j)) ----
      gen.seed(eval_seed + j);
      pd.reset();
      double dy_eta0 = 0.0; // dot(y_sim, eta): folded into the simulation pass
      for (int i = 0; i < n; ++i) {
        const double v = (mu_p[i] < kPoisInvMax)
                             ? rpois_inv(mu_p[i], emu_p[i], gen)
                             : static_cast<double>(pd(gen, big_p[i]));
        ys[i] = v;
        dy_eta0 += v * eta_p[i];
      }

      const uint64_t t1 = prof ? glim_prof::now_ns() : 0; // PROF

      // ---- refit from beta_vals; log-likelihood at the fit comes back too
      // ----
      beta_hat = beta_vals;
      const PoisFitInfo info =
          fit_poisson_core(X, ys, beta_hat, eta_raw_p, mu_p, fit_ws);

      const uint64_t t2 = prof ? glim_prof::now_ns() : 0; // PROF

      if (info.separated)
        prop_separated += 1.0;

      const double llX_j = dy_eta0 - sum_mu;
      if (llX_j - info.ll <= f_x)
        count_less++;

      if (prof) { // PROF
        const uint64_t t3 = glim_prof::now_ns();
        sim_ns += (t1 - t0);
        fit_ns += (t2 - t1);
        ll_ns += (t3 - t2);
        iter_sum += static_cast<uint64_t>(info.iters);
        if (!info.converged)
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

  return {static_cast<double>(count_less) / m, prop_separated / m};
}

// Wrapper
PoissonPlResult glm_poisson_pl_cpp(const arma::mat &X, const arma::vec &y,
                                   const arma::vec &mle_coefs, double mle_val,
                                   const arma::vec &beta_vals, int m,
                                   bool approx, uint32_t base_seed,
                                   int eval_index, arma::mat &Y_sim_workspace,
                                   bool inner_parallel) {
  PoissonWorkspace ws(X.n_rows, X.n_cols);
  return glm_poisson_pl_cpp(X, y, mle_coefs, mle_val, beta_vals, m, approx,
                            base_seed, eval_index, Y_sim_workspace, ws,
                            inner_parallel);
}