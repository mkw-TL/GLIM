#include "glim_prof.h"
#include "headers.h"
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace arma;

// ----------------------------------------------------------------------------
// Key identities used below (gamma, log link, eta = X b, mu = exp(eta)):
//
//   dev/2 = sum(y * exp(-eta)) + sum(eta) - sum(log y) - n
//   ll(k) = n (k ln k - lgamma k) + (k-1) sum(log y) - k [sum(y exp(-eta)) +
//   sum(eta)]
//
// so once sum(log y) is known for a replicate, a solver iteration needs ONE exp
// pass (no log), and the loglik at the fitted point needs only the final
// deviance:  ll_hat = n (k ln k - lgamma k) - sum(log y) - k dev/2 - k n.
// Caveat: the deviance written this way subtracts O(n)-sized sums, so it loses
// precision if the dispersion is astronomically small (< ~1e-8). Not a concern
// for realistic data.
// ----------------------------------------------------------------------------

double calculate_deviance_gamma(const arma::vec &y, const arma::vec &mu) {
  return 2.0 * arma::accu(y / mu - arma::log(y / mu) - 1.0);
}

// Solver step tolerance. The log-likelihood deficit from stopping at distance
// delta from the optimum is ~ 0.5 * k * n * delta^2, which for delta = 1e-5 is
// far below the O(1) scale of the llX_j - mle_sim comparisons. (Was 1e-6.)
static constexpr double kStepTol = 1e-5;

// Fast fit. `slogy` = sum(log(y)) (computed once per replicate by the caller).
// On return `dev_out` is the gamma deviance at the returned beta.
// ws.mu / ws.mu_try are used as buffers for 1/mu (inverse mean), NOT mu.
// ws.eta is left untouched (the caller's serial path relies on it).
arma::vec fit_gamma_log_cpp(const arma::mat &X, const arma::mat &XtX,
                            const arma::vec &y, const arma::vec &initial_beta,
                            bool approx, GammaWorkspace &ws, double slogy,
                            double &dev_out) {
  const arma::uword n = X.n_rows;
  const double dn = static_cast<double>(n);
  arma::vec beta = initial_beta;
  arma::vec eta(n); // clamped linear predictor at the trial point (one alloc)

  // Evaluates the deviance at b and fills inv_mu = exp(-eta). One exp pass.
  auto eval_point = [&](const arma::vec &b, arma::vec &inv_mu) -> double {
    eta = X * b;
    eta.clamp(-30.0, 30.0);
    inv_mu = arma::exp(-eta);
    return 2.0 * (arma::dot(y, inv_mu) + arma::accu(eta) - slogy - dn);
  };

  double current_dev = eval_point(beta, ws.mu);
  int it_used = 0;
  bool converged = false;

  for (int it = 0; it < 20; it++) {
    it_used = it + 1;
    // grad = X'(y/mu - 1); ws.mu holds 1/mu at the current beta.
    ws.grad = X.t() * (y % ws.mu - 1.0);

    bool success =
        arma::solve(ws.step, XtX, ws.grad,
                    arma::solve_opts::fast + arma::solve_opts::no_approx);
    if (!success)
      success = arma::solve(ws.step, XtX, ws.grad, arma::solve_opts::no_approx);
    if (!success) {
      ws.step = arma::pinv(XtX) * ws.grad;
    }

    ws.temp_beta = beta + ws.step;
    double proposed_dev = eval_point(ws.temp_beta, ws.mu_try);

    int half_iter = 0;
    while ((!std::isfinite(proposed_dev) || proposed_dev > current_dev) &&
           half_iter < 25) {
      ws.step *= 0.5;
      ws.temp_beta = beta + ws.step;
      proposed_dev = eval_point(ws.temp_beta, ws.mu_try);
      half_iter++;
    }

    beta = ws.temp_beta;
    ws.mu.swap(ws.mu_try); // accepted 1/mu becomes current, no copy
    current_dev = proposed_dev;

    if (arma::norm(ws.step) < kStepTol) {
      converged = true;
      break;
    }
  }

  if (glim_prof::enabled.load(std::memory_order_relaxed)) { // PROF
    glim_prof::newton_iters.fetch_add(it_used, std::memory_order_relaxed);
    if (!converged)
      glim_prof::nonconverged.fetch_add(1, std::memory_order_relaxed);
  }

  dev_out = current_dev;
  return beta;
}

// Backward-compatible overload (same signature as before).
arma::vec fit_gamma_log_cpp(const arma::mat &X, const arma::mat &XtX,
                            const arma::vec &y, const arma::vec &initial_beta,
                            bool approx, GammaWorkspace &ws) {
  double dev;
  return fit_gamma_log_cpp(X, XtX, y, initial_beta, approx, ws,
                           arma::accu(arma::log(y)), dev);
}

// [[Rcpp::export]]
arma::vec fit_gamma_log_r(const arma::mat &X, const arma::vec &y,
                          const arma::vec &initial_beta) {
  GammaWorkspace ws;
  arma::mat XtX = X.t() * X;
  return fit_gamma_log_cpp(X, XtX, y, initial_beta, false, ws);
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
double pearson_estimate_dispersion_gamma(const arma::vec &y,
                                         const arma::vec &mu_hat, double p) {
  return (double)arma::accu(arma::square(y - mu_hat) / arma::square(mu_hat)) /
         ((y.n_elem - p));
}

// Scalar part of the shape MLE: solves log(nu) - digamma(nu) = D_mean, where
// D_mean = mean(y/mu - log(y/mu) - 1) = dev / (2 n). O(1) work.
// Edge cases now return a SHAPE (the old code returned the opposite end for
// both caps, i.e. a huge dispersion for a perfect fit and a tiny one for a
// near-zero shape). Shape is capped to [1e-5, 1e5].
static double gamma_shape_from_dmean(double D_mean) {
  // Edge case: perfect fit -> infinite shape; cap it.
  if (D_mean <= 1e-10) {
    return 1e5;
  }

  // Thom-type initial guess (Laurent-series approximation to digamma)
  double nu = (1.0 + std::sqrt(1.0 + (4.0 / 3.0) * D_mean)) / (4.0 * D_mean);

  const int max_iter = 100;
  const double tol = 1e-8;
  for (int i = 0; i < max_iter; i++) {
    double f = std::log(nu) - boost::math::digamma(nu) - D_mean;
    double f_prime = (1.0 / nu) - boost::math::polygamma(1, nu);
    double next_nu = nu - f / f_prime;
    if (next_nu <= 0) {
      next_nu = nu * 0.5;
    }
    if (std::abs(next_nu - nu) < tol) {
      nu = next_nu;
      break;
    }
    nu = next_nu;
  }
  return std::min(std::max(nu, 1e-5), 1e5);
}

// [[Rcpp::export]]
double mle_estimate_dispersion_gamma(const arma::vec &y,
                                     const arma::vec &mu_hat, double p) {
  // Safety net against zero or negative values disrupting logs
  arma::vec ratio = y / mu_hat;
  ratio.elem(arma::find(ratio < 1e-8)).fill(1e-8);
  const double D_mean = arma::mean(ratio - arma::log(ratio) - 1.0);
  return 1.0 / gamma_shape_from_dmean(D_mean);
}

static inline double gamma_ll_base(double n, double k) {
  return n * (k * std::log(k) - std::lgamma(k));
}

// 1. Core Engine (Accepts precomputed eta_hat AND workspace)
//    eta_hat is the LINEAR PREDICTOR at the MLE (not exp of it).
//    Y_sim_workspace is no longer used: each replicate is simulated into a
//    small per-thread vector right before it is fitted (stays in cache, no
//    n x m matrix). The parameter is kept so callers need not change.
double glm_gamma_pl_cpp(const arma::mat &X, const arma::mat &XtX,
                        const arma::vec &y, const arma::vec &mle_coefs,
                        double mle_val, const arma::vec &beta_vals, int m,
                        bool approx, uint32_t base_seed, int eval_index,
                        const arma::vec &eta_hat, arma::mat &Y_sim_workspace,
                        GammaWorkspace &ws, bool inner_parallel) {
  (void)Y_sim_workspace;
  const int n = X.n_rows;
  const int P = X.n_cols;
  const double dn = static_cast<double>(n);

  ws.eta = arma::clamp(X * beta_vals, -50.0, 50.0); // avoids D_mean explosions
  ws.mu = arma::exp(ws.eta);
  ws.mu.elem(arma::find(ws.mu < 1e-8)).fill(1e-8);

  const double dispersion =
      mle_estimate_dispersion_gamma(y, ws.mu, beta_vals.n_elem);
  const double shape = 1.0 / dispersion;
  const double true_ll = compute_gamma_ll(y, ws.eta, shape);
  const double mle_ll = compute_gamma_ll(y, eta_hat, shape);
  const double f_x = true_ll - mle_ll;

  // Per-evaluation constants (fixed beta): computed once, reused for all m
  // replicates. ws.eta is not modified by the fit, so these stay valid.
  const arma::vec inv_mu0 = arma::exp(-ws.eta);
  const double sum_eta0 = arma::accu(ws.eta);
  const arma::vec scale = ws.mu * dispersion;

  bool is_already_parallel = false;
#ifdef _OPENMP
  is_already_parallel = omp_in_parallel();
#endif
  const bool run_inner_parallel = inner_parallel && !is_already_parallel;
  const uint32_t eval_seed =
      base_seed + static_cast<uint32_t>(eval_index * 10007);

  const bool prof = glim_prof::enabled.load(std::memory_order_relaxed); // PROF

  int count_less = 0;
  uint64_t sim_ns = 0, fit_ns = 0, ll_ns = 0; // PROF

#pragma omp parallel if (run_inner_parallel)                                   \
    reduction(+ : count_less, sim_ns, fit_ns, ll_ns)
  {
    // Threads get private scratch; the serial path reuses ws (no allocation).
    GammaWorkspace local_ws;
    if (run_inner_parallel)
      local_ws = GammaWorkspace(n, P);
    GammaWorkspace &fit_ws = run_inner_parallel ? local_ws : ws;

    arma::vec y_sim(n);
    std::mt19937 gen;
    std::gamma_distribution<double> rgamma(shape, 1.0); // built ONCE
    const double *sc = scale.memptr();

#pragma omp for schedule(guided)
    for (int j = 0; j < m; ++j) {
      const uint64_t t0 = prof ? glim_prof::now_ns() : 0; // PROF

      // ---- simulate replicate j (deterministic in (eval_seed, j)) ----
      gen.seed(eval_seed + j);
      rgamma.reset(); // drop cached normal so replicate j depends only on its
                      // seed
      double *ys = y_sim.memptr();
      for (int i = 0; i < n; i++) {
        const double v = rgamma(gen) * sc[i]; // Gamma(shape, scale_i)
        ys[i] = (v < 1e-10) ? 1e-10 : v;
      }

      const uint64_t t1 = prof ? glim_prof::now_ns() : 0; // PROF
      const double slogy = arma::accu(arma::log(y_sim));  // once per replicate
      const uint64_t t2 = prof ? glim_prof::now_ns() : 0; // PROF

      // ---- refit; also returns the deviance at the fitted beta ----
      double dev;
      arma::vec beta_sim_hat = fit_gamma_log_cpp(X, XtX, y_sim, beta_vals,
                                                 approx, fit_ws, slogy, dev);
      const uint64_t t3 = prof ? glim_prof::now_ns() : 0; // PROF

      // ---- shape MLE and the two log-likelihoods from sufficient sums ----
      const double k = gamma_shape_from_dmean(dev / (2.0 * dn));
      const double base = gamma_ll_base(dn, k);
      const double llX_j =
          base + (k - 1.0) * slogy - k * (arma::dot(y_sim, inv_mu0) + sum_eta0);
      const double mle_sim = base - slogy - k * (0.5 * dev + dn);
      if (llX_j - mle_sim < f_x)
        count_less++;

      if (prof) { // PROF
        const uint64_t t4 = glim_prof::now_ns();
        sim_ns += (t1 - t0);
        ll_ns += (t2 - t1) + (t4 - t3);
        fit_ns += (t3 - t2);
      }
    }
  }
  if (prof) { // PROF
    glim_prof::ns_sim.fetch_add(sim_ns, std::memory_order_relaxed);
    glim_prof::ns_fit.fetch_add(fit_ns, std::memory_order_relaxed);
    glim_prof::ns_ll.fetch_add(ll_ns, std::memory_order_relaxed);
    glim_prof::n_fits.fetch_add(m, std::memory_order_relaxed);
  }

  return static_cast<double>(count_less) / m;
}

// 2. Convenience Wrapper (Calculates MLE vectors, requires workspace)
double glm_gamma_pl_cpp(const arma::mat &X, const arma::mat &XtX,
                        const arma::vec &y, const arma::vec &mle_coefs,
                        const double mle_val, const arma::vec &beta_vals, int m,
                        bool approx, uint32_t base_seed, int eval_index,
                        arma::mat &Y_sim_workspace, GammaWorkspace &ws,
                        bool inner_parallel) {
  arma::vec eta_hat = arma::clamp(X * mle_coefs, -50.0, 50.0);
  return glm_gamma_pl_cpp(X, XtX, y, mle_coefs, mle_val, beta_vals, m, approx,
                          base_seed, eval_index, eta_hat, Y_sim_workspace, ws,
                          inner_parallel);
}