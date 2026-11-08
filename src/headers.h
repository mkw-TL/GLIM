#ifndef HEADERS_H
#define HEADERS_H

#ifdef _OPENMP
#include <omp.h>
#endif

#define ARMA_DONT_PRINT_ERRORS
#define ARMA_NO_DEBUG
#include <RcppArmadillo.h>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/polygamma.hpp>
#include <cmath>
#include <progress.hpp>
#include <progress_bar.hpp>

struct LogisticResult {
  arma::mat beta;
  bool separated;
};
struct LogisticPlResult {
  double poss;
  double sim_separated;
};
struct LogisticResult1D {
  arma::vec beta;
};
struct PoissonResult {
  arma::vec beta;
  bool separated;
};
struct PoissonPlResult {
  double poss;
  double prop_separated;
};
struct PoissonPlResult1D {
  double beta;
};

struct GammaWorkspace {
  arma::vec eta, mu, mu_try, grad, step, temp_beta;
  GammaWorkspace() = default;
  GammaWorkspace(arma::uword n, arma::uword p)
      : eta(n), mu(n), mu_try(n), grad(p), step(p), temp_beta(p) {}
};

struct PoissonWorkspace {
  arma::vec eta, mu, grad, step;
  arma::mat XW, XTWX;
  PoissonWorkspace() = default;
  PoissonWorkspace(arma::uword n, arma::uword p)
      : eta(n), mu(n), grad(p), step(p), XW(n, p), XTWX(p, p) {}
};

struct InvGaussWorkspace {
  arma::vec eta, new_eta, mu, w, grad, step, new_beta;
  arma::mat XW, XTWX;
  InvGaussWorkspace() = default;
  InvGaussWorkspace(arma::uword n, arma::uword p)
      : eta(n), new_eta(n), mu(n), w(n), grad(p), step(p), new_beta(p),
        XW(n, p), XTWX(p, p) {}
};

struct GaussianWorkspace {
  arma::vec beta, mu_hat;
  GaussianWorkspace() = default;
  GaussianWorkspace(arma::uword n, arma::uword p) : beta(p), mu_hat(n) {}
};

// A forward declaration so that other cpp files know that this exists.
PoissonPlResult glm_poisson_pl_cpp(const arma::mat &X, const arma::vec &y,
                                   const arma::vec &mle_coefs, double mle_val,
                                   const arma::vec &beta_vals, int m,
                                   bool approx, bool radial, uint32_t base_seed,
                                   int eval_index, arma::mat &Y_sim_workspace,
                                   PoissonWorkspace &ws);

PoissonPlResult glm_poisson_pl_cpp(const arma::mat &X, const arma::vec &y,
                                   const arma::vec &mle_coefs, double mle_val,
                                   const arma::vec &beta_vals, int m,
                                   bool approx, bool radial,

                                   uint32_t base_seed, int eval_index,
                                   arma::mat &Y_sim_workspace);

double glm_gamma_pl_cpp(const arma::mat &X, const arma::mat &XtX,
                        const arma::vec &y, const arma::vec &mle_coefs,
                        const double mle_val, const arma::vec &beta_vals, int m,
                        bool approx, bool radial, uint32_t base_seed,
                        int eval_index, const arma::vec &eta_hat,
                        arma::mat &Y_sim_workspace, GammaWorkspace &ws);

// All scratch space needed by the logistic IRLS fit and the simulation loop, so
// that nothing is allocated inside the hot loops. `ensure()` is a no-op when
// the buffers already have the right size (Armadillo's set_size early-returns),
// so default-constructed workspaces are safe to pass in.
struct LogisticWorkspace {
  // IRLS scratch (overwritten by every fit)
  arma::vec eta, w, resid, beta, grad, step;
  arma::mat XW, XTWX;
  // Simulated response for the current replicate
  arma::vec y_sim;
  // Linear predictor / probabilities at the proposal beta (read-only during
  // fits)
  arma::vec eta0, p0;

  LogisticWorkspace() = default;
  LogisticWorkspace(arma::uword n, arma::uword P) { ensure(n, P); }

  void ensure(arma::uword n, arma::uword P) {
    eta.set_size(n);
    w.set_size(n);
    resid.set_size(n);
    y_sim.set_size(n);
    eta0.set_size(n);
    p0.set_size(n);
    XW.set_size(n, P);
    XTWX.set_size(P, P);
    beta.set_size(P);
    grad.set_size(P);
    step.set_size(P);
  }
};

inline double sum_softplus(const arma::vec &eta) {
  double s = 0.0;
  for (arma::uword i = 0; i < eta.n_elem; ++i)
    s += (eta(i) > 0) ? (eta(i) + std::log1p(std::exp(-eta(i))))
                      : std::log1p(std::exp(eta(i)));
  return s;
}

// In-place IRLS: ws.beta holds the starting value on entry and the fit on exit.
// Returns true if (quasi-)separation was detected. Allocation-free once the
// workspace is sized.
bool fit_logistic_inplace(const arma::mat &X, const arma::vec &y,
                          LogisticWorkspace &ws);

// Convenience wrapper that returns a copy of the fitted beta.
LogisticResult fit_logistic(const arma::mat &X, const arma::vec &y,
                            const arma::vec &initial_beta,
                            LogisticWorkspace &ws);

// Fills the first m columns of U with U(0,1) draws. Column j depends only on
// (seed, j), so results do not depend on the number of threads.
void fill_uniforms(arma::mat &U, uint64_t seed, int m, bool use_threads);

// Possibility at beta_vals using PRE-DRAWN uniforms U (n x >= m). The simulated
// responses are y_sim = 1{U < p(beta_vals)}, so repeated calls with the same U
// (common random numbers) vary smoothly in beta. This is what the Robbins-Monro
// loop in radial_code should call.
LogisticPlResult glm_logis_pl_from_uniforms(const arma::mat &X,
                                            const arma::vec &y, double mle_val,
                                            const arma::vec &beta_vals,
                                            const arma::mat &U, int m,
                                            bool run_inner_parallel,
                                            LogisticWorkspace &ws);

LogisticPlResult glm_logis_pl_cpp(const arma::mat &X, const arma::vec &y,
                                  const arma::vec &mle_coefs, double mle_val,
                                  const arma::vec &beta_vals, int m,
                                  bool approx, bool radial, uint32_t base_seed,
                                  int eval_index, arma::mat &Y_sim_workspace,
                                  LogisticWorkspace &ws);

double glm_gaussian_pl_cpp(const arma::mat &X, const arma::vec &y,
                           const arma::vec &mle_coefs, double mle_val,
                           const arma::vec &beta_vals, int m, bool approx,
                           bool radial, uint32_t base_seed, int eval_index,
                           double sigma, arma::mat &Y_sim_workspace,
                           GaussianWorkspace &ws);

double glm_gaussian_pl_cpp(const arma::mat &X, const arma::vec &y,
                           const arma::vec &mle_coefs, double mle_val,
                           const arma::vec &beta_vals, int m, bool approx,
                           bool radial, uint32_t base_seed, int eval_index,
                           arma::mat &Y_sim_workspace, GaussianWorkspace &ws);

double glm_invgauss_pl_cpp(const arma::mat &X, const arma::vec &y,
                           const arma::vec &mle_coefs, double mle_val,
                           const arma::vec &beta_vals, int m, bool approx,
                           bool radial, uint32_t base_seed, int eval_index,
                           const arma::vec &mu_hat, arma::mat &Y_sim_workspace,
                           InvGaussWorkspace &ws);

double glm_invgauss_pl_cpp(const arma::mat &X, const arma::vec &y,
                           const arma::vec &mle_coefs, double mle_val,
                           const arma::vec &beta_vals, int m, bool approx,
                           bool radial, uint32_t base_seed, int eval_index,
                           arma::mat &Y_sim_workspace);

#endif