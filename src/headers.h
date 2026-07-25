#ifndef HEADERS_H
#define HEADERS_H

#ifdef _OPENMP
#include <omp.h>
#endif

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
  bool orig_separated;
  double sim_separated;
};
struct LogisticResult1D {
  arma::vec beta;
  bool orig_separated;
};
struct PoissonResult {
  arma::vec beta;
  bool success;
  bool separated;
};
struct PoissonPlResult {
  double poss;
  bool orig_separated;
  double prop_separated;
};
struct PoissonPlResult1D {
  double beta;
  bool orig_separated;
};

// A forward declaration so that other cpp files know that this exists.
PoissonPlResult glm_poisson_pl_cpp(const arma::mat &X, const arma::vec &y,
                                   const arma::vec &mle_coefs,
                                   const arma::vec &beta_vals, int m,
                                   bool approx, bool radial, uint32_t base_seed,
                                   int eval_index);

double glm_gamma_pl_cpp(arma::mat &X, const arma::mat &XtX, const arma::vec &y,
                        const arma::vec &mle_coefs, const arma::vec &beta_vals,
                        int m, bool approx, bool radial, uint32_t base_seed,
                        int eval_index);

LogisticPlResult glm_logis_pl_cpp(const arma::mat &X, const arma::vec &y,
                                  const arma::vec &mle_coefs,
                                  const arma::vec &beta_vals, int m,
                                  bool approx, bool radial, uint32_t base_seed,
                                  int eval_index);

double glm_gaussian_pl_cpp(const arma::mat &X, const arma::vec &y,
                           const arma::vec &mle_coefs,
                           const arma::vec &beta_vals, int m, bool approx,
                           bool radial, uint32_t base_seed, int eval_index);

double glm_invgauss_pl_cpp(const arma::mat &X, const arma::vec &y,
                           const arma::vec &mle_coefs,
                           const arma::vec &beta_vals, int m, bool approx,
                           bool radial, uint32_t base_seed, int eval_index);

#endif