#ifndef HEADERS_H
#define HEADERS_H

#include <RcppArmadillo.h>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/polygamma.hpp>
#include <cmath>
#include <progress.hpp>
#include <progress_bar.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

struct LogisticResult {
  arma::mat beta;
  bool seperated;
};
struct LogisticPlResult {
  double poss;
  bool orig_seperated;
  double sim_seperated;
};
struct LogisticResult1D {
  arma::vec beta;
  bool orig_seperated;
};
struct PoissonResult {
  arma::vec beta;
  bool success;
  bool seperated;
};
struct PoissonPlResult {
  double poss;
  bool orig_seperated;
  double prop_seperated;
};
struct PoissonPlResult1D {
  double beta;
  bool orig_seperated;
};

// A forward declaration so that other cpp files know that this exists.
PoissonPlResult glm_poisson_pl_cpp(const arma::mat &X, const arma::vec &y,
                                   const arma::vec &mle_coefs,
                                   const arma::vec &beta_vals, int m,
                                   bool approx, bool appendix);

double glm_gamma_pl_cpp(arma::mat &X, const arma::mat &XtX, const arma::vec &y,
                        const arma::vec &mle_coefs, const arma::vec &beta_vals,
                        int m, bool approx, bool appendix);

LogisticPlResult glm_logis_pl_cpp(const arma::mat &X, const arma::vec &y,
                                  const arma::vec &mle_coefs,
                                  const arma::vec &beta_vals, int m,
                                  bool approx, bool appendix);

double glm_gaussian_pl_cpp(const arma::mat &X, const arma::vec &y,
                           const arma::vec &mle_coefs,
                           const arma::vec &beta_vals, int m, bool approx,
                           bool appendix);

double glm_invgauss_pl_cpp(const arma::mat &X, const arma::vec &y,
                           const arma::vec &mle_coefs,
                           const arma::vec &beta_vals, int m, bool approx,
                           bool appendix);

#endif