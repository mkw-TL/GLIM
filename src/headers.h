#ifndef HEADERS_H
#define HEADERS_H

#include <RcppArmadillo.h>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/polygamma.hpp>
#include <cmath>
#include <omp.h>

struct LogisticResult {
  arma::vec beta;
  bool seperated;
};
struct LogisticPlResult {
  double poss;
  bool orig_seperated;
  double sim_separated;
};
struct LogisticResult1D {
  double beta;
  bool orig_seperated;
};

// A forward declaration so that other cpp files know that this exists.
double glm_poisson_pl_cpp(const arma::mat &X, const arma::vec &y,
                          const arma::vec &mle_coefs,
                          const arma::vec &beta_vals, int m, bool approx);

double glm_gamma_pl_cpp(arma::mat &X, const arma::mat &XtX, const arma::vec &y,
                        const arma::vec &mle_coefs, const arma::vec &beta_vals,
                        int m, bool approx);

LogisticPlResult glm_logis_pl_cpp(const arma::mat &X, const arma::vec &y,
                                  const arma::vec &mle_coefs,
                                  const arma::vec &beta_vals, int m,
                                  bool approx);

double glm_gaussian_pl_cpp(const arma::mat &X, const arma::vec &y,
                           const arma::vec &mle_coefs,
                           const arma::vec &beta_vals, int m, bool approx);

double glm_invgauss_pl_cpp(const arma::mat &X, const arma::vec &y,
                           const arma::vec &mle_coefs,
                           const arma::vec &beta_vals, int m, bool approx);

#endif