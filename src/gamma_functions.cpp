#include "headers.h"
#include <random>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace arma;

double calculate_deviance_gamma(const arma::vec &y, const arma::vec &mu) {
  return 2.0 * arma::accu(y / mu - arma::log(y / mu) - 1.0);
}

// IRLS Gamma regression solver (Log link), fisher weights, W = 1.
// [[Rcpp::export]]
arma::vec fit_gamma_log_cpp(const arma::mat &X, const arma::mat &XtX,
                            const arma::vec &y, const arma::vec &initial_beta,
                            bool approx) {
  arma::vec proposed_beta = initial_beta;
  // Initialization so that the y and eta_hat values start off close to
  // eachother.
  //
  // Note that formally there is a 1/nu here. However, we will cancel it out
  // with the gradient.
  arma::vec proposed_mu =
      arma::exp(arma::clamp(X * proposed_beta, -30.0, 30.0));
  proposed_mu =
      arma::clamp(proposed_mu, std::numeric_limits<double>::epsilon(), 1e15);
  double current_dev = calculate_deviance_gamma(y, proposed_mu);

  arma::vec step(X.n_cols);

  for (int i = 0; i < 20; i++) {

    // Gradient d_ll/d_eta is deriv of: (-e^-eta * y - eta) * nu
    arma::vec eta = X * proposed_beta;
    arma::vec mu = arma::exp(arma::clamp(eta, -30.0, 30.0));
    mu = arma::clamp(mu, std::numeric_limits<double>::epsilon(), 1e15);
    arma::vec grad = X.t() * (y / mu - 1.0);

    // The G matrix (diag(g'(mu) / alpha(mu)) = deriv(log(mu)) / 1)

    // Now we do step-halfing. Ensures that any improvement we make does
    // decrease the varaince. Avoids wild steps. Calculate the proposed step
    bool success = arma::solve(step, XtX, grad, arma::solve_opts::fast);
    if (!success) {
      // Rcpp::Rcout << "Less fast algorithm used \n";
      success = arma::solve(step, XtX, grad);
    }
    if (!success) {
      break;
      // Rcpp::Rcout << "Things broke";
    }
    // Propose a new beta
    arma::vec temp_beta = proposed_beta + step;
    arma::vec proposed_mu = arma::exp(arma::clamp(X * temp_beta, -30.0, 30.0));
    proposed_mu =
        arma::clamp(proposed_mu, std::numeric_limits<double>::epsilon(), 1e15);
    double proposed_dev = calculate_deviance_gamma(y, proposed_mu);
    // Step-Halving Loop
    int half_iter = 0;
    while ((!std::isfinite(proposed_dev) || proposed_dev > current_dev) &&
           half_iter < 25) {
      step = step / 2.0;                // Cut the step in half
      temp_beta = proposed_beta + step; // Try again
      proposed_mu = arma::exp(arma::clamp(X * temp_beta, -30.0, 30.0));
      proposed_mu = arma::clamp(proposed_mu,
                                std::numeric_limits<double>::epsilon(), 1e15);
      proposed_dev = calculate_deviance_gamma(y, proposed_mu);
      half_iter++;
    }
    proposed_beta = temp_beta;
    current_dev = proposed_dev;

    if (arma::norm(step) < 1e-6)
      break;
  }
  return proposed_beta;
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
double mle_estimate_dispersion_gamma(arma::vec y, arma::vec mu_hat, double p) {
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

// 2. The main simulation function
// Note that beta_vals is not the entire matrix of all possible betas, but just
// for a single vector.
// [[Rcpp::export]]
double glm_gamma_pl_cpp(arma::mat &X, const arma::mat &XtX, const arma::vec &y,
                        const arma::vec &mle_coefs, const arma::vec &beta_vals,
                        int m, bool approx, bool appendix) {
  int n = X.n_rows;

  // Compute true expected values based on proposed betas
  arma::vec eta = X * beta_vals;
  eta.elem(arma::find(eta > 50)).fill(50);
  eta.elem(arma::find(eta < -50)).fill(-50); // avoids D_mean explosions
  arma::vec mu = arma::exp(eta);
  // TODO: #11 Add more warnings
  // Note that Rcpp::Rcout will not play nicely with any parallelization
  // Rcpp::Rcout << "Predictions clamped";
  // Prevent mu from getting infinitesimally small
  mu.elem(arma::find(mu < 1e-8)).fill(1e-8);

  double dispersion = mle_estimate_dispersion_gamma(y, mu, beta_vals.n_elem);
  double shape = 1 / dispersion;
  // Compute full log-likelihood for the observed data under proposed beta
  double true_ll = compute_gamma_ll(y, eta, shape);

  // Note that the dispersion parameter is not estimated via mle in R's glm.
  // Note, however, that we are simply accepcting a dispersion parameter as
  // given in an argument
  arma::vec eta_hat = X * mle_coefs;
  eta_hat.elem(arma::find(eta_hat > 50)).fill(50);
  eta_hat.elem(arma::find(eta_hat < -50)).fill(-50);
  double mle_ll = compute_gamma_ll(y, eta_hat, shape);
  double f_x = true_ll - mle_ll;

  thread_local std::random_device rd;
  // rd() is a non-deterministic random number
  thread_local std::mt19937 gen(rd());

  // Simulate Y matrix inline using R's Gamma RNG
  arma::mat Y(n, m);
  arma::vec scale = mu * dispersion;
  for (int j = 0; j < m; ++j) {
    for (int i = 0; i < n; i++) {
      std::gamma_distribution<double> rgamma(shape, scale(i));
      // Uses gamma scale property
      double sim_val = rgamma(gen);
      Y(i, j) = (sim_val < 1e-10) ? 1e-10 : sim_val;
    }
  }

  int count_less = 0;

  // If doing the imvar thing, then we want to parallelize the 100 glm evals per
  // beta, rather than the betas.
#pragma omp parallel for reduction(+ : count_less) if (approx == true)
  for (int j = 0; j < m; ++j) {
    arma::vec y_sim = Y.col(j);
    arma::vec beta_sim_hat =
        fit_gamma_log_cpp(X, XtX, y_sim, beta_vals, approx);
    // Starting the IRLS algorithm at the beta_coefs that generated the Y
    arma::vec eta_sim_hat = X * beta_sim_hat;
    arma::vec mu_sim_hat = exp(eta_sim_hat);

    // Need to pass in data to use the mle estimator.
    double shape_sim_hat = 1 / mle_estimate_dispersion_gamma(
                                   y_sim, mu_sim_hat, beta_sim_hat.n_elem);

    double llX_j = compute_gamma_ll(y_sim, eta, shape_sim_hat);

    // Evaluate simulated MLE log-likelihood
    double mle_sim = compute_gamma_ll(y_sim, eta_sim_hat, shape_sim_hat);
    double f_X_j = llX_j - mle_sim;

    if (f_X_j < f_x) {
      count_less++;
    }
  }
  return (double)1.0 * count_less / m;
}
