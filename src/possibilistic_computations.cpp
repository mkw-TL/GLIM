// Translated to c++ by Gemini
// Thoroughly vetted
#include <RcppArmadillo.h>
#include <cmath>
#include <string>
#include <omp.h>
#include <random>
#include <boost/math/special_functions/polygamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace arma;


// Define an Enum for the families so we aren't comparing strings in the loop
enum class GlmFamily {
  Gaussian,
  Binomial,
  Poisson,
  Gamma,
  InverseGaussian,
  Unknown
};


// Helper to convert R strings to our C++ Enum (done once outside the loop)
GlmFamily string_to_family(const std::string& fam) {
  if (fam == "gaussian" || fam == "normal") return GlmFamily::Gaussian;
  if (fam == "binomial" || fam == "logistic") return GlmFamily::Binomial;
  if (fam == "poisson") return GlmFamily::Poisson;
  if (fam == "gamma") return GlmFamily::Gamma;
  if (fam == "inverse-gaussian" || fam == "inverse.gaussian") return GlmFamily::InverseGaussian;
  return GlmFamily::Unknown;
}







// IRLS logistic regression solver. Canonical link
// This completely replaces fastglm for the inner simulation loop
// Does not need to estimate the dispersion parameter since a function of the mean
// [[Rcpp::export]]
arma::vec fit_logistic_cpp(const arma::mat& X, const arma::vec& y, const arma::vec& mle_coefs) {
  arma::vec beta = mle_coefs;

  for(int i = 0; i < 15; ++i) { // 15 iterations is usually plenty for convergence
    arma::vec eta = X * beta;
    arma::vec p = 1.0 / (1.0 + arma::exp(-eta));

    // Weights for IRLS
    arma::vec w = p % (1.0 - p);

    // Prevent strictly zero weights to avoid singular matrices
    w.elem(arma::find(w < 1e-6)).fill(1e-6);

    arma::mat XTWX = X.t() * arma::diagmat(w) * X;
    arma::vec grad = X.t() * (y - p);

    arma::vec step;
    // Attempt fast solve, fallback to robust solve if nearly singular
    bool success = arma::solve(step, XTWX, grad, arma::solve_opts::fast);
    if(!success) {
      arma::solve(step, XTWX, grad);
    }

    beta += step;

    // Check for convergence
    if(arma::norm(step) < 1e-6) break;
  }
  return beta;
}

// 2. The main simulation function
// Note that dispersion is not needed here. Only passing because it keeps consistency in the argument.
// [[Rcpp::export]]
double glm_logis_pl_cpp(const arma::mat& X, const arma::vec& y, const arma::vec& mle_coefs,
                        const arma::vec& beta_vals, int m, bool approx) {
  int n = X.n_rows;

  // Compute true probabilities based on proposed betas
  arma::vec eta = X * beta_vals;
  arma::vec p = 1.0 / (1.0 + arma::exp(-eta));

  arma::vec eta_hat = X * mle_coefs;
  double mle_val = arma::dot(y, eta_hat) - arma::sum(arma::log1p(arma::exp(eta_hat)));

  // Precompute constant scalar for f.x
  double sum_log_term = arma::sum(arma::log1p(arma::exp(eta)));
  double f_x = arma::dot(y, eta) - sum_log_term - mle_val;

  // Computing new random binomial data:
  thread_local std::random_device rd;
  thread_local std::mt19937 gen(rd());
  std::uniform_real_distribution<double> runif(0.0, 1.0);
  arma::mat Y(n, m);

  for(int j = 0; j < m; ++j) {
    for(int i = 0; i < n; ++i) {
      // ternary operator. Yields value of 1 if true and 0 if false. 
      Y(i, j) = (runif(gen) < p(i)) ? 1.0 : 0.0;
    }
  }
  // Fast cross-product for all M simulations
  arma::rowvec llX = eta.t() * Y - sum_log_term;

  int count_less = 0;

  // Inner loop: fit models entirely in C++
  #pragma omp parallel for schedule(static) reduction(+:count_less) if(approx == true)
  for(int j = 0; j < m; ++j) {
    arma::vec y_sim = Y.col(j);

    arma::vec sim_coefs = fit_logistic_cpp(X, y_sim, mle_coefs);
    arma::vec eta_sim_hat = X * sim_coefs;

    double mle_sim = arma::dot(y_sim, eta_sim_hat) - arma::sum(arma::log1p(arma::exp(eta_sim_hat)));
    double f_X_j = llX(j) - mle_sim;

    if(f_X_j < f_x) {
      count_less++;
    }
  }

  return (double)count_less / m;
}

// Gamma regression (log-link)
//
//


double calculate_deviance_gamma(const arma::vec& y, const arma::vec& mu) {
  return 2.0 * arma::accu(y / mu - arma::log(y / mu) - 1.0);
}

// IRLS Gamma regression solver (Log link), fisher weights, W = 1.
// [[Rcpp::export]]
arma::vec fit_gamma_log_cpp(const arma::mat& X, const arma::mat& XtX, const arma::vec& y, const arma::vec& mle_coefs) {
  arma::vec beta = mle_coefs;
  // Initialization so that the y and eta_hat values start off close to eachother.
  //
  // Note that formally there is a 1/nu here. However, we will cancel it out with the gradient.
  arma::mat XTX = XtX;
  double current_dev = calculate_deviance_gamma(y, arma::exp(X*beta));

  arma::vec step(X.n_rows);
  arma::vec proposed_beta(X.n_cols);
  arma::vec proposed_mu(X.n_rows);

  for(int i = 0; i < 20; ++i) {
    arma::vec eta = X * beta;
    arma::vec mu = arma::exp(eta);

    // Gradient d_ll/d_eta is deriv of: (-e^-eta * y - eta) * nu
    arma::vec grad = X.t() * (y / mu - 1.0);

    // Now we do step-halfing. Ensures that any improvement we make does decrease the varaince. Avoids wild steps.
    // Calculate the proposed step
    arma::vec step;
    bool success = arma::solve(step, XTX, grad, arma::solve_opts::fast);
    if (!success) {
      arma::solve(step, XTX, grad);
    }
    // Propose a new beta
    proposed_beta = beta + step;
    arma::vec proposed_mu = arma::exp(X * proposed_beta);
    double proposed_dev = calculate_deviance_gamma(y, proposed_mu);
    // Step-Halving Loop
    int half_iter = 0;
    while (proposed_dev > current_dev && half_iter < 10) {
        step = step / 2.0;                         // Cut the step in half
        proposed_beta = beta + step;          // Try again
        proposed_mu = arma::exp(X * proposed_beta);
        proposed_dev = calculate_deviance_gamma(y, proposed_mu);
        half_iter++;
    }
    beta = proposed_beta;
    current_dev = proposed_dev;

    if(arma::norm(step) < 1e-6) break;
  }
  return beta;
}

// Helper function to compute full Gamma log-likelihood matching R's logLik()
// [[Rcpp::export]]
double compute_gamma_ll(const arma::vec& y, const arma::vec& eta, double shape) {
  double term1 = y.n_elem * (shape * std::log(shape) - std::lgamma(shape));
  double term2 = (shape - 1.0) * arma::sum(arma::log(y));
  double term3 = -shape * (arma::dot(y, arma::exp(-eta)) + arma::sum(eta));
  return term1 + term2 + term3;
}

// [[Rcpp::export]]
arma::vec compute_gamma_ll_mat(const arma::vec&y, const arma::mat& eta, double shape) {
  arma::vec ll(eta.n_cols);
  for(int i = 0; i < eta.n_cols; ++i) {
    ll(i) = compute_gamma_ll(y, eta.col(i), shape);
  }
  return ll;
}

// Uses Fletcher's correction to Pearson's MoM estimator. (Not yet -- keeping it simple)
// [[Rcpp::export]]
double pearson_estimate_dispersion_gamma(arma::vec y, arma::vec mu_hat, double p) {
    // // Have canceled out a mu on top and bottom
  // double sbar = arma::accu((y - mu_hat)/ (2*mu_hat))/y.n_elem;
  return (double)arma::accu(arma::square(y - mu_hat) / arma::square(mu_hat)) / ((y.n_elem - p));
}

// [[Rcpp::export]]
double mle_estimate_dispersion_gamma(arma::vec y, arma::vec mu_hat, double p) {
// Safety net against zero or negative values disrupting logs
  arma::vec ratio = y/mu_hat;
  ratio.elem(arma::find(ratio < 1e-8)).fill(1e-8);
    
  // D_mean is the mean deviance-like component: mean( y/mu - log(y/mu) - 1 )
  // Note that this is the unscaled deviance. The estimated dispersion parameter
  // are the same for a particular dataset.
  double D_mean = arma::mean(ratio - arma::log(ratio) - 1.0);

  // Edge case: If the model fits perfectly, D_mean hits 0, implying infinite shape.
  if (D_mean <= 1e-10) {
      return 99999.0; 
  }
    
  // Compute an initial guess using Thom's Approximation
  // (Uses a laurent series to approximate digamma, and then solve the resulting quadratic)
  // We want to taylor expand nu around zero, but we have 1/nu. There is a pole there, and thus we can't use taylor approx
  // Laurent expansion still valid, though.
  double nu = (1.0 + std::sqrt(1.0 + (4.0 / 3.0) * D_mean)) / (4.0 * D_mean);
    
    // 1D Newton-Raphson Loop to solve: log(nu) - digamma(nu) - D_mean = 0
    int max_iter = 100;
    double tol = 1e-8;
    
    for (int i = 0; i < max_iter; ++i) {
      // psi is the digamma function
        double f = std::log(nu) - boost::math::digamma(nu) - D_mean;
        double f_prime = (1.0 / nu) - boost::math::polygamma(1, nu);

        double step = f / f_prime;
        double next_nu = nu - step;
        
        // Safety measure: Newton steps can occasionally swing negative 
        // if the curve is exceptionally steep. If so, reduce the current dispersion by half.
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
  return(1/nu);
}

// 2. The main simulation function
// Note that beta_vals is not the entire matrix of all possible betas, but just for a single vector.
// [[Rcpp::export]]
double glm_gamma_pl_cpp(const arma::mat& X, const arma::vec& y, const arma::vec& mle_coefs,
                        const arma::vec& beta_vals, int m, bool approx) {
  int n = X.n_rows;
  
  // Compute true expected values based on proposed betas
  arma::vec eta = X * beta_vals;
  eta.elem(arma::find(eta > 700)).fill(700);
  eta.elem(arma::find(eta < -70)).fill(-70); // avoids D_mean explosions
  arma::vec mu = arma::exp(eta);
  // TODO: #11 Add more warnings
  // Note that Rcpp::Rcout will not play nicely with any parallelization
  // Rcpp::Rcout << "Predictions clamped";
  // Prevent mu from getting infinitesimally small
  mu.elem(arma::find(mu < 1e-8)).fill(1e-8);
  
  double dispersion = mle_estimate_dispersion_gamma(y, mu, beta_vals.n_elem);
  double shape = 1/dispersion;
  arma::mat XTX = X.t() * X;
  
  // Compute full log-likelihood for the observed data under proposed beta
  double true_ll = compute_gamma_ll(y, eta, shape);

  // Note that the dispersion parameter is not estimated via mle in R's glm.
  // Note, however, that we are simply accepcting a dispersion parameter as given in an argument
  arma::vec eta_hat = X * mle_coefs;
  double mle_ll = compute_gamma_ll(y, eta_hat, shape);
  double f_x = true_ll - mle_ll;

  thread_local std::random_device rd;
  // rd() is a non-deterministic random number
  thread_local std::mt19937 gen(rd());

  // Simulate Y matrix inline using R's Gamma RNG
  arma::mat Y(n, m);
  arma::vec scale = mu * dispersion;
  for(int j = 0; j < m; ++j) {
    for(int i = 0; i < n; ++i) {
      std::gamma_distribution<double> rgamma(shape, scale(i));
      Y(i, j) = rgamma(gen);
    }
  }

  // This computes \ell(beta, phi, X). phi and beta use the provided values.
  // Precompute constant pieces of log-likelihood across all M simulations
  double constant_ll_term = n * (shape * std::log(shape) - std::lgamma(shape));
  // Vectorized cross-product step for part of the log-likelihood evaluation
  arma::rowvec term3_all = -shape * (arma::exp(-eta).t() * Y + arma::sum(eta));

  int count_less = 0;

  #pragma omp parallel for reduction(+:count_less) if(approx == true)
  for(int j = 0; j < m; ++j) {
    arma::vec y_sim = Y.col(j);
    arma::vec beta_sim_hat = fit_gamma_log_cpp(X, XTX,y_sim, mle_coefs);
    arma::vec eta_sim_hat = X * beta_sim_hat;
    arma::vec mu_sim_hat = exp(eta_sim_hat);

    // Need to pass in data to use the mle estimator.
    double shape_sim_hat = 1/mle_estimate_dispersion_gamma(y_sim, mu_sim_hat, beta_sim_hat.n_elem);

    // May be an an error in this calculation. Keep it simple for now.
    // // Calculate the simulated dependent term: (shape - 1) * sum(log(y_sim))
    // double term2_j = (shape_sim_hat - 1.0) * arma::sum(arma::log(y_sim));
    // double llX_j = constant_ll_term + term2_j + term3_all(j);
    double llX_j = compute_gamma_ll(y_sim, eta, shape_sim_hat);

    // Evaluate simulated MLE log-likelihood
    double mle_sim = compute_gamma_ll(y_sim, eta_sim_hat, shape_sim_hat);
    double f_X_j = llX_j - mle_sim;

    if(f_X_j < f_x) {
      count_less++;
    }
  }

  return (double)count_less / m;
}



// Anything below is still in progress. Not that above isn't, but you know. 
// =====================================================================
// GAUSSIAN REGRESSION (IDENTITY LINK)
// =====================================================================

// Identity link Gaussian regression is just Ordinary Least Squares (OLS)
// No IRLS loop is required.
// [[Rcpp::export]]
arma::vec fit_gaussian_cpp(const arma::mat& X, const arma::vec& y) {
  arma::vec beta;
  bool success = arma::solve(beta, X, y, arma::solve_opts::fast);
  if(!success) {
    arma::solve(beta, X, y);
  }
  return beta;
}

// Helper function to compute Gaussian log-likelihood
// [[Rcpp::export]]
double compute_gaussian_ll(const arma::vec& y, const arma::vec& mu, double sigma, int n) {
  double ll = -(n / 2.0) * std::log(2.0 * M_PI * sigma * sigma);
  - arma::sum(arma::pow(y - mu, 2)) / (2.0 * sigma * sigma);
  return ll;
}

// Main simulation function for Gaussian
// [[Rcpp::export]]
double glm_gaussian_pl_cpp(const arma::mat& X, const arma::vec& y, const arma::vec& mle_coefs,
                           const arma::vec& beta_vals, int m, bool approx) {
  int n = X.n_rows;

  arma::vec mu = X * beta_vals; // Identity link
  // Estimated dispersion
  double estimated_dispersion = arma::accu((y - mu) % (y - mu)) / (y.n_elem - beta_vals.n_elem);

  double sigma = std::sqrt(estimated_dispersion);

  double true_ll = compute_gaussian_ll(y, mu, sigma, n);
  arma::vec mu_hat = X * mle_coefs;
  double mle_val = compute_gaussian_ll(y, mu_hat, sigma, n);
  double f_x = true_ll - mle_val;

  // Needed to change from arma to a thread safe version. 

thread_local std::random_device rd;
  thread_local std::mt19937 gen(rd());
  std::normal_distribution<double> rnorm(0.0, 1.0);

  arma::mat Y(n, m);
  for(int j = 0; j < m; ++j) {
    for(int i = 0; i < n; ++i) {
      Y(i, j) = mu(i) + sigma * rnorm(gen);
    }
  }

  int count_less = 0;

  for(int j = 0; j < m; ++j) {
    arma::vec y_sim = Y.col(j);

    arma::vec coefs = fit_gaussian_cpp(X, y_sim);
    arma::vec mu_hat = X * coefs;

    double mle_sim = compute_gaussian_ll(y_sim, mu_hat, sigma, n);
    double llX_j = compute_gaussian_ll(y_sim, mu, sigma, n);

    double f_X_j = llX_j - mle_sim;

    if(f_X_j < f_x) {
      count_less++;
    }
  }

  return (double)count_less / m;
}

// =====================================================================
// POISSON REGRESSION (LOG LINK)
// =====================================================================

// IRLS Poisson regression solver (Log link)
// [[Rcpp::export]]
arma::vec fit_poisson_log_cpp(const arma::mat& X, const arma::vec& y, const arma::vec& mle_coefs) {
  arma::vec beta = mle_coefs;

  arma::vec step(X.n_cols);
  arma::vec proposed_beta(X.n_cols);
  arma::vec proposed_mu(X.n_rows);

  for(int i = 0; i < 20; ++i) {
    arma::vec eta = X * beta;

    // Cap eta to prevent overflow when calculating expected values
    eta.elem(arma::find(eta > 20)).fill(20);
    arma::vec mu = arma::exp(eta);

    // Weights for Poisson log-link
    arma::vec w = mu;
    w.elem(arma::find(w < 1e-8)).fill(1e-8);

    arma::mat XTWX = X.t() * arma::diagmat(w) * X;
    arma::vec grad = X.t() * (y - mu);

    bool success = arma::solve(step, XTWX, grad, arma::solve_opts::fast);
    if(!success) {
      if(!arma::solve(step, XTWX, grad)) break;
    }

    beta += step;
    if(arma::norm(step) < 1e-6) break;
  }
  return beta;
}

// [[Rcpp::export]]
double glm_poisson_ll(arma::vec& eta, arma::vec& mu, const arma::vec& y) {

  double term1 = arma::dot(y, eta);

  // Sum of mu
  double term2 = arma::accu(mu);

  // log(y!) is equivalent to lgamma(y + 1)
  double term3 = arma::accu(arma::lgamma(y + 1.0));

  return term1 - term2 - term3;
}

// Main simulation function for Poisson
// [[Rcpp::export]]
double glm_poisson_pl_cpp(const arma::mat& X, const arma::vec& y, const arma::vec& mle_coefs,
                          const arma::vec& beta_vals, int m, bool approx) {
  int n = X.n_rows;

  arma::vec eta = X * beta_vals;
  arma::vec mu = arma::exp(eta);

  double true_ll = glm_poisson_ll(eta, mu, y);
  arma::vec eta_hat = X * mle_coefs;
  arma::vec mu_hat = arma::exp(eta_hat);
  double mle_val = glm_poisson_ll(eta_hat, mu_hat, y);

  double f_x = true_ll - mle_val;

  arma::mat Y(n, m);

  // Constructor
  thread_local std::random_device rd;
  // rd() is a non-deterministic random number
  thread_local std::mt19937 gen(rd());
  // Is using the random import to get access to these distributions
  for(int j = 0; j < m; ++j) {
    for(int i = 0; i < n; ++i) {
      std::poisson_distribution<int> rpois(mu(i));
      Y(i, j) = rpois(gen);
    }
  }

  int count_less = 0;

  for(int j = 0; j < m; ++j) {
    arma::vec y_sim = Y.col(j);

    arma::vec coefs_sim = fit_poisson_log_cpp(X, y_sim, mle_coefs);
    arma::vec eta_hat_sim = X * coefs_sim;
    arma::vec mu_hat_sim = arma::exp(eta_hat_sim);

    double mle_sim = 0.0;
    double llX_j = 0.0;
    // Compare log likelihoods. Gamma y+1 is y!
    for(int i = 0; i < n; ++i) {
      mle_sim += y_sim(i) * std::log(mu_hat_sim(i)) - mu_hat_sim(i) - std::lgamma(y_sim(i) + 1.0);
      llX_j += y(i) * std::log(mu(i)) - mu(i) - std::lgamma(y(i) + 1.0);
    }

    double f_X_j = llX_j - mle_sim;

    if(f_X_j < f_x) {
      count_less++;
    }
  }

  return (double)count_less / m;
}

// =====================================================================
// INVERSE GAUSSIAN REGRESSION (1/MU^2 LINK)
// =====================================================================

// R does not have a native rinvgauss. Implemented is the Michael,
// Schucany, and Haas (1976) algorithm to simulate it via C++.
// Pass the pointer to the random number generator
// Don't need to export it as an R object, because R doesn't know what the mercene twister generator is
double rinvgauss_single(double mu, double lambda, std::mt19937& gen) {
  // <> denotes the type expected out of this template
  // Loosely, templates can be thought of an instance of a class
  std::normal_distribution<double> rnorm(0.0, 1.0);
  std::uniform_real_distribution<double> runif(0.0, 1.0);
  double v = rnorm(gen);

  double y_sq = v * v;
  double x = mu + (mu * mu * y_sq) / (2.0 * lambda) -
    (mu / (2.0 * lambda)) * std::sqrt(4.0 * mu * lambda * y_sq + mu * mu * y_sq * y_sq);

  double u = runif(gen);
  if (u <= mu / (mu + x)) {
    return x;
  } else {
    return (mu * mu) / x;
  }
}

// [[Rcpp::export]]
double mle_estimate_dispersion_inv_gauss(const arma::vec& y, const double ybar) {
  // Taking the mle wrt mu yields ybar (presuming mu =/= 0)
  return y.n_elem / (arma::accu(ybar * ybar * y / arma::dot((y - ybar),(y-ybar))));
}

// IRLS Inverse Gaussian regression solver (1/mu^2 link)
// [[Rcpp::export]]
arma::vec fit_invgauss_cpp(const arma::mat& X, const arma::vec& y, const arma::vec& mle_coefs) {
  arma::vec beta = mle_coefs;
  for(int i = 0; i < 30; ++i) {
    arma::vec eta = X * beta;
    // Enforce strict positivity constraint for the link function
    eta.elem(arma::find(eta < 1e-6)).fill(1e-6);

    arma::vec mu = arma::pow(eta, -0.5);

    // For Inverse Gaussian 1/mu^2 link:
    // W = diag(mu^3) as opposed to diag(mu^3 / 4) for canonical link
    // Modified gradient step RHS = -2 * X^T (y - mu)
    arma::vec w = arma::pow(mu, 3.0);

    // Clamp extreme weights to preserve numerical stability
    w.elem(arma::find(w < 1e-8)).fill(1e-8);
    w.elem(arma::find(w > 1e8)).fill(1e8);
    // -2 is pulled out front to make computations nicer
    arma::mat XTWX = X.t() * arma::diagmat(w) * X;
    arma::vec grad = -2.0 * X.t() * (y - mu);

    arma::vec step;
    bool success = arma::solve(step, XTWX, grad, arma::solve_opts::fast);
    if(!success) {
      if(!arma::solve(step, XTWX, grad)) break;
    }

    // Line search (step halving) to ensure eta remains > 0
    arma::vec new_beta = beta + step;
    arma::vec new_eta = X * new_beta;
    int iter_halve = 0;
    while (arma::any(new_eta <= 0.0) && iter_halve < 10) {
      step /= 2.0;
      new_beta = beta + step;
      new_eta = X * new_beta;
      iter_halve++;
    }

    beta = new_beta;
    if(arma::norm(step) < 1e-6) break;
  }
  return beta;
}

// Helper function to compute Inverse Gaussian log-likelihood
// [[Rcpp::export]]
double compute_invgauss_ll(const arma::vec& y, const arma::vec& mu, double gamma_val, int n) {
  // gamma is 1/estimated_phi
  double term1 = 0.5 * n * std::log(gamma_val / (2.0 * M_PI));
  double term2 = -1.5 * arma::sum(arma::log(y));
  double term3 = -gamma_val * arma::sum(arma::pow(y - mu, 2) / (2.0 * arma::pow(mu, 2) % y));
  return term1 + term2 + term3;
}

// Main simulation function for Inverse Gaussian
// [[Rcpp::export]]
double glm_invgauss_pl_cpp(const arma::mat& X, const arma::vec& y, const arma::vec& mle_coefs,
                           const arma::vec& beta_vals,
                           int m, bool approx) {
  int n = X.n_rows;

  arma::vec eta = X * beta_vals;
  arma::vec mu = arma::pow(eta, -0.5);

  double sbar = arma::accu(2*mu % (y - mu) / (mu%mu)) / y.n_elem;
  double estimated_dispersion = arma::accu(((y - mu) % (y - mu)) / (((mu % mu % mu) * (y.n_elem - beta_vals.n_elem)) * (1 + sbar)));
  double gamma = 1 / estimated_dispersion;

  arma::vec eta_hat = X * mle_coefs;
  arma::vec mu_hat = arma::pow(eta_hat, -.5);

  double true_ll = compute_invgauss_ll(y, mu, gamma, n);
  double mle_val = compute_invgauss_ll(y, mu_hat, gamma, n);
  double f_x = true_ll - mle_val;

  thread_local std::random_device rd;
  thread_local std::mt19937 gen(rd());

  arma::mat Y(n, m);
  for(int j = 0; j < m; ++j) {
    for(int i = 0; i < n; ++i) {
      Y(i, j) = rinvgauss_single(mu(i), gamma, gen);
    }
  }

  int count_less = 0;

  for(int j = 0; j < m; ++j) {
    arma::vec y_sim = Y.col(j);

    arma::vec coefs = fit_invgauss_cpp(X, y_sim, mle_coefs);
    arma::vec eta_hat = X * coefs;

    // Validate eta_hat to compute simulated MLE likelihood
    eta_hat.elem(arma::find(eta_hat < 1e-6)).fill(1e-6);
    arma::vec mu_hat = arma::pow(eta_hat, -0.5);

    double mle_sim = compute_invgauss_ll(y_sim, mu_hat, gamma, n);
    double llX_j = compute_invgauss_ll(y_sim, mu, gamma, n);

    double f_X_j = llX_j - mle_sim;

    if(f_X_j < f_x) {
      count_less++;
    }
  }

  return (double)count_less / m;
}



// Need to bring the main function (which calls all other functions) after any functions that it calls

// [[Rcpp::export]]
arma::mat fit_glm_omp_cpp(const arma::mat& X, 
                          const arma::vec& y, 
                          const arma::vec& mle_coefs,
                          const arma::mat& betas, 
                          std::string family_str, // Pass string from R
                          bool approx,
                          int num_threads = 1,
                          int m = 100) {
  
  // Convert the string to an Enum once right here
  GlmFamily family = string_to_family(family_str);
  // This is how we access enums in cpp.
  if (family == GlmFamily::Unknown) {
    Rcpp::stop("Family not supported in C++ backend.");
  }

  int n_evals = betas.n_rows;
  int n_cols = betas.n_cols; 
  arma::vec plausabilities(n_evals);
  arma::mat XtX = X.t() * X;
  
  // If _OPENMP is defined, then it will run the omp function. Otherwise, will not throw an error
  #ifdef _OPENMP
  omp_set_num_threads(num_threads);
  #endif

  // The Parallel Loop. Schedule(static) means that each thread is assigned roughly the same amount of work
  // schedule(dynamic) has a bit more overhead which we don't need here.
  // Don't want the overhead of allocating different threads if it is fast enough to execute on a single
  #pragma omp parallel for schedule(static) if(n_evals > 10)
  for (int i = 0; i < n_evals; ++i) {
    arma::vec beta_vals = betas.row(i).t();
    double pl;
    
    // Ending colon is a part of the case statement.
    switch(family) {
      case GlmFamily::Gamma:
        pl = glm_gamma_pl_cpp(X, y, mle_coefs, beta_vals, m, approx);
        break;
        
      case GlmFamily::Binomial:
        pl = glm_logis_pl_cpp(X, y, mle_coefs, beta_vals, m, approx);
        break;
        
      case GlmFamily::Poisson:
        pl = glm_poisson_pl_cpp(X, y, mle_coefs, beta_vals, m, approx);
        break;

      case GlmFamily::InverseGaussian:
        pl = glm_invgauss_pl_cpp(X, y, mle_coefs, beta_vals, m, approx);
        break;

      case GlmFamily::Gaussian:
        pl = glm_gaussian_pl_cpp(X, y, mle_coefs, beta_vals, m, approx);
        break;
        
      default:
        // Fallback or placeholder for Gaussian/Identity
        pl = -1;
        // TODO #13 need a better exit method.
        break;
    }
    
    plausabilities(i) = pl;
  }
  
  return plausabilities;
}
