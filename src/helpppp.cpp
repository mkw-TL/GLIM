// Written by Gemini
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// IRLS logistic regression solver. Canonical link
// This completely replaces fastglm for the inner simulation loop
// Does not need to estimate the dispersion parameter since a function of the mean
// [[Rcpp::export]]
arma::vec fit_logistic_cpp(const arma::mat& X, const arma::vec& y) {
  arma::vec beta = arma::zeros(X.n_cols);

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
double glm_logis_pl_cpp(const arma::mat& X, const arma::vec& y,
                        const arma::vec& beta_vals, const double dispersion, const arma::vec& mle_coefs, double mle_val, int m) {
  int n = X.n_rows;

  // Compute true probabilities based on proposed betas
  arma::vec eta = X * beta_vals;
  arma::vec p = 1.0 / (1.0 + arma::exp(-eta));

  // Precompute constant scalar for f.x
  double sum_log_term = arma::sum(arma::log1p(arma::exp(eta)));
  double f_x = arma::dot(y, eta) - sum_log_term - mle_val;

  // 1. Generate an n x m matrix of Uniform(0,1) numbers in one shot
  arma::mat U = arma::randu<arma::mat>(n, m);
  arma::mat Y(n, m);

  // 2. Loop over columns and apply vectorized thresholding
  for(int j = 0; j < m; ++j) {
    // U.col(j) < p creates a boolean/unsigned vector (0s and 1s)
    // arma::conv_to converts it back to doubles to fit inside Y
    Y.col(j) = arma::conv_to<arma::vec>::from( U.col(j) < p );
  }
  // Fast cross-product for all M simulations
  arma::rowvec llX = eta.t() * Y - sum_log_term;

  int count_less = 0;

  // Inner loop: fit models entirely in C++
  for(int j = 0; j < m; ++j) {
    arma::vec y_sim = Y.col(j);

    arma::vec coefs = fit_logistic_cpp(X, y_sim);
    arma::vec eta_hat = X * coefs;

    double mle_sim = arma::dot(y_sim, eta_hat) - arma::sum(arma::log1p(arma::exp(eta_hat)));
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
//


double calculate_deviance_gamma(const arma::vec& y, const arma::vec& mu) {
  return 2.0 * arma::accu(y / mu - arma::log(y / mu) - 1.0);
}

// 1. IRLS Gamma regression solver (Log link), fisher weights, W = 1.
// [[Rcpp::export]]
arma::vec fit_gamma_log_cpp(const arma::mat& X, const arma::vec& y) {
  arma::vec beta = arma::zeros(X.n_cols);
  // log(0) would give negative infinity. 
  if (arma::mean(y) == 0) {
    Rcpp::Rcout << "What the heck";
  }
  beta(0) = std::log(arma::mean(y)); // Intercept = log(mean of y)
  // TODO: Note that this presumes an intercept in the first column.
  // Initialization so that the y and eta_hat values start off close to eachother.
  //
  // Note that formally there is a 1/nu here. However, we will cancel it out with the gradient.
  arma::mat XTX = X.t() * X;
  double current_dev = calculate_deviance_gamma(y, arma::exp(X*beta));


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
    arma::vec proposed_beta = beta + step;
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
    // 5. Accept the step
    beta = proposed_beta;
    current_dev = calculate_deviance_gamma(y, arma::exp(X*beta));

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
  // Mean deviance
    
  // Edge case: If the model fits perfectly, D_mean hits 0, implying infinite shape.
  if (D_mean <= 1e-10) {
      return 99999.0; 
  }
    
  // 2. Compute an initial guess using Thom's Approximation
  // (Uses a laurent series to approximate digamma, and then solve)
  double nu = (1.0 + std::sqrt(1.0 + (4.0 / 3.0) * D_mean)) / (4.0 * D_mean);
    
    // 3. 1D Newton-Raphson Loop to solve: log(nu) - digamma(nu) - D_mean = 0
    int max_iter = 100;
    double tol = 1e-8;
    
    for (int i = 0; i < max_iter; ++i) {
        // f(nu) = log(nu) - digamma(nu) - D_mean
        double f = std::log(nu) - R::digamma(nu) - D_mean;
        
        // f'(nu) = 1/nu - trigamma(nu)
        double f_prime = (1.0 / nu) - R::trigamma(nu);
        
        double step = f / f_prime;
        double next_nu = nu - step;
        
        // Safety measure: Newton steps can occasionally swing negative 
        // if the curve is exceptionally steep. If so, reduce the current dispersion by half.
        if (next_nu <= 0) {
            next_nu = nu * 0.5; 
        }
        
        // Check for convergence (relative to the size of nu)
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
double glm_gamma_pl_cpp(const arma::mat& X, const arma::vec& y,
                        const arma::vec& beta_vals, double dispersion, const arma::vec& mle_coefs, double mle_val,
                        int m) {
  int n = X.n_rows;
  
  // Compute true expected values based on proposed betas
arma::vec eta = X * beta_vals;
eta.elem(arma::find(eta > 700)).fill(700);
  eta.elem(arma::find(eta < -70)).fill(-70); // avoids D_mean explosions
  arma::vec mu = arma::exp(eta);
  // Rcpp::Rcout << "Predictions clamped";
  // Prevent mu from getting infinitesimally small
  mu.elem(arma::find(mu < 1e-8)).fill(1e-8);

  // arma::vec ratio = y / mu;
  // // Note that nu = shape. Estimates via mle, rather than pearson
  // dispersion is 1/nu
  // scale is mu / nu
  


  
  // Compute full log-likelihood for the observed data under proposed beta
  double true_ll = compute_gamma_ll(y, eta, dispersion);
  // Note that the dispersion parameter is not estimated via mle in R's glm.
  // Rcpp::Rcout << "true_ll: " << true_ll;
  arma::vec eta_hat = X * mle_coefs;
  // Rcpp::Rcout << "eta_hat: " << eta_hat;
  double mle_ll = compute_gamma_ll(y, eta_hat, dispersion);
  // Need to ensure that the value I compute for mle_ll is the same as mle_val
  Rcpp::Rcout << "mle_ll: (should be same as mle_val)" << mle_ll;
  Rcpp::Rcout << "mle_val: " << mle_val;
  // mle_val is calculated in R with a plugin estimator for phi, rather than the mle.
  // Checking if there is a difference
  Rcpp::Rcout << "true_ll: " << true_ll;
  double f_x = true_ll - mle_ll;
  Rcpp::Rcout << "f_x: " << f_x;

  // Simulate Y matrix inline using R's Gamma RNG
  arma::mat Y(n, m);
  // Shape parameter here is nu, which is 1/phi (because phi = 1/nu)
  double shape = 1/dispersion;
  // y / shape
  arma::vec scale = mu * dispersion;
  for(int j = 0; j < m; ++j) {
    for(int i = 0; i < n; ++i) {
      Y(i, j) = R::rgamma(shape, scale(i));
    }
  }

  // This computes \ell(beta, phi, X). phi and beta use the provided values.
  // Precompute constant pieces of log-likelihood across all M simulations
  double constant_ll_term = n * (shape * std::log(shape) - std::lgamma(shape));
  // Vectorized cross-product step for part of the log-likelihood evaluation
  arma::rowvec term3_all = -shape * (arma::exp(-eta).t() * Y + arma::sum(eta));

  int count_less = 0;

  // Inner loop: fit models entirely in C++
  for(int j = 0; j < m; ++j) {
    arma::vec y_sim = Y.col(j);
    // mu_hat needs to be here. 
    arma::vec beta_sim_hat = fit_gamma_log_cpp(X, y_sim);
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
    Rcpp::Rcout << "llX_j: " << llX_j;
    Rcpp::Rcout << "mle_sim" << mle_sim;
    double f_X_j = llX_j - mle_sim;
    Rcpp::Rcout << "f_X_j: " << f_X_j;

    if(f_X_j < f_x) {
      count_less++;
    }
  }

  return (double)count_less / m;
}


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
  double ll = -(n / 2.0) * std::log(2.0 * M_PI * sigma * sigma)
  - arma::sum(arma::pow(y - mu, 2)) / (2.0 * sigma * sigma);
  return ll;
}

// Main simulation function for Gaussian
// [[Rcpp::export]]
double glm_gaussian_pl_cpp(const arma::mat& X, const arma::vec& y,
                           const arma::vec& beta_vals, const double dispersion, const arma::vec& mle_coefs, double mle_val,
                           int m) {
  int n = X.n_rows;

  arma::vec mu = X * beta_vals; // Identity link
  // Estimated dispersion
  double estimated_dispersion = arma::accu((y - mu) % (y - mu)) / (y.n_elem - beta_vals.n_elem);

  double sigma = std::sqrt(estimated_dispersion);

  double true_ll = compute_gaussian_ll(y, mu, sigma, n);
  double f_x = true_ll - mle_val;

  // 1. Batch generate an n x m matrix of Standard Normals N(0,1)
  arma::mat Y = arma::randn<arma::mat>(n, m);

  // 2. Scale by standard deviation
  Y *= sigma;

  // 3. Add the mean vector (mu) to every column simultaneously
  Y.each_col() += mu;

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
arma::vec fit_poisson_log_cpp(const arma::mat& X, const arma::vec& y) {
  arma::vec beta = arma::zeros(X.n_cols);

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

    arma::vec step;
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
double glm_poisson_pl_cpp(const arma::mat& X, const arma::vec& y,
                          const arma::vec& beta_vals, const double dispersion, const arma::vec& mle_coefs, double mle_val, int m) {
  int n = X.n_rows;

  arma::vec eta = X * beta_vals;
  arma::vec mu = arma::exp(eta);

  double true_ll = glm_poisson_ll(eta, mu, y);

  double f_x = true_ll - mle_val;

  arma::mat Y(n, m);
  for(int j = 0; j < m; ++j) {
    for(int i = 0; i < n; ++i) {
      Y(i, j) = R::rpois(mu(i));
    }
  }

  int count_less = 0;

  for(int j = 0; j < m; ++j) {
    arma::vec y_sim = Y.col(j);

    arma::vec coefs = fit_poisson_log_cpp(X, y_sim);
    arma::vec eta_hat = X * coefs;
    arma::vec mu_hat = arma::exp(eta_hat);

    double mle_sim = 0.0;
    double llX_j = 0.0;
    for(int i = 0; i < n; ++i) {
      mle_sim += R::dpois(y_sim(i), mu_hat(i), 1);
      llX_j += R::dpois(y_sim(i), mu(i), 1);
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
// [[Rcpp::export]]
double rinvgauss_single(double mu, double lambda) {
  double v = R::rnorm(0, 1);
  double y_sq = v * v;
  double x = mu + (mu * mu * y_sq) / (2.0 * lambda) -
    (mu / (2.0 * lambda)) * std::sqrt(4.0 * mu * lambda * y_sq + mu * mu * y_sq * y_sq);

  double u = R::runif(0, 1);
  if (u <= mu / (mu + x)) {
    return x;
  } else {
    return (mu * mu) / x;
  }
}

// IRLS Inverse Gaussian regression solver (1/mu^2 link)
// [[Rcpp::export]]
arma::vec fit_invgauss_cpp(const arma::mat& X, const arma::vec& y) {
  arma::vec beta = arma::zeros(X.n_cols);

  // Initialization: For 1/mu^2 link, eta = X*beta must be strictly > 0
  double mean_y = arma::mean(y);
  beta(0) = 1.0 / (mean_y * mean_y);

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
double glm_invgauss_pl_cpp(const arma::mat& X, const arma::vec& y,
                           const arma::vec& beta_vals, const double dispersion, const arma::vec& mle_coefs, double mle_val,
                           int m) {
  int n = X.n_rows;

  arma::vec eta = X * beta_vals;
  arma::vec mu = arma::pow(eta, -0.5);

  double sbar = arma::accu(2*mu % (y - mu) / (mu%mu)) / y.n_elem;
  double estimated_dispersion = arma::accu(((y - mu) % (y - mu)) / (((mu % mu % mu) * (y.n_elem - beta_vals.n_elem)) * (1 + sbar)));
  double gamma = 1 / estimated_dispersion;

  double true_ll = compute_invgauss_ll(y, mu, gamma, n);
  double f_x = true_ll - mle_val;

  arma::mat Y(n, m);
  for(int j = 0; j < m; ++j) {
    for(int i = 0; i < n; ++i) {
      Y(i, j) = rinvgauss_single(mu(i), gamma);
    }
  }

  int count_less = 0;

  for(int j = 0; j < m; ++j) {
    arma::vec y_sim = Y.col(j);

    arma::vec coefs = fit_invgauss_cpp(X, y_sim);
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
