// Translated to c++ by Gemini
// Thoroughly vetted

#include "headers.h"
#include <atomic>
#include <random>

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
GlmFamily string_to_family(const std::string &fam) {
  if (fam == "gaussian" || fam == "normal")
    return GlmFamily::Gaussian;
  if (fam == "binomial" || fam == "logistic")
    return GlmFamily::Binomial;
  if (fam == "poisson")
    return GlmFamily::Poisson;
  if (fam == "gamma")
    return GlmFamily::Gamma;
  if (fam == "inverse-gaussian" || fam == "inverse.gaussian")
    return GlmFamily::InverseGaussian;
  return GlmFamily::Unknown;
}

class StdoutProgressBar : public ProgressBar {
private:
  int _max_ticks;
  int _ticks_printed;

public:
  StdoutProgressBar() : _max_ticks(50), _ticks_printed(0) {}

  virtual void display() {
    // 50 characters wide header
    Rprintf("0%%   10   20   30   40   50   60   70   80   90   100%%\n");
    Rprintf("|----|----|----|----|----|----|----|----|----|----|\n");
  }

  virtual void update(float progress) {
    // Convert 0.0 - 1.0 progress fraction into target ticks out of 50
    int target_ticks = static_cast<int>(progress * _max_ticks);

    // Only print new '=' characters when a milestone is hit
    while (_ticks_printed < target_ticks && _ticks_printed < _max_ticks) {
      Rprintf("=");
      _ticks_printed++;
    }
  }

  virtual void end_display() {
    // Fill in any remaining ticks to hit 100% cleanly and wrap to a new line
    while (_ticks_printed < _max_ticks) {
      Rprintf("=");
      _ticks_printed++;
    }
    Rprintf("\n");
  }
};

// For efficient radial code (multiple uses)
arma::vec generate_unit_matrix(int rows, int d, std::minstd_rand &gen) {
  std::normal_distribution<double> rnorm(0.0, 1.0);

  arma::vec u(d);
  for (int i = 0; i < d; ++i) {
    u[i] = rnorm(gen);
  }

  double norm = arma::norm(u, 2);
  // Guard against the (astronomically unlikely) all-zero draw
  if (norm < 1e-300) {
    norm = 1e-300;
  }
  return u / norm;
}

// version for R
// [[Rcpp::export]]
arma::vec generate_unit_matrix_r(int rows, int d, uint32_t base_seed, int i) {
  // Initialize the generator exactly how your R code expects
  // (Adjust the math here if your old R code seeded it differently)
  std::minstd_rand gen(base_seed + i);

  // Call the internal helper
  return generate_unit_matrix(rows, d, gen);
}

// [[Rcpp::export]]
arma::mat fit_glm_omp_cpp(arma::mat &X, const arma::vec &y,
                          const arma::vec &mle_coefs, const arma::mat &betas,
                          std::string family, // Pass string from R
                          int num_threads = 1, int m = 100,
                          bool parallel = true, bool approx = false,
                          bool radial = false, uint32_t base_seed = 31415) {

  // Convert the string to an Enum once right here
  GlmFamily fam = string_to_family(family);
  // This is how we access enums in cpp.
  if (fam == GlmFamily::Unknown) {
    Rcpp::stop("Family not supported in C++ backend.");
  }

  int n_evals = betas.n_rows;
  arma::vec plausabilities(n_evals);
  arma::mat XtX = X.t() * X;
  double separation_issues = 0;

  if (y.n_elem != X.n_rows || mle_coefs.n_elem != X.n_cols ||
      betas.n_cols != X.n_cols) {
    Rcpp::stop("Dimension mismatch: X is %d x %d, y has %d, mle_coefs has %d, "
               "beta_vals has %d",
               X.n_rows, X.n_cols, y.n_elem, mle_coefs.n_elem, betas.n_cols);
  }

  std::atomic<bool> interrupted(false);

  std::atomic<bool> singular_warning(false);

  std::atomic<int> progress_count(0);
  // Only update the console every 2% of total iterations to protect performance
  int tick_step = n_evals / 50;
  if (tick_step < 1)
    tick_step = 1;

  int next_percentage_milestone = 0;
  int percentage_step = 2; // Update every 2%

  bool show_progress = (!approx && !radial);
  StdoutProgressBar pb;
  Progress p(n_evals, show_progress, pb);

// X = scale_design_matrix_cpp(X);
// The Parallel Loop. Schedule(static) means that each thread is assigned
// roughly the same amount of work schedule(dynamic) has a bit more overhead
// which we don't need here. Don't want the overhead of allocating different
// threads if it is fast enough to execute on a single
#pragma omp parallel num_threads(num_threads) if (approx == false &&           \
                                                      parallel == true &&      \
                                                      radial == false)
  {
    // 2. Allocate workspace ONCE per thread
    arma::mat Y_sim_thread_workspace(X.n_rows, m);

// 3. Divide the loop among the threads
#pragma omp for schedule(guided)
    for (int i = 0; i < n_evals; i++) {
#ifdef _OPENMP
      if (omp_get_thread_num() == 0) {
        if (Progress::check_abort()) {
          interrupted = true;
        }
      }
#else
      if (Progress::check_abort()) {
        interrupted = true;
      }
#endif

      if (interrupted) {
        continue;
      }
      arma::vec beta_vals = betas.row(i).t();
      double pl;
      // Ending colon is a part of the case statement.
      switch (fam) {
      case GlmFamily::Gamma: {
        pl = glm_gamma_pl_cpp(X, XtX, y, mle_coefs, beta_vals, m, approx, false,
                              singular_warning, base_seed, i,
                              Y_sim_thread_workspace);
        break;
      }

      case GlmFamily::Binomial: {
        LogisticPlResult result = glm_logis_pl_cpp(
            X, y, mle_coefs, beta_vals, m, approx, false, singular_warning,
            base_seed, i, Y_sim_thread_workspace);
        pl = result.poss;
        break;
      }

      case GlmFamily::Poisson: {
        PoissonPlResult result = glm_poisson_pl_cpp(
            X, y, mle_coefs, beta_vals, m, approx, false, singular_warning,
            base_seed, i, Y_sim_thread_workspace);
        // if (result.prop_separated > 0) {
        //   // Note that sim_separated is a percentage of how many sims (for
        //   that
        //   // beta value) went poorly. Don't have a conceptual idea on how to
        //   pass
        //   // this back out.
        //   seperation_issues++;
        // }
        pl = result.poss;
        break;
      }

      case GlmFamily::InverseGaussian: {
        pl = glm_invgauss_pl_cpp(X, y, mle_coefs, beta_vals, m, approx, false,
                                 singular_warning, base_seed, i,
                                 Y_sim_thread_workspace);
        break;
      }

      case GlmFamily::Gaussian: {
        pl = glm_gaussian_pl_cpp(X, y, mle_coefs, beta_vals, m, approx, false,
                                 singular_warning, base_seed, i,
                                 Y_sim_thread_workspace);
        break;
      }

      default:
        pl = -1;
        // This should never be reached, since filtering occurs in R before
        // being passed here.
        break;
      }
      plausabilities(i) = pl;
      p.increment();
    }
  }

  // We are now passed the OMP section

  if (interrupted) {
    Rcpp::stop("Computation aborted by user.");
  }

  if (singular_warning) {
    Rcpp::warning("System is singular, attempting approximate solution");
  }
  return plausabilities;
}

inline double w(double a_val, double b_val, int s) {
  // Optimization for common exact exponents
  if (b_val == 1.0) {
    return a_val / (1.0 + s);
  } else if (b_val == 0.5) {
    return a_val / std::sqrt(1.0 + s);
  }

  return a_val / std::pow(1.0 + s, b_val);
}

// [[Rcpp::export]]
arma::vec imvar(arma::mat X, arma::vec y, arma::vec xi,
                const std::string family, double alpha, const arma::vec &mle,
                const double mle_val, const arma::mat &J_vectors,
                const arma::vec &J_values, double dispersion,
                bool generate_grid, double tol = 1e-2, double a_val = 2.0,
                double b_val = 0.65, int max_it = 25, bool parallel = true,
                int m = 1000, uint32_t base_seed = 31) {
  int D = mle.size();

  if (family == "binomial") {
    arma::mat Y_sim(X.n_rows, 2);
    std::atomic singular_warning(false);
    LogisticPlResult result = glm_logis_pl_cpp(X, y, mle, mle, 2, false, false,
                                               singular_warning, 1, 1, Y_sim);
    if (result.orig_separated) {
      Rcpp::stop("Initial data is separated. Exiting calculation");

      arma::vec pos{
          1, arma::fill::value(
                 arma::datum::nan)}; // Need to have curly braces, as this
                                     // function lives in a .h file, and C++
                                     // doesn't want to use () for instantiation
      return (pos);
    }
  }

  // Setup Chi-Squared distribution to replicate R's qchisq()
  boost::math::chi_squared dist(D);
  double q_val = boost::math::quantile(dist, 1.0 - alpha);

  // These are all the dimensions we would like to traverse
  for (int d = 0; d < D; d++) {
    // log(xi) because we are getting the exponentiated version
    double xi_d = std::log(xi(d));

    // J_vectors.col(d) is J$vectors[, d]
    // This calculates our current best Q, and Cholesky decomp
    arma::vec posts_d =
        J_vectors.col(d) *
        std::sqrt(dispersion * q_val * std::abs(1.0 / J_values(d)));

    int it = 1;
    arma::vec posts_xi_d(D);
    arma::mat MPlus(D, 1);
    arma::mat MMinus(D, 1);
    while (true) {

      xi_d = std::max(-20.0, std::min(10.0, xi_d));
      // Xi scales singular values. exp parameterization avoids negative xi.
      posts_xi_d = posts_d * std::exp(xi_d / 2.0);
      MPlus = mle + posts_xi_d;
      MMinus = mle - posts_xi_d;

      // Unique seeds for MPlus vs MMinus and across Newton-Raphson iterations:
      uint32_t step_stride = base_seed + static_cast<uint32_t>(d * 500000) +
                             static_cast<uint32_t>(it * 20000);

      uint32_t seed_plus = step_stride;
      uint32_t seed_minus = step_stride + 10000;

      double val1 = fit_glm_omp_cpp(X, y, mle, MPlus.t(), family, 1, m,
                                    parallel, true, false, seed_plus)(0, 0);
      double val2 = fit_glm_omp_cpp(X, y, mle, MMinus.t(), family, 1, m,
                                    parallel, true, false, seed_minus)(0, 0);
      double g_xi = std::max(val1, val2) - alpha;
      if ((g_xi <= std::abs(tol)) || (it >= max_it)) {
        break;
      } else {
        xi_d = xi_d + w(a_val, b_val, it) * g_xi;
        it++;
      }
    }

    xi_d = std::max(-20.0, std::min(10.0, xi_d));
    xi(d) = std::exp(xi_d);
  }
  return xi;
}

// Written by gemini. Uses the trick to seperate into cases
inline double sum_softplus(const arma::vec &eta) {
  double sum_val = 0.0;
  for (arma::uword i = 0; i < eta.n_elem; ++i) {
    sum_val += (eta(i) > 0) ? (eta(i) + std::log1p(std::exp(-eta(i))))
                            : std::log1p(std::exp(eta(i)));
  }
  return sum_val;
}

// [[Rcpp::export]]
arma::mat radial_code(int num_samps, arma::mat X, arma::vec y,
                      arma::vec mle_coefs, arma::mat eig_vecs,
                      arma::vec eig_vals, std::string family, double dispersion,
                      int m, double tol, int max_it, int a_val, int b_val,
                      uint32_t base_seed = 0) {
  int d = X.n_cols;
  arma::mat sampled_betas(mle_coefs.n_elem, num_samps);

  std::atomic<bool> singular_warning(false);

  arma::mat XtX = X.t() * X; // Precompute here, not inside the loop!
  arma::mat Y_sim(X.n_rows, m);
  arma::vec eta_hat = X * mle_coefs;

  // Family-specific precomputations
  arma::vec p_hat;
  bool orig_separated = false;
  double mle_val_logis = 0.0;

  arma::vec eta_hat_gamma;

  double mle_ll_poisson = 0.0;

  double sigma_gauss = 0.0;
  double mle_val_gauss = 0.0;

  arma::vec mu_hat_invgauss;

  // 1. Parse the family and precompute heavy math ONCE (safely on the main
  // thread)
  GlmFamily fam = string_to_family(family);
  if (fam == GlmFamily::Unknown) {
    Rcpp::stop("Family not supported in C++ backend.");
  }

  // --- Initial Separation Checks ---
  // Pass base_seed and an arbitrary eval_index (e.g., 0) for these checks
  if (fam == GlmFamily::Binomial) {
    LogisticPlResult result =
        glm_logis_pl_cpp(X, y, mle_coefs, mle_coefs, 2, false, false,
                         singular_warning, base_seed, 0, Y_sim);
    if (result.orig_separated) {
      Rcpp::stop("Initial data is separated. Exiting calculation");
      arma::mat betas{1, 1, arma::fill::value(arma::datum::nan)};
      return (betas);
    }
    eta_hat = X * mle_coefs;
    p_hat = 1.0 / (1.0 + arma::exp(-eta_hat));
    orig_separated = arma::any(p_hat < 1e-8) || arma::any(p_hat > (1.0 - 1e-8));
    mle_val_logis = arma::dot(y, eta_hat) - sum_softplus(eta_hat);
  } else if (fam == GlmFamily::Poisson) {
    PoissonPlResult result =
        glm_poisson_pl_cpp(X, y, mle_coefs, mle_coefs, 2, false, false,
                           singular_warning, base_seed, 0, Y_sim);
    if (result.orig_separated) {
      Rcpp::stop("Initial data is separated. Exiting calculation");
      arma::mat betas{1, 1, arma::fill::value(arma::datum::nan)};
      return (betas);
    }
    arma::vec eta_hat_pois = arma::clamp(eta_hat, -10.0, 10.0);
    mle_ll_poisson = arma::dot(y, eta_hat_pois) -
                     arma::accu(exp(eta_hat_pois)) -
                     arma::accu(arma::lgamma(y + 1.0));
  } else if (fam == GlmFamily::Gamma) {
    // Precompute for Core Engine
    eta_hat_gamma = eta_hat;
    eta_hat_gamma.elem(arma::find(eta_hat_gamma > 50)).fill(50);
    eta_hat_gamma.elem(arma::find(eta_hat_gamma < -50)).fill(-50);
  } else if (fam == GlmFamily::Gaussian) {
    // Precompute for Core Engine
    sigma_gauss = std::sqrt(arma::accu((y - eta_hat) % (y - eta_hat)) /
                            (y.n_elem - mle_coefs.n_elem));
    mle_val_gauss =
        -(y.n_elem / 2.0) * std::log(2.0 * M_PI * sigma_gauss * sigma_gauss) -
        arma::accu((y - eta_hat) % (y - eta_hat)) /
            (2 * sigma_gauss * sigma_gauss);
  } else if (fam == GlmFamily::InverseGaussian) {
    // Precompute for Core Engine
    mu_hat_invgauss = arma::pow(eta_hat, -0.5);
  }
  StdoutProgressBar pb;
  Progress p(num_samps, true, pb);

  std::atomic<bool> interrupted(false);

  // --- The Parallel Loop ---
#pragma omp parallel
  {
    // Allocate workspace ONCE per thread
    arma::mat Y_sim_thread_workspace(X.n_rows, m);

// Divide the loop among the threads
#pragma omp for schedule(static)
    for (int j = 0; j < num_samps - 1; j++) {
#ifdef _OPENMP
      if (omp_get_thread_num() == 0) {
        if (Progress::check_abort()) {
          interrupted = true;
        }
      }
#else
      if (Progress::check_abort()) {
        interrupted = true;
      }
#endif

      if (interrupted) {
        continue;
      }

      // 2. Deterministic RNG
      // Offset by a large enough number (e.g., 9999) so the uniform draw
      // doesn't overlap with the inner GLM simulation seeds!
      uint32_t outer_seed = base_seed + static_cast<uint32_t>(j * 10007) + 9999;
      std::minstd_rand gen(outer_seed);
      std::uniform_real_distribution<double> runif(0.0, 1.0);

      double unif_alphas = runif(gen);

      // Note: Ensure `generate_unit_matrix` is entirely thread-safe!
      // If it relies on random numbers, you may need to pass `gen` or
      // `outer_seed` into it.

      arma::vec u = generate_unit_matrix(1, d, gen);
      arma::vec scaled_u = u / arma::sqrt(eig_vals);

      // Rotate by eigenvectors to match the tilt of the data
      arma::vec elliptical_dir = eig_vecs * scaled_u;
      arma::vec dir = arma::normalise(elliptical_dir);

      double starting_xi = 1.0;
      int it = 1;
      bool approx = false;

      while (true) {
        bool thread_is_primary = true;
#ifdef _OPENMP
        thread_is_primary = (omp_get_thread_num() == 0);
#endif
        if (thread_is_primary) {
          if (p.check_abort() & (it % 5 == 0)) {
            interrupted = true;
          }
        }
        if (interrupted) {
          break;
        }
        arma::vec dir_xi = dir * std::exp(starting_xi / 2.0);
        arma::vec beta_proposal = mle_coefs + dir_xi;

        double val1 = 0.0;

        // 3. Pass `base_seed` and iteration index `j` to inner GLM evaluators
        switch (fam) {
        case GlmFamily::Binomial: {
          val1 =
              glm_logis_pl_cpp(X, y, mle_coefs, beta_proposal, m, approx, true,
                               singular_warning, base_seed, j, mle_val_logis,
                               orig_separated, Y_sim_thread_workspace)
                  .poss;
          break;
        }
        case GlmFamily::Gamma: {
          val1 = glm_gamma_pl_cpp(X, XtX, y, mle_coefs, beta_proposal, m,
                                  approx, true, singular_warning, base_seed, j,
                                  eta_hat_gamma, Y_sim_thread_workspace);
          break;
        }
        case GlmFamily::Poisson: {
          PoissonPlResult result =
              glm_poisson_pl_cpp(X, y, mle_coefs, beta_proposal, m, approx,
                                 false, singular_warning, base_seed, j,
                                 mle_ll_poisson, Y_sim_thread_workspace);
          val1 = result.poss;
          break;
        }
        case GlmFamily::InverseGaussian: {
          val1 = glm_invgauss_pl_cpp(X, y, mle_coefs, beta_proposal, m, approx,
                                     true, singular_warning, base_seed, j,
                                     mu_hat_invgauss, Y_sim_thread_workspace);
          break;
        }
        case GlmFamily::Gaussian: {
          val1 = glm_gaussian_pl_cpp(
              X, y, mle_coefs, beta_proposal, m, approx, true, singular_warning,
              base_seed, j, sigma_gauss, mle_val_gauss, Y_sim_thread_workspace);
          break;
        }
        default:
          break;
        }

        double g_xi = val1 - unif_alphas;

        if ((std::abs(g_xi) <= tol) || (it >= max_it)) {
          sampled_betas.col(j) = beta_proposal;
          break;
        } else {
          starting_xi = starting_xi + w(a_val, b_val, it) * g_xi;
          it++;
        }
      }
      p.increment();
    }
  }

  if (interrupted) {
    Rcpp::stop("Computation interrupted by user.");
  }

  if (singular_warning) {
    Rcpp::warning("solve(): system is singular; attempting approx solution");
  }

  sampled_betas.col(num_samps - 1) = mle_coefs;
  return sampled_betas;
}