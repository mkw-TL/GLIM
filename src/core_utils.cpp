// Translated to c++ by Gemini
// Thoroughly vetted

#include "headers.h"
#include <atomic>

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

// [[Rcpp::export]]
arma::mat generate_unit_matrix(int n, int d, uint32_t base_seed = 0,
                               int eval_index = 0) {
  // Derive a deterministic seed unique to this evaluation point
  uint32_t eval_seed = base_seed + static_cast<uint32_t>(eval_index * 10007);
  std::mt19937 gen(eval_seed);
  std::normal_distribution<double> rnorm(0.0, 1.0);

  // Initialize matrix: d rows (dimensions) by n columns (samples)
  arma::mat m(d, n);

  // Populate matrix with standard normal draws
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < d; ++i) {
      m(i, j) = rnorm(gen);
    }
  }

  // Normalize column-by-column (dim = 0) using L2 Euclidean norm (p = 2)
  return arma::normalise(m, 2, 0);
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
               X.n_rows, X.n_cols, y.n_elem, mle_coefs.n_elem,
               betas.row(1).n_elem);
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

  if (fam == GlmFamily::Binomial) {
    LogisticPlResult result = glm_logis_pl_cpp(
        X, y, mle_coefs, mle_coefs, 2, false, false, singular_warning, 1, 1);
    if (result.orig_separated) {
      Rcpp::stop("Initial data is separated. Exiting calculation");

      arma::mat pos{
          1, 1,
          arma::fill::value(
              arma::datum::nan)}; // Need to have curly braces, as this function
                                  // lives in a .h file, and C++ doesn't want to
                                  // use () for instantiation
      return (pos);
    }
  }

  bool show_progress = (!approx && !radial);
  StdoutProgressBar pb;
  Progress p(n_evals, show_progress, pb);

  // X = scale_design_matrix_cpp(X);
  // The Parallel Loop. Schedule(static) means that each thread is assigned
  // roughly the same amount of work schedule(dynamic) has a bit more overhead
  // which we don't need here. Don't want the overhead of allocating different
  // threads if it is fast enough to execute on a single
#pragma omp parallel for num_threads(num_threads) schedule(                    \
        guided) if (approx == false && parallel == true && radial == false)
  for (int i = 0; i < n_evals; i++) {
    // Only Thread 0 checks R's interrupt signal to avoid multi-threading
    // conflicts
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
                            singular_warning, base_seed, i);
      break;
    }

    case GlmFamily::Binomial: {
      LogisticPlResult result =
          glm_logis_pl_cpp(X, y, mle_coefs, beta_vals, m, approx, false,
                           singular_warning, base_seed, i);
      // if (result.sim_separated > 0) {
      //   // Note that sim_separated is a percentage of how many sims (for that
      //   // beta value) went poorly. Don't have a conceptual idea on how to
      //   pass
      //   // this back out.
      //   seperation_issues++;
      // }
      pl = result.poss;
      break;
    }

    case GlmFamily::Poisson: {
      PoissonPlResult result =
          glm_poisson_pl_cpp(X, y, mle_coefs, beta_vals, m, approx, false,
                             singular_warning, base_seed, i);
      // if (result.prop_separated > 0) {
      //   // Note that sim_separated is a percentage of how many sims (for that
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
                               singular_warning, base_seed, i);
      break;
    }

    case GlmFamily::Gaussian: {
      pl = glm_gaussian_pl_cpp(X, y, mle_coefs, beta_vals, m, approx, false,
                               singular_warning, base_seed, i);
      break;
    }

    default:
      pl = -1;
      // This should never be reached, since filtering occurs in R before being
      // passed here.
      break;
    }
    plausabilities(i) = pl;
    p.increment();
  }

  if (interrupted) {
    Rcpp::stop("Computation aborted by user.");
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
      if (it >= max_it) {
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

// [[Rcpp::export]]
arma::mat radial_code(int num_samps, arma::mat X, arma::vec y,
                      arma::vec mle_coefs, arma::mat eig_vecs,
                      arma::vec eig_vals, std::string family, double dispersion,
                      int m, double tol, int max_it, int a_val, int b_val,
                      uint32_t base_seed = 0) {
  int d = X.n_cols;
  arma::mat sampled_betas(mle_coefs.n_elem, num_samps);

  std::atomic<bool> singular_warning(false);

  // 1. Parse the family and precompute heavy math ONCE (safely on the main
  // thread)
  GlmFamily fam = string_to_family(family);
  if (fam == GlmFamily::Unknown) {
    Rcpp::stop("Family not supported in C++ backend.");
  }
  arma::mat XtX = X.t() * X; // Precompute here, not inside the loop!

  // --- Initial Separation Checks ---
  // Pass base_seed and an arbitrary eval_index (e.g., 0) for these checks
  if (fam == GlmFamily::Binomial) {
    LogisticPlResult result =
        glm_logis_pl_cpp(X, y, mle_coefs, mle_coefs, 2, false, false,
                         singular_warning, base_seed, 0);
    if (result.orig_separated) {
      Rcpp::stop("Initial data is separated. Exiting calculation");
      arma::mat betas{1, 1, arma::fill::value(arma::datum::nan)};
      return (betas);
    }
  }
  if (fam == GlmFamily::Poisson) {
    PoissonPlResult result =
        glm_poisson_pl_cpp(X, y, mle_coefs, mle_coefs, 2, false, false,
                           singular_warning, base_seed, 0);
    if (result.orig_separated) {
      Rcpp::stop("Initial data is separated. Exiting calculation");
      arma::mat betas{1, 1, arma::fill::value(arma::datum::nan)};
      return (betas);
    }
  }

  StdoutProgressBar pb;
  Progress p(num_samps, true, pb);

  std::atomic<bool> interrupted(false);

  // --- The Parallel Loop ---
#pragma omp parallel for schedule(static)
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
    std::mt19937 gen(outer_seed);
    std::uniform_real_distribution<double> runif(0.0, 1.0);

    double unif_alphas = runif(gen);

    // Note: Ensure `generate_unit_matrix` is entirely thread-safe!
    // If it relies on random numbers, you may need to pass `gen` or
    // `outer_seed` into it.
    arma::vec u = generate_unit_matrix(1, d);
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
        val1 = glm_logis_pl_cpp(X, y, mle_coefs, beta_proposal, m, approx, true,
                                singular_warning, base_seed, j)
                   .poss;
        break;
      }
      case GlmFamily::Gamma: {
        val1 = glm_gamma_pl_cpp(X, XtX, y, mle_coefs, beta_proposal, m, approx,
                                true, singular_warning, base_seed, j);
        break;
      }
      case GlmFamily::Poisson: {
        PoissonPlResult result =
            glm_poisson_pl_cpp(X, y, mle_coefs, beta_proposal, m, approx, false,
                               singular_warning, base_seed, j);
        val1 = result.poss;
        break;
      }
      case GlmFamily::InverseGaussian: {
        val1 = glm_invgauss_pl_cpp(X, y, mle_coefs, beta_proposal, m, approx,
                                   true, singular_warning, base_seed, j);
        break;
      }
      case GlmFamily::Gaussian: {
        val1 = glm_gaussian_pl_cpp(X, y, mle_coefs, beta_proposal, m, approx,
                                   true, singular_warning, base_seed, j);
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
    // Safely increment the progress bar from multiple threads
    if (!interrupted) {
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