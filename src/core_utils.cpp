// Translated to c++ by Gemini
// Thoroughly vetted

#include "headers.h"

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

// [[Rcpp::export]]
arma::mat generate_unit_matrix(int n, int d) {
  // Thread-local variables persist across calls but are completely unique to
  // each thread
  thread_local std::random_device rd;
  thread_local std::mt19937 gen(rd());
  std::normal_distribution<double> rnorm(0.0, 1.0);

  // Initialize matrix: d rows (dimensions) by n columns (samples)
  arma::mat m(d, n);

  // Note that this is the transpose
  // Populate the matrix with standard normals using our thread-safe engine
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < d; i++) {
      m(i, j) = rnorm(gen);
    }
  }

  // arma::normalise(matrix, p-norm, dimension)
  // p = 2 specifies the standard Euclidean L2 norm
  // dim = 0 tells Armadillo to process column-by-column (dim = 1 would do rows)
  return arma::normalise(m, 2, 0);
}

// [[Rcpp::export]]
arma::mat fit_glm_omp_cpp(arma::mat &X, const arma::vec &y,
                          const arma::vec &mle_coefs, const arma::mat &betas,
                          std::string family, // Pass string from R
                          int num_threads = 1, int m = 100,
                          bool parallel = true, bool approx = false,
                          bool appendix = false) {

  // Convert the string to an Enum once right here
  GlmFamily fam = string_to_family(family);
  // This is how we access enums in cpp.
  if (fam == GlmFamily::Unknown) {
    Rcpp::stop("Family not supported in C++ backend.");
  }

  int n_evals = betas.n_rows;
  arma::vec plausabilities(n_evals);
  arma::mat XtX = X.t() * X;
  double seperation_issues = 0;

  if (y.n_elem != X.n_rows || mle_coefs.n_elem != X.n_cols ||
      betas.n_cols != X.n_cols) {
    Rcpp::stop("Dimension mismatch: X is %d x %d, y has %d, mle_coefs has %d, "
               "beta_vals has %d",
               X.n_rows, X.n_cols, y.n_elem, mle_coefs.n_elem,
               betas.row(1).n_elem);
  }

  std::atomic<int> progress_count(0);
  // Only update the console every 2% of total iterations to protect performance
  int tick_step = n_evals / 50;
  if (tick_step < 1)
    tick_step = 1;

// If _OPENMP is defined, then it will run the omp function. Otherwise, will not
// throw an error
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#endif

  int next_percentage_milestone = 0;
  int percentage_step = 2; // Update every 2%

  if (fam == GlmFamily::Binomial) {
    LogisticPlResult result =
        glm_logis_pl_cpp(X, y, mle_coefs, mle_coefs, 2, false, false);
    if (result.orig_seperated) {
      Rcpp::stop("Initial data is seperated. Exiting calculation");

      arma::mat pos{
          1, 1,
          arma::fill::value(
              arma::datum::nan)}; // Need to have curly braces, as this function
                                  // lives in a .h file, and C++ doesn't want to
                                  // use () for instantiation
      return (pos);
    }
  }

  // X = scale_design_matrix_cpp(X);
  // The Parallel Loop. Schedule(static) means that each thread is assigned
  // roughly the same amount of work schedule(dynamic) has a bit more overhead
  // which we don't need here. Don't want the overhead of allocating different
  // threads if it is fast enough to execute on a single
#pragma omp parallel for schedule(                                             \
        guided) if (approx == false && parallel == true && appendix == false)
  for (int i = 0; i < n_evals; i++) {
    arma::vec beta_vals = betas.row(i).t();
    double pl;
    // Ending colon is a part of the case statement.
    switch (fam) {
    case GlmFamily::Gamma: {
      pl = glm_gamma_pl_cpp(X, XtX, y, mle_coefs, beta_vals, m, approx, false);
      break;
    }

    case GlmFamily::Binomial: {
      LogisticPlResult result =
          glm_logis_pl_cpp(X, y, mle_coefs, beta_vals, m, approx, false);
      if (result.sim_seperated > 0) {
        // Note that sim_seperated is a percentage of how many sims (for that
        // beta value) went poorly. Don't have a conceptual idea on how to pass
        // this back out.
        seperation_issues++;
      }
      pl = result.poss;
      break;
    }

    case GlmFamily::Poisson: {
      PoissonPlResult result =
          glm_poisson_pl_cpp(X, y, mle_coefs, beta_vals, m, approx, false);
      if (result.prop_seperated > 0) {
        // Note that sim_seperated is a percentage of how many sims (for that
        // beta value) went poorly. Don't have a conceptual idea on how to pass
        // this back out.
        seperation_issues++;
      }
      pl = result.poss;
      break;
    }

    case GlmFamily::InverseGaussian: {
      pl = glm_invgauss_pl_cpp(X, y, mle_coefs, beta_vals, m, approx, false);
      break;
    }

    case GlmFamily::Gaussian: {
      pl = glm_gaussian_pl_cpp(X, y, mle_coefs, beta_vals, m, approx, false);
      break;
    }

    default:
      pl = -1;
      // TODO #13 need a better exit method.
      break;
    }

    plausabilities(i) = pl;
    int current_progress = ++progress_count; // This is atomic!

    int thread_id = 0;
#ifdef _OPENMP
    thread_id = omp_get_thread_num();
#endif

    if (thread_id == 0 && approx == false && appendix == false) {
      // Calculate what the *actual current loop progress* is right now
      int current_percentage =
          (int)((double)100.0 * current_progress / n_evals);

      // If the actual progress has caught up to or passed our next milestone,
      // print it!
      if (current_percentage >= next_percentage_milestone) {
        int bar_width = 40;
        int pos = (int)(bar_width * ((double)current_percentage / 100.0));

        Rcpp::Rcout << "\rCalculating Plausibilities: [";
        for (int b = 0; b < bar_width; ++b) {
          if (b < pos)
            Rcpp::Rcout << "=";
          else if (b == pos)
            Rcpp::Rcout << ">";
          else
            Rcpp::Rcout << " ";
        }
        Rcpp::Rcout << "] " << current_percentage << "%" << std::flush;

        // Advance the milestone target past the current percentage
        next_percentage_milestone = current_percentage + percentage_step;
      }
    }
  }
  if (approx == false && appendix == false) {
    int bar_width = 40;
    Rcpp::Rcout << "\rCalculating Plausibilities: [";
    for (int b = 0; b < bar_width; ++b) {
      Rcpp::Rcout << "="; // Fill the entire bar cleanly
    }
    Rcpp::Rcout << "] 100%\n"
                << std::flush; // Print 100% and break to a new line
  }
  return plausabilities;
}

double w(double a_val, double b_val, int s) {
  return (double)a_val / std::pow(1.0 + s, b_val);
}

// [[Rcpp::export]]
arma::vec imvar(arma::mat X, arma::vec y, arma::vec xi,
                const std::string family, double alpha, const arma::vec &mle,
                const double mle_val, const arma::mat &J_vectors,
                const arma::vec &J_values, double dispersion, double tol = 1e-2,
                double a_val = 2.0, double b_val = 0.65, int max_it = 25,
                bool parallel = true, int m = 100) {
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

      double val1 = fit_glm_omp_cpp(X, y, mle, MPlus.t(), family, 1, m,
                                    parallel, true, false)(0, 0);
      double val2 = fit_glm_omp_cpp(X, y, mle, MMinus.t(), family, 1, m,
                                    parallel, true, false)(0, 0);
      double g_xi = std::max(val1, val2) - alpha;
      if (std::abs(g_xi) <= tol || it >= max_it) {
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
arma::mat appendix_code(int num_samps, arma::mat X, arma::vec y,
                        arma::vec mle_coefs, arma::mat eig_vecs,
                        arma::vec eig_vals, std::string family,
                        double dispersion, int m, double tol, int max_it,
                        int a_val, int b_val) {
  int d = X.n_cols;
  arma::mat sampled_betas(mle_coefs.n_elem, num_samps);

  // 1. Parse the family and precompute heavy math ONCE (safely on the main
  // thread)
  GlmFamily fam = string_to_family(family);
  if (fam == GlmFamily::Unknown) {
    Rcpp::stop("Family not supported in C++ backend.");
  }
  arma::mat XtX = X.t() * X; // Precompute here, not inside the loop!

  if (fam == GlmFamily::Binomial) {
    LogisticPlResult result =
        glm_logis_pl_cpp(X, y, mle_coefs, mle_coefs, 2, false, false);
    if (result.orig_seperated) {
      Rcpp::stop("Initial data is seperated. Exiting calculation");

      arma::mat betas{
          1, 1,
          arma::fill::value(
              arma::datum::nan)}; // Need to have curly braces, as this function
                                  // lives in a .h file, and C++ doesn't want to
                                  // use () for instantiation
      return (betas);
    }
  }
  if (fam == GlmFamily::Poisson) {
    PoissonPlResult result =
        glm_poisson_pl_cpp(X, y, mle_coefs, mle_coefs, m = 2, false, false);
    if (result.orig_seperated) {
      Rcpp::stop("Initial data is seperated. Exiting calculation");

      arma::mat betas{
          1, 1,
          arma::fill::value(
              arma::datum::nan)}; // Need to have curly braces, as this function
                                  // lives in a .h file, and C++ doesn't want to
                                  // use () for instantiation
      return (betas);
    }
  }

  Progress p(num_samps - 1, true);
  std::atomic<bool> interrupted(false);

  // The Parallel Loop
#pragma omp parallel for schedule(static)
  for (int j = 0; j < num_samps - 1;
       j++) { // -1 so that we can add the MLE at the end
    // Thread-safe RNG instantiation inside the loop
    if (interrupted) {
      continue;
    }
    std::random_device rd;
    std::mt19937 gen(rd() +
                     j); // Seed with j to ensure uniqueness across threads
    std::uniform_real_distribution<double> runif(0.0, 1.0);

    double unif_alphas = runif(gen);

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
      arma::vec beta_proposal =
          mle_coefs + dir_xi; // Don't need to go negative since the unit
                              // sphere covers this negative direction

      double val1 = 0.0;

      switch (fam) {
      case GlmFamily::Binomial: {
        val1 = glm_logis_pl_cpp(X, y, mle_coefs, beta_proposal, m, approx, true)
                   .poss;
        break;
      }
      case GlmFamily::Gamma: {
        val1 = glm_gamma_pl_cpp(X, XtX, y, mle_coefs, beta_proposal, m, approx,
                                true);
        break;
      }
      case GlmFamily::Poisson: {
        PoissonPlResult result = glm_poisson_pl_cpp(
            X, y, mle_coefs, beta_proposal, m, approx, false);
        val1 = result.poss;
        break;
      }
      case GlmFamily::InverseGaussian: {
        val1 = glm_invgauss_pl_cpp(X, y, mle_coefs, beta_proposal, m, approx,
                                   true);
        break;
      }
      case GlmFamily::Gaussian: {
        val1 = glm_gaussian_pl_cpp(X, y, mle_coefs, beta_proposal, m, approx,
                                   true);
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

  sampled_betas.col(num_samps - 1) = mle_coefs; // Remember off by one indexing
  return sampled_betas;
}
