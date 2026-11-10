// Translated to c++ by Gemini
// Thoroughly vetted

#include "headers.h"
#include <atomic>
#include <exception>

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
double compute_ll() { return 0; }

// [[Rcpp::export]]
arma::mat fit_glm_omp_cpp(arma::mat &X, const arma::vec &y,
                          const arma::vec &mle_coefs, double mle_val,
                          const arma::mat &betas,
                          std::string family, // Pass string from R
                          int num_threads = 1, int m = 1000,
                          bool parallel = true, bool approx = false,
                          bool radial = false, bool show_progress = true,
                          uint32_t base_seed = 31415) {

  // Convert the string to an Enum once right here
  GlmFamily fam = string_to_family(family);
  // This is how we access enums in cpp.
  if (fam == GlmFamily::Unknown) {
    Rcpp::stop("Family not supported in C++ backend.");
  }

  int n_evals = betas.n_rows;
  arma::vec plausabilities(n_evals);
  arma::mat XtX = X.t() * X;

  if (y.n_elem != X.n_rows || mle_coefs.n_elem != X.n_cols ||
      betas.n_cols != X.n_cols) {
    Rcpp::stop("Dimension mismatch: X is %d x %d, y has %d, mle_coefs has %d, "
               "beta_vals has %d",
               X.n_rows, X.n_cols, y.n_elem, mle_coefs.n_elem, betas.n_cols);
  }

  std::atomic<bool> interrupted(false);

  std::atomic<int> progress_count(0);
  // Only update the console every 2% of total iterations to protect performance
  int tick_step = n_evals / 50;
  if (tick_step < 1)
    tick_step = 1;

  // If done via radial or approx, then the progress bar happens elsewhere
  show_progress = (!approx && !radial && show_progress);
  StdoutProgressBar pb;
  Progress p(n_evals, show_progress, pb);

  std::atomic<bool> had_error(false);
  std::string err_msg;

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

      if (interrupted || had_error) {
        continue;
      }
      try {
        arma::vec beta_vals = betas.row(i).t();
        double pl;
        // Ending colon is a part of the case statement.
        switch (fam) {
        case GlmFamily::Gamma: {
          GammaWorkspace gamma_ws;
          arma::vec eta_hat = exp(
              X *
              mle_coefs); // Could put these computations outside even further,
                          // but would decrease code legibility and the amount
                          // of work spent to compute these are minimal.
          pl = glm_gamma_pl_cpp(X, XtX, y, mle_coefs, mle_val, beta_vals, m,
                                approx, false, base_seed, i, eta_hat,
                                Y_sim_thread_workspace, gamma_ws);
          break;
        }

        case GlmFamily::Binomial: {
          LogisticWorkspace ws;
          LogisticPlResult result =
              glm_logis_pl_cpp(X, y, mle_coefs, mle_val, beta_vals, m, approx,
                               false, base_seed, i, Y_sim_thread_workspace, ws);
          pl = result.poss;
          break;
        }

        case GlmFamily::Poisson: {
          PoissonWorkspace ws;
          PoissonPlResult result = glm_poisson_pl_cpp(
              X, y, mle_coefs, mle_val, beta_vals, m, approx, false, base_seed,
              i, Y_sim_thread_workspace, ws);
          // if (result.prop_separated > 0) {
          //   // Note that sim_separated is a percentage of how many sims (for
          //   that
          //   // beta value) went poorly. Don't have a conceptual idea on how
          //   to pass
          //   // this back out.
          //   seperation_issues++;
          // }
          pl = result.poss;
          break;
        }

        case GlmFamily::InverseGaussian: {
          InvGaussWorkspace ws;

          arma::vec mu_hat_invgauss = arma::pow(X * mle_coefs, -0.5);

          pl = glm_invgauss_pl_cpp(X, y, mle_coefs, mle_val, beta_vals, m,
                                   approx, false, base_seed, i, mu_hat_invgauss,
                                   Y_sim_thread_workspace, ws);
          break;
        }

        case GlmFamily::Gaussian: {
          GaussianWorkspace ws;
          double sigma_gauss =
              sqrt(arma::accu((y - X * mle_coefs) % (y - X * mle_coefs)) /
                   (y.n_elem - X.n_cols));
          pl = glm_gaussian_pl_cpp(X, y, mle_coefs, mle_val, beta_vals, m,
                                   approx, false, base_seed, i, sigma_gauss,
                                   Y_sim_thread_workspace, ws);
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
      } catch (const std::exception &e) {
#pragma omp critical(glim_err)
        {
          if (!had_error) {
            err_msg = e.what();
            had_error = true;
          }
        }
      } catch (...) {
#pragma omp critical(glim_err)
        {
          if (!had_error) {
            err_msg = "unknown C++ exception";
            had_error = true;
          }
        }
      }
    }
  }
  if (had_error) {
    Rcpp::stop("glim: failed evaluating grid point: %s", err_msg);
  }

  // We are now passed the OMP section

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

  // TODO maybe a seperation check? Or maybe can be done in R

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

      double val1 =
          fit_glm_omp_cpp(X, y, mle, mle_val, MPlus.t(), family, 1, m, parallel,
                          true, false, false, seed_plus)(0, 0);
      double val2 =
          fit_glm_omp_cpp(X, y, mle, mle_val, MMinus.t(), family, 1, m,
                          parallel, true, false, false, seed_minus)(0, 0);
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

// [[Rcpp::export]]
arma::mat radial_code(int num_samps, arma::mat &X, arma::vec &y, double mle_val,
                      arma::vec &mle_coefs, arma::mat &eig_vecs,
                      arma::vec &eig_vals, std::string family,
                      double dispersion, int m, double tol, int max_it,
                      double a_val, double b_val, int num_omp_threads,
                      int base_seed = 0) {
  tol = std::max(tol, 2.0 / m);
  int d = X.n_cols;
  arma::mat sampled_betas(mle_coefs.n_elem, num_samps);

  arma::mat XtX = X.t() * X; // Precompute here, not inside the loop!
  arma::mat Y_sim(X.n_rows, m);
  arma::vec eta_hat = X * mle_coefs;

  // Family-specific precomputations
  arma::vec p_hat;

  arma::vec eta_hat_gamma;

  double sigma_gauss = 0.0;

  arma::vec mu_hat_invgauss;

  // 1. Parse the family and precompute heavy math ONCE (safely on the main
  // thread)
  GlmFamily fam = string_to_family(family);
  if (fam == GlmFamily::Unknown) {
    Rcpp::stop("Family not supported in C++ backend.");
  }

  // --- Initial Separation Checks ---
  // Pass base_seed and an arbitrary eval_index (e.g., 0) for these checks
  // TODO put in R
  if (fam == GlmFamily::Binomial) {
    eta_hat = X * mle_coefs;
    p_hat = 1.0 / (1.0 + arma::exp(-eta_hat));
  } else if (fam == GlmFamily::Poisson) {
    //
  } else if (fam == GlmFamily::Gamma) {
    // Precompute for Core Engine
    eta_hat_gamma = eta_hat;
  } else if (fam == GlmFamily::Gaussian) {
    // Precompute for Core Engine
    sigma_gauss = std::sqrt(arma::accu((y - eta_hat) % (y - eta_hat)) /
                            (y.n_elem - mle_coefs.n_elem));
  } else if (fam == GlmFamily::InverseGaussian) {
    // Precompute for Core Engine
    mu_hat_invgauss = arma::pow(eta_hat, -0.5);
  }
  // Since R is doing all of the randomization, the base_seed shouldn't be
  // needed anymore
  Rcpp::NumericVector unif_alphas = Rcpp::runif(num_samps, 1e-12, 1.0 - 1e-12);
  Rcpp::NumericVector q_vals = Rcpp::qchisq(1.0 - unif_alphas, d);
  Rcpp::NumericVector norm_draws = Rcpp::rnorm(num_samps * d, 0.0, 1.0);

  // Gets the pointers to the data, rather than using a pointer to SEXP
  // (representation of Rcpp::NumericVector). Allows us to directly work
  // with it as a C list.
  double *p_unif_alphas = REAL(unif_alphas);
  double *p_q_vals = REAL(q_vals);

  // Map the Rcpp vector to an Armadillo matrix (d rows, num_samps cols)
  // without copying memory
  arma::mat Z(norm_draws.begin(), d, num_samps, false);
  arma::mat XW_workspace(y.n_rows, d);
  arma::vec eta_workspace(d);
  arma::vec p_workspace(d);
  arma::vec w_workspace(d);
  // 1/sqrt(lambda_k): maps a direction in the whitened space back to beta space
  // (V * Lambda^{-1/2}). The floor guards against zero/negative eigenvalues.
  const arma::vec inv_sqrt_vals =
      1.0 / arma::sqrt(arma::clamp(eig_vals, 1e-12, arma::datum::inf));
  arma::mat X_pinv;
  if (fam == GlmFamily::Gaussian) {
    X_pinv = arma::pinv(X); // once, not m times per evaluation
  }

  StdoutProgressBar pb;
  Progress p(num_samps, true, pb);

  std::atomic<bool> interrupted(false);
  // Per-sample diagnostics. Plain int/double (NOT std::vector<bool>, which
  // packs
  // bits and is not safe to write from different threads).
  std::vector<int> diag_iters(num_samps, 0);
  std::vector<char> diag_converged(num_samps, 0);
  std::vector<double> diag_first_g(
      num_samps, 0.0); // g at iteration 1 = quality of the start
  std::vector<double> diag_secs(num_samps, 0.0);

  // --- The Parallel Loop ---
#pragma omp parallel num_threads(num_omp_threads)
  {
    // Allocate workspace ONCE per thread
    // To not have dynamic allocation of memory each time through the while loop
    arma::mat Y_sim_thread_workspace(X.n_rows, m);
    arma::vec beta_proposal(d);
    arma::vec u(d);
    arma::vec dir(d);
    LogisticWorkspace logis_ws;
    GammaWorkspace gamma_ws;
    PoissonWorkspace pois_ws;
    InvGaussWorkspace ig_ws;
    GaussianWorkspace gauss_ws;
    if (fam == GlmFamily::Poisson) {
      pois_ws = PoissonWorkspace(X.n_rows, d);
    }
    if (fam == GlmFamily::InverseGaussian) {
      ig_ws = InvGaussWorkspace(X.n_rows, d);
    }
    if (fam == GlmFamily::Gaussian) {
      gauss_ws = GaussianWorkspace(X.n_rows, d);
    }
    if (fam == GlmFamily::Binomial) {
      logis_ws = LogisticWorkspace(X.n_rows, d);
    }
    if (fam == GlmFamily::Gamma) {
      gamma_ws = GammaWorkspace(X.n_rows, d);
    }

    int thread_local_completed = 0;
// Divide the loop among the threads
#pragma omp for schedule(dynamic)
    for (int j = 0; j < num_samps; j++) {
#ifdef _OPENMP
      if (omp_get_thread_num() == 0) {
        if (Progress::check_abort()) {
          interrupted = true;
        }
      }
#endif
      if (interrupted) {
        continue;
      }

      // Can keep the allocation here, since not in the while() loop.
      double val1;
      double g_xi;
      double q_val = p_q_vals[j];
      double alpha_val = p_unif_alphas[j];

      // Extract col j as our random normal vector
      u = Z.col(j);
      const double u_norm = arma::norm(u, 2);

      // Direction: u/||u|| is uniform on the sphere in the whitened space. Map
      // it back to beta space through V * Lambda^{-1/2} (same construction as
      // `spatial_dir` in glim_inner_prob_approx_samples), then normalise.
      dir = eig_vecs * (u % inv_sqrt_vals);
      const double dir_len = arma::norm(dir, 2); // = ||V Lambda^-1/2 u||
      dir /= dir_len;

      // Starting radius: where the ray mle + r*dir meets the asymptotic
      // contour (b - mle)' J (b - mle) / dispersion = q.
      //   dir' J dir = ||u||^2 / dir_len^2   =>   r0 = sqrt(disp*q) * dir_len /
      //   ||u||
      const double r0 = std::sqrt(dispersion * q_val) * dir_len / u_norm;
      double starting_xi = 2.0 * std::log(r0);
      int it = 1;
      bool approx = false;

      double prev_xi = 0.0;
      double prev_g = 0.0;

#ifdef _OPENMP
      const double t_start = omp_get_wtime();
#else
      const double t_start = 0.0;
#endif

      // Draw the uniforms once per sample, outside the Robbins-Monro loop.
      // Every iteration then re-thresholds the same U, so the possibility is a
      // deterministic function of beta (needed for the stochastic approximation
      // to settle) and we skip n*m RNG calls per iteration.
      if (fam == GlmFamily::Binomial) {
        fill_uniforms(Y_sim_thread_workspace,
                      static_cast<uint64_t>(base_seed) * 1000003ULL +
                          static_cast<uint64_t>(j),
                      m, false);
      }

      while (true) {
        double mult = std::exp(starting_xi / 2.0);
        for (int i = 0; i < d; ++i) {
          beta_proposal[i] = mle_coefs[i] + dir[i] * mult;
        }
        val1 = 0.0;

        // 3. Pass `base_seed` and iteration index `j` to inner GLM evaluators
        switch (fam) {
        case GlmFamily::Binomial: {
          // Uniforms were drawn above; only the threshold p(beta) changes here.
          // (The old call passed mle_val as `eval_index` by mistake.)
          val1 = glm_logis_pl_from_uniforms(X, y, mle_val, beta_proposal,
                                            Y_sim_thread_workspace, m, false,
                                            logis_ws)
                     .poss;
          break;
        }
        case GlmFamily::Gamma: {
          val1 = glm_gamma_pl_cpp(X, XtX, y, mle_coefs, mle_val, beta_proposal,
                                  m, approx, true, j, 0, eta_hat_gamma,
                                  Y_sim_thread_workspace, gamma_ws);
          break;
        }
        case GlmFamily::Poisson: {
          PoissonPlResult result = glm_poisson_pl_cpp(
              X, y, mle_coefs, mle_val, beta_proposal, m, approx, false, j, 1,
              Y_sim_thread_workspace, pois_ws);
          val1 = result.poss;
          break;
        }
        case GlmFamily::InverseGaussian: {
          val1 = glm_invgauss_pl_cpp(X, y, mle_coefs, mle_val, beta_proposal, m,
                                     approx, true, j, 0, mu_hat_invgauss,
                                     Y_sim_thread_workspace, ig_ws);
          break;
        }
        case GlmFamily::Gaussian: {
          val1 = glm_gaussian_pl_cpp(X, y, mle_coefs, mle_val, beta_proposal, m,
                                     approx, true, base_seed, j, sigma_gauss,
                                     Y_sim_thread_workspace, gauss_ws);
          break;
        }
        default:
          break;
        }
        g_xi = val1 - alpha_val;
        if (it == 1)
          diag_first_g[j] = g_xi;

        if ((std::abs(g_xi) <= tol) || (it >= max_it)) {
          sampled_betas.col(j) = beta_proposal;
          diag_iters[j] = it;
          diag_converged[j] = (std::abs(g_xi) <= tol);
#ifdef _OPENMP
          diag_secs[j] = omp_get_wtime() - t_start;
#endif
          break;
        } else {

          double step = 0.0;
          int update_method = 1;

          if (update_method == 1) {
            // -----------------------------------------------------
            // SECANT METHOD
            // -----------------------------------------------------
            if (it == 1) {
              step = w(a_val, b_val, it) * g_xi;
            } else {
              double denom = g_xi - prev_g;
              if (std::abs(denom) < 1e-8) {
                // Denominator too close to zero (e.g., deterministic flat
                // region)
                step = w(a_val, b_val, it) * g_xi;
              } else {
                step = -g_xi * (starting_xi - prev_xi) / denom;
                // Clamp step to avoid wild jumps from small denominators
                step = std::max(-2.0, std::min(2.0, step));
              }
            }
            prev_xi = starting_xi;
            prev_g = g_xi;
          } else if (update_method == 2) {
            // -----------------------------------------------------
            // CALIBRATED CHI-SQUARE METHOD
            // -----------------------------------------------------
            double x = std::exp(starting_xi) / dispersion;
            // Derivative of Possibility: -f_chi2(x) * x
            double slope = -R::dchisq(x, d, 0) * x;

            if (std::abs(slope) < 1e-8) {
              // Flat tail/extreme value fallback
              step = w(a_val, b_val, it) * g_xi;
            } else {
              step = -g_xi / slope;
              // Clamp
              step = std::max(-2.0, std::min(2.0, step));
            }
          } else {
            // -----------------------------------------------------
            // ORIGINAL ROBBINS-MONRO
            // -----------------------------------------------------
            step = w(a_val, b_val, it) * g_xi;
          }

          starting_xi = starting_xi + step;
          it++;
        }
      }
      thread_local_completed++;
      if (thread_local_completed >= 20) {
#pragma omp critical(progress_update)
        {
          if (Progress::check_abort()) {
            interrupted = true;
          }
          p.increment(thread_local_completed);
        }
        thread_local_completed = 0;
      }
    }

    // Flush any remaining completed iterations per thread at loop completion
    if (thread_local_completed > 0) {
#pragma omp critical(progress_update)
      {
        p.increment(thread_local_completed);
      }
    }
  }

  if (interrupted) {
    Rcpp::stop("Computation interrupted by user.");
  }

  {
    int n_hit_max = 0;
    long total_iters = 0;
    double sum_abs_first_g = 0.0, sum_secs = 0.0, max_secs = 0.0;
    std::vector<int> hist(max_it + 1, 0);
    for (int j = 0; j < num_samps - 1;
         ++j) { // last column is overwritten with the MLE
      total_iters += diag_iters[j];
      if (!diag_converged[j])
        ++n_hit_max;
      if (diag_iters[j] <= max_it)
        ++hist[diag_iters[j]];
      sum_abs_first_g += std::abs(diag_first_g[j]);
      sum_secs += diag_secs[j];
      max_secs = std::max(max_secs, diag_secs[j]);
    }
    // Rcpp::Rcout << "\n--- radial diagnostics ---\n"
    //             << "samples hitting max_it (" << max_it << "): " << n_hit_max
    //             << " / " << N << " (" << 100.0 * n_hit_max / N << "%)\n"
    //             << "mean iterations per sample: " << (double)total_iters / N
    //             << "\n"
    //             << "mean |g| at iteration 1 (start quality): "
    //             << sum_abs_first_g / N << "\n"
    //             << "total possibility evaluations: " << total_iters
    //             << " (each = " << m << " logistic fits)\n"
    //             << "per-sample seconds: mean " << sum_secs / N << ", max "
    //             << max_secs << "\n"
    //             << "iteration histogram:\n";
    // for (int k = 1; k <= max_it; ++k)
    //   if (hist[k] > 0)
    //     Rcpp::Rcout << "  " << k << " iters: " << hist[k] << "\n";
  }

  sampled_betas.col(num_samps - 1) = mle_coefs;
  return sampled_betas;
}