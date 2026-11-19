
// glim_prof.cpp -- R-facing control for the profiler.
// Named prof_cpp_* so the R wrappers glim_prof_start() etc. can be pure R.
#include "glim_prof.h"
#include <Rcpp.h>

// [[Rcpp::export]]
void prof_cpp_start(std::string dir) {
  using namespace glim_prof;
  std::lock_guard<std::mutex> lk(mtx);
  if (f_calls.is_open())
    f_calls.close();
  if (f_imvar.is_open())
    f_imvar.close();
  f_calls.open(dir + "/calls.csv", std::ios::trunc);
  f_imvar.open(dir + "/imvar.csv", std::ios::trunc);
  if (!f_calls.is_open() || !f_imvar.is_open())
    Rcpp::stop("glim_prof: could not open log files in %s", dir);
  f_calls << "id,t_start,n_evals,n,p,m,family,approx,radial,parallel,threads,"
             "outer,inner,wall_s,cpu_s,eff_cores,imbalance,plaus_min,"
             "plaus_max,plaus_mean,n_zero,n_nan,us_per_fit_wall\n";
  f_imvar << "call,d,it,xi_d,val1,val2,g_xi,wall_s\n";
  call_id = 0;
  ns_sim = 0;
  ns_fit = 0;
  ns_ll = 0;
  n_pl = 0;
  n_fits = 0;
  newton_iters = 0;
  nonconverged = 0;
  enabled = true;
}

// [[Rcpp::export]]
void prof_cpp_stop() {
  using namespace glim_prof;
  enabled = false;
  std::lock_guard<std::mutex> lk(mtx);
  if (f_calls.is_open())
    f_calls.close();
  if (f_imvar.is_open())
    f_imvar.close();
}

// [[Rcpp::export]]
Rcpp::List prof_cpp_summary() {
  using namespace glim_prof;
  double fits = (double)n_fits.load();
  return Rcpp::List::create(
      Rcpp::Named("thread_sec_simulate") = ns_sim.load() / 1e9,
      Rcpp::Named("thread_sec_fit") = ns_fit.load() / 1e9,
      Rcpp::Named("thread_sec_loglik") = ns_ll.load() / 1e9,
      Rcpp::Named("n_plausibility_evals") = (double)n_pl.load(),
      Rcpp::Named("n_glm_fits") = fits,
      Rcpp::Named("mean_solver_iters") =
          fits > 0 ? newton_iters.load() / fits : NA_REAL,
      Rcpp::Named("frac_nonconverged") =
          fits > 0 ? nonconverged.load() / fits : NA_REAL);
}
