
// glim_prof.h  --  lightweight, optional profiling for glim.
// Needs C++17 (inline variables): put `CXX_STD = CXX17` in src/Makevars.
// When disabled (default) the cost is one relaxed atomic load per call.
#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>

namespace glim_prof {

inline std::atomic<bool> enabled{false};
inline std::atomic<uint64_t> call_id{0};

// Phase accumulators, in nanoseconds of *thread time* (sums across threads).
inline std::atomic<uint64_t> ns_sim{0}; // generating simulated responses
inline std::atomic<uint64_t> ns_fit{0}; // refitting the GLM on each replicate
inline std::atomic<uint64_t> ns_ll{0};  // dispersion + log-likelihood work

// Counters
inline std::atomic<uint64_t> n_pl{0};         // plausibility evaluations
inline std::atomic<uint64_t> n_fits{0};       // GLM refits
inline std::atomic<uint64_t> newton_iters{0}; // total solver iterations
inline std::atomic<uint64_t> nonconverged{0}; // fits hitting the iteration cap

inline std::mutex mtx;
inline std::ofstream f_calls, f_imvar;

inline double now() {
  using clk = std::chrono::steady_clock;
  return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}
inline uint64_t now_ns() {
  using clk = std::chrono::steady_clock;
  return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
             clk::now().time_since_epoch())
      .count();
}

inline void write_row(std::ofstream &f, const std::string &line) {
  std::lock_guard<std::mutex> lk(mtx);
  if (f.is_open()) {
    f << line << '\n';
    f.flush();
  }
}

// Tiny CSV row builder
class Row {
  std::ostringstream s_;
  bool first_ = true;

public:
  Row() { s_ << std::setprecision(10); }
  template <class T> Row &add(const T &v) {
    if (!first_)
      s_ << ',';
    first_ = false;
    s_ << v;
    return *this;
  }
  std::string str() const { return s_.str(); }
};

} // namespace glim_prof
