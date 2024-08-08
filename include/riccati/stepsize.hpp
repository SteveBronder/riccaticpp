#ifndef INCLUDE_RICCATI_STEPSIZE_HPP
#define INCLUDE_RICCATI_STEPSIZE_HPP

#include <riccati/utils.hpp>

namespace riccati {
/**
 * Chooses the stepsize for spectral Chebyshev steps, based on the variation
 * of 1/w, the approximate timescale over which the solution changes. If over
 *  the suggested interval h 1/w changes by a fraction of  \f[\pm epsilon_h\f]
 * or more, the interval is halved, otherwise it's accepted.
 *
 *  @tparam SolverInfo A riccati solver like object
 *  @tparam FloatingPoint A floating point
 *  @param info Solverinfo object which is used to retrieve Solverinfo.xp, the
 * (p+1) Chebyshev nodes used for interpolation to determine the stepsize.
 *  @param x0 Current value of the independent variable.
 *  @param h Initial estimate of the stepsize.
 *  @param epsilon_h Tolerance parameter defining how much 1/w(x) is allowed to
 * change over the course of the step.
 *
 *  @return Refined stepsize over which 1/w(x) does not change by more than
 * epsilon_h/w(x).
 *
 */
template <typename SolverInfo, typename FloatingPoint>
inline FloatingPoint choose_nonosc_stepsize(SolverInfo& info, FloatingPoint x0,
                                            FloatingPoint h,
                                            FloatingPoint epsilon_h) {
  auto ws = omega(info, riccati::scale(info.xp(), x0, h));
  if (ws.maxCoeff() > (1 + epsilon_h) / std::abs(h)) {
    return choose_nonosc_stepsize(info, x0, h / 2.0, epsilon_h);
  } else {
    return h;
  }
}

/**
 * @brief Chooses an appropriate step size for the Riccati step based on the
 * accuracy of Chebyshev interpolation of w(x) and g(x).
 *
 * This function determines an optimal step size `h` over which the functions
 * `w(x)` and `g(x)` can be represented with sufficient accuracy by evaluating
 * their values at `p+1` Chebyshev nodes. It performs interpolation to `p`
 * points halfway between these nodes and compares the interpolated values with
 * the actual values of `w(x)` and `g(x)`. If the largest relative error in `w`
 * or `g` exceeds the tolerance `epsh`, the step size `h` is reduced. This
 * process ensures that the Chebyshev interpolation of `w(x)` and `g(x)` over
 * the step [`x0`, `x0+h`] has a relative error no larger than `epsh`.
 *
 * @param info SolverInfo object - Object containing pre-computed information
 * and methods for evaluating functions `w(x)` and `g(x)`, as well as
 * interpolation matrices and node positions.
 * @param x0 float - The current value of the independent variable.
 * @param h float - The initial estimate of the step size.
 * @param epsilon_h float - Tolerance parameter defining the maximum relative error
 * allowed in the Chebyshev interpolation of `w(x)` and `g(x)` over the proposed
 * step.
 * @return float - The refined step size over which the Chebyshev interpolation
 * of `w(x)` and `g(x)` satisfies the relative error tolerance `epsh`.
 */
template <typename SolverInfo, typename FloatingPoint>
inline auto choose_osc_stepsize(SolverInfo& info, FloatingPoint x0,
                                FloatingPoint h, FloatingPoint epsilon_h) {
  auto t = eval(info.alloc_, riccati::scale(info.xp_interp(), x0, h));
  auto s = eval(info.alloc_, riccati::scale(info.xp(), x0, h));
  // TODO: Use a memory arena for these
  auto ws = omega(info, s).eval();
  auto gs = gamma(info, s).eval();
  auto omega_analytic = eval(info.alloc_, omega(info, t));
  auto omega_estimate = info.L() * ws;
  auto gamma_analytic = eval(info.alloc_, gamma(info, t));
  auto gamma_estimate = info.L() * gs;
  FloatingPoint max_omega_err
      = (((omega_estimate - omega_analytic).array() / omega_analytic.array())
             .abs())
            .maxCoeff();
  FloatingPoint max_gamma_err
      = (((gamma_estimate - gamma_analytic).array() / gamma_analytic.array())
             .abs())
            .maxCoeff();
  FloatingPoint max_err = std::max(max_omega_err, max_gamma_err);
  if (max_err <= epsilon_h) {
    if (info.p_ != info.n_) {
      auto xn_scaled = eval(info.alloc_, riccati::scale(info.xn(), x0, h));
      ws = omega(info, xn_scaled);
      gs = gamma(info, xn_scaled);
    }
    return std::make_tuple(h, ws, gs);
  } else {
    auto h_scaling = std::min(
        0.7, 0.9 * std::pow(epsilon_h / max_err, (1.0 / (info.p_ - 1.0))));
    return choose_osc_stepsize(info, x0, h * h_scaling, epsilon_h);
  }
}

}  // namespace riccati
#endif
