#ifndef INCLUDE_RICCATI_STEP_HPP
#define INCLUDE_RICCATI_STEP_HPP

#include <riccati/chebyshev.hpp>
#include <Eigen/Dense>
#include <complex>
#include <cmath>
#include <tuple>

namespace riccati {

/**
 * @brief Performs a single Chebyshev step with adaptive node count for solving
 * differential equations.
 *
 * This function advances the solution of a differential equation from `x = x0`
 * by a step of size `h`, starting from the initial conditions `y(x0) = y0` and
 * `y'(x0) = dy0`. It employs a Chebyshev spectral method with an adaptive
 * number of nodes, starting with `info.nini` nodes and doubling the count in
 * each iteration until the relative accuracy `epsres` is achieved or the number
 * of nodes exceeds `info.nmax`. The relative error is assessed by comparing the
 * predicted values of the dependent variable at the end of the step for the
 * current and previous iterations. If the desired accuracy isn't met with the
 * maximum number of nodes, step size `h` may need to be reduced, `info.nmax`
 * increased, or a different numerical method considered.
 *
 * @param info SolverInfo object - Object containing pre-computed information
 * for the solver, like differentiation matrices and Chebyshev nodes, and
 * methods for evaluating functions `w(x)` and `g(x)` over the interval `[x0,
 * x0+h]`.
 * @param x0 float - The starting value of the independent variable.
 * @param h float - Step size for the spectral method.
 * @param y0 complex - Initial value of the dependent variable at `x0`.
 * @param dy0 complex - Initial derivative of the dependent variable at `x0`.
 * @param epsres float - Tolerance for the relative accuracy of the Chebyshev
 * step.
 * @return std::tuple<std::complex<double>, std::complex<double>, float, int> -
 * A tuple containing:
 *         1. std::complex<double> - Value of the dependent variable at the end
 * of the step, at `x = x0 + h`.
 *         2. std::complex<double> - Value of the derivative of the dependent
 * variable at the end of the step, at `x = x0 + h`.
 *         3. float - The (absolute) value of the relative difference of the
 * dependent variable at the end of the step as predicted in the last and the
 * previous iteration.
 *         4. int - Flag indicating success (`1`) if the asymptotic series has
 * reached the desired `epsres` residual, `0` otherwise.
 */
template <typename SolverInfo, typename Scalar, typename YScalar>
inline auto nonosc_step(SolverInfo &&info, Scalar x0, Scalar h, YScalar y0,
                        YScalar dy0, Scalar epsres) {
  using complex_t = std::complex<Scalar>;

  Scalar maxerr = 10 * epsres;
  auto N = info.nini_;
  auto Nmax = info.nmax_;
  auto cheby = spectral_chebyshev(info, x0, h, y0, dy0, 0);
  auto yprev = std::get<0>(cheby);
  auto dyprev = std::get<1>(cheby);
  auto xprev = std::get<2>(cheby);
  int iter = 0;
  while (maxerr > epsres) {
    iter++;
    N *= 2;
    if (N > Nmax) {
      return std::make_tuple(false, complex_t(0.0, 0.0), complex_t(0.0, 0.0),
                             maxerr, yprev, dyprev, iter);
    }
    auto cheb_num = static_cast<int>(std::log2(N / info.nini_));
    auto cheby2 = spectral_chebyshev(info, x0, h, y0, dy0, cheb_num);
    auto y = std::get<0>(std::move(cheby2));
    auto dy = std::get<1>(std::move(cheby2));
    auto x = std::get<2>(std::move(cheby2));
    maxerr = std::abs((yprev(0) - y(0)) / y(0));
    if (std::isnan(maxerr)) {
      maxerr = std::numeric_limits<Scalar>::infinity();
    }
    yprev = std::move(y);
    dyprev = std::move(dy);
    xprev = std::move(x);
  }
  return std::make_tuple(true, yprev(0), dyprev(0), maxerr, yprev, dyprev,
                         iter);
}

/**
 * @brief Performs a single Riccati step for solving differential equations with
 * oscillatory behavior.
 *
 * This function advances the solution from `x0` by `h`, starting from the
 * initial conditions `y(x0) = y0` and `y'(x0) = dy0`, using an asymptotic
 * expansion approach tailored for Riccati-type differential equations. It
 * iteratively increases the order of the asymptotic series used for the Riccati
 * equation until a residual of `epsres` is reached or the residual stops
 * decreasing, indicating that the asymptotic series cannot approximate the
 * solution with the required accuracy over the given interval. In such cases,
 * it is recommended to reduce the step size `h` or consider an alternative
 * approximation method. The function also computes the total phase change of
 * the dependent variable over the step.
 *
 * @tparam SolverInfo A riccati solver like object
 * @tparam OmegaVec An Eigen vector
 * @tparam GammaVec An Eigen vector
 * @tparam Scalar A scalar type for the x values
 * @tparam YScalar A scalar type for the y values
 * @param info SolverInfo object - Object containing pre-computed information
 * for the solver, like differentiation matrices and methods for evaluating
 * functions `w(x)` and `g(x)` over the interval `[x0, x0+h]`.
 * @param omega_s Vector of the frequency function `w(x)` evaluated at the
 * Chebyshev nodes over the interval `[x0, x0+h]`.
 * @param gamma_s Vector of the friction function `g(x)` evaluated at the
 * Chebyshev nodes over the interval `[x0, x0+h]`.
 * @param x0 float - The starting value of the independent variable.
 * @param h float - Step size for the Riccati step.
 * @param y0 complex - Initial value of the dependent variable at `x0`.
 * @param dy0 complex - Initial derivative of the dependent variable at `x0`.
 * @param epsres float - Tolerance for the relative accuracy of the Riccati
 * step.
 * @return std::tuple<std::complex<double>, std::complex<double>, float, int,
 * std::complex<double>> - A tuple containing:
 *         1. std::complex<double> - Value of the dependent variable at the end
 * of the step, at `x = x0 + h`.
 *         2. std::complex<double> - Value of the derivative of the dependent
 * variable at the end of the step, at `x = x0 + h`.
 *         3. float - Maximum value of the residual (after the final iteration
 * of the asymptotic approximation) over the Chebyshev nodes across the
 * interval.
 *         4. int - Flag indicating success (`1`) if the asymptotic series has
 * reached the desired `epsres` residual, `0` otherwise.
 *         5. std::complex<double> - Total phase change (not mod 2π) of the
 * dependent variable over the step.
 * @warning This function relies on `info.wn`, `info.gn` being set correctly for
 * a step of size `h`. If `solve()` is calling this function, that is taken care
 * of automatically, but it needs to be done manually otherwise.
 */
template <bool DenseOut, typename SolverInfo, typename OmegaVec,
          typename GammaVec, typename Scalar, typename YScalar>
inline auto osc_step(SolverInfo &&info, OmegaVec &&omega_s, GammaVec &&gamma_s,
                     Scalar x0, Scalar h, YScalar y0, YScalar dy0,
                     Scalar epsres) {
  using complex_t = std::complex<Scalar>;
  using vectorc_t = vector_t<complex_t>;
  print("h",h);
  print("x0", x0);
  print("y0", y0);
  print("dy0", dy0);
  print("epsres", epsres);
  print("omega_s", omega_s);
  print("gamma_s", gamma_s);
  bool success = true;
  auto &&Dn = info.Dn();
  auto y = eval(info.alloc_, complex_t(0.0, 1.0) * omega_s);
  auto delta = [&](const auto &r, const auto &y) {
    return (-r.array() / (Scalar{2.0} * (y.array() + gamma_s.array())))
        .matrix()
        .eval();
  };
  auto R = [&](const auto &d) {
    return Scalar{2.0} / h * (Dn * d) + d.array().square().matrix();
  };
  auto Ry
      = (complex_t(0.0, 1.0) * Scalar{2.0}
         * (Scalar{1.0} / h * (Dn * omega_s) + gamma_s.cwiseProduct(omega_s)))
            .eval();
  Scalar maxerr = Ry.array().abs().maxCoeff();
  print("maxerr", maxerr);
  print("Dn", Dn);
  print("y", y);
  print("Ry", Ry);
  arena_matrix<vectorc_t> deltay(info.alloc_, Ry.size(), 1);
  Scalar prev_err = std::numeric_limits<Scalar>::infinity();
  while (maxerr > epsres) {
    deltay = delta(Ry, y);
    y += deltay;
    Ry = R(deltay);
    maxerr = Ry.array().abs().maxCoeff();
    if (maxerr >= (Scalar{2.0} * prev_err)) {
      success = false;
      break;
    }
    prev_err = maxerr;
  }
  deltay = delta(Ry, y);
  y += deltay;
  Ry = R(deltay);
  maxerr = Ry.array().abs().maxCoeff();
  if (maxerr >= (Scalar{2.0} * prev_err)) {
    success = false;
  }
  prev_err = maxerr;
  print("maxerr", maxerr);
  print("y_post", y.eval());
  print("Ry_post", Ry);
  if constexpr (DenseOut) {
    auto u1
        = eval(info.alloc_, h / Scalar{2.0} * (info.integration_matrix_ * y));
    auto f1 = eval(info.alloc_, (u1).array().exp().matrix());
    auto f2 = eval(info.alloc_, f1.conjugate());
    auto du2 = eval(info.alloc_, y.conjugate());
    auto ap_top = (dy0 - y0 * du2(du2.size() - 1));
    auto ap_bottom = (y(y.size() - 1) - du2(du2.size() - 1));
    auto ap = ap_top / ap_bottom;
    auto am = (dy0 - y0 * y(y.size() - 1))
              / (du2(du2.size() - 1) - y(y.size() - 1));
    auto y1 = eval(info.alloc_, ap * f1 + am * f2);
    auto dy1 = eval(info.alloc_,
                    ap * y.cwiseProduct(f1) + am * du2.cwiseProduct(f2));
    Scalar phase = std::imag(f1(0));
    return std::make_tuple(success, y1(0), dy1(0), maxerr, phase, u1, y,
                           std::make_pair(ap, am));
  } else {
    auto u1 = (h / Scalar{2.0} * (info.quadwts_.dot(y)));
    print("u1", u1);
    complex_t f1 = std::exp(u1);
    auto f2 = std::conj(f1);
    auto du2 = y.conjugate().eval();
    auto ap_top = (dy0 - y0 * du2(du2.size() - 1));
    auto ap_bottom = (y(y.size() - 1) - du2(du2.size() - 1));
    auto ap = ap_top / ap_bottom;
    auto am = (dy0 - y0 * y(y.size() - 1))
              / (du2(du2.size() - 1) - y(y.size() - 1));
    auto y1 = (ap * f1 + am * f2);
    auto dy1 = (ap * y * f1 + am * du2 * f2).eval();
    print("quadwts", info.quadwts_);
    print("ap", ap);
    print("am", am);
    print("f1", f1);
    print("f2", f2);
    print("du2", du2);
    print("y1", y1);
    print("dy1", dy1);
    Scalar phase = std::imag(f1);
    return std::make_tuple(success, y1, dy1(0), maxerr, phase,
                           arena_matrix<vectorc_t>(info.alloc_, y.size()),
                           arena_matrix<vectorc_t>(info.alloc_, y.size()),
                           std::make_pair(ap, am));
  }
}

/**
 * @brief Solves the differential equation y'' + 2gy' + w^2y = 0 over a given
 * interval.
 *
 * This function solves the differential equation on the interval (xi, xf),
 * starting from the initial conditions y(xi) = yi and y'(xi) = dyi. It keeps
 * the residual of the ODE below eps, and returns an interpolated solution
 * (dense output) at the points specified in x_eval.
 *
 * @tparam SolverInfo Type of the solver info object containing differentiation
 * matrices, etc.
 * @tparam Scalar Numeric scalar type, typically float or double.
 * @tparam Vec Type of the vector for dense output values, should match Scalar
 * type.
 *
 * @tparam SolverInfo Type of the solver info object containing differentiation
 * matrices, etc.
 * @tparam Scalar Numeric scalar type, typically float or double.
 * @tparam Vec Type of the vector for dense output values.
 * @param[in] info SolverInfo object containing necessary information for the
 * solver.
 * @param[in] xi Starting value of the independent variable.
 * @param[in] xf Ending value of the independent variable.
 * @param[in] yi Initial value of the dependent variable at xi.
 * @param[in] dyi Initial derivative of the dependent variable at xi.
 * @param[in] eps Relative tolerance for the local error of both Riccati and
 * Chebyshev type steps.
 * @param[in] epsilon_h Relative tolerance for choosing the stepsize of Riccati
 * steps.
 * @param[in] init_stepsize initial stepsize for the integration
 * @param[in] x_eval List of x-values where the solution is to be interpolated
 * (dense output) and returned.
 * @param[in] hard_stop If true, forces the solver to have a potentially smaller
 * last stepsize to stop exactly at xf.
 * @return A tuple containing multiple elements representing the results of the
 * ODE solving process: 0. std::vector<Scalar>: A vector containing the x-values
 * at which the solution was evaluated or interpolated. These values correspond
 * to the points in the interval [xi, xf] and include the points specified in
 * x_eval if dense output was requested.
 * 1. std::vector<std::complex<Scalar>>: A vector of complex numbers
 * representing the solution y(x) of the differential equation at each x-value
 * from the corresponding vector of x-values.
 * 2. std::vector<std::complex<Scalar>>: A vector of complex numbers
 * representing the derivative of the solution, y'(x), at each x-value from the
 * corresponding vector of x-values.
 * 3. std::vector<int>: A vector indicating the success status of the solver at
 * each step. Each element corresponds to a step in the solver process, where a
 * value of 1 indicates success, and 0 indicates failure.
 * 4. std::vector<int>: A vector indicating the type of step taken at each point
 * in the solution process. Each element corresponds to a step in the solver
 * process, where a value of 1 indicates an oscillatory step and 0 indicates a
 *   non-oscillatory step.
 * 5. std::vector<Scalar>: A vector containing the phase angle at each step of
 * the solution process if relevant. This is especially applicable for
 * oscillatory solutions and may not be used for all types of differential
 * equations.
 * 6. @ref Eigen::Matrix<std::complex<Scalar>, -1, 1>: A vector containing the
 * interpolated solution at the specified `x_eval` The function returns these
 * vectors encapsulated in a standard tuple, providing comprehensive information
 * about the solution process, including where the solution was evaluated, the
 * values and derivatives of the solution at those points, success status of
 * each step, type of each step, and phase angles where applicable.
 */
template <typename SolverInfo, typename Scalar, typename Vec>
inline auto step(SolverInfo &info, Scalar xi, Scalar xf,
                 std::complex<Scalar> yi, std::complex<Scalar> dyi, Scalar eps,
                 Scalar epsilon_h, Scalar init_stepsize, Vec &&x_eval,
                 bool hard_stop = false) {
  using vectord_t = vector_t<Scalar>;
  using complex_t = std::complex<Scalar>;
  using vectorc_t = vector_t<complex_t>;
  Scalar direction = init_stepsize > 0 ? 1 : -1;
  if (init_stepsize * (xf - xi) < 0) {
    throw std::domain_error(
        "Direction of integration does not match stepsize sign,"
        " adjusting it so that integration happens from xi to xf.");
  }
  // Check that yeval and x_eval are right size
  constexpr bool dense_output = compile_size_v<Vec> != 0;
  if constexpr (dense_output) {
    if (!x_eval.size()) {
      throw std::domain_error("Dense output requested but x_eval is size 0!");
    }
    // TODO: Better error messages
    auto x_eval_max = (direction * x_eval.maxCoeff());
    auto x_eval_min = (direction * x_eval.minCoeff());
    auto xi_intdir = direction * xi;
    auto xf_intdir = direction * xf;
    const bool high_range_err = xf_intdir < x_eval_max;
    const bool low_range_err = xi_intdir > x_eval_min;
    if (high_range_err || low_range_err) {
      if (high_range_err && low_range_err) {
        throw std::out_of_range(
            std::string{"The max and min of the output points ("}
            + std::to_string(x_eval_min) + std::string{", "}
            + std::to_string(x_eval_max)
            + ") lie outside the high and low of the integration range ("
            + std::to_string(xi_intdir) + std::string{", "}
            + std::to_string(xf_intdir) + ")!");
      }
      if (high_range_err) {
        throw std::out_of_range(
            std::string{"The max of the output points ("}
            + std::to_string(x_eval_max)
            + ") lies outside the high of the integration range ("
            + std::to_string(xf_intdir) + ")!");
      }
      if (low_range_err) {
        throw std::out_of_range(
            std::string{"The min of the output points ("}
            + std::to_string(x_eval_min)
            + ") lies outside the low of the integration range ("
            + std::to_string(xi_intdir) + ")!");
      }
    }
  }

  // Initialize vectors for storing results
  Scalar xs;
  complex_t ys;
  complex_t dys;
  int successes;
  int steptypes;
  Scalar phases;
  Eigen::Matrix<complex_t, -1, 1> yeval;   //(x_eval.size());
  Eigen::Matrix<complex_t, -1, 1> dyeval;  //(x_eval.size());

  complex_t y = yi;
  complex_t dy = dyi;
  complex_t yprev = y;
  complex_t dyprev = dy;
  auto scale_xi = scale(info.xp().array(), xi, init_stepsize).eval();
  auto omega_is = omega(info, scale_xi).eval();
  auto gamma_is = gamma(info, scale_xi).eval();
  Scalar wi = omega_is.mean();
  Scalar gi = gamma_is.mean();
  Scalar dwi = (2.0 / init_stepsize * (info.Dn() * omega_is)).mean();
  Scalar dgi = (2.0 / init_stepsize * (info.Dn() * gamma_is)).mean();
  Scalar hslo_ini = direction
                    * std::min(static_cast<Scalar>(1e8),
                               static_cast<Scalar>(std::abs(1.0 / wi)));
  Scalar hosc_ini
      = direction
        * std::min(std::min(static_cast<Scalar>(1e8),
                            static_cast<Scalar>(std::abs(wi / dwi))),
                   std::abs(gi / dgi));

  if (hard_stop) {
    if (direction * (xi + hosc_ini) > direction * xf) {
      hosc_ini = xf - xi;
    }
    if (direction * (xi + hslo_ini) > direction * xf) {
      hslo_ini = xf - xi;
    }
  }
  auto hslo = choose_nonosc_stepsize(info, xi, hslo_ini, Scalar(0.2));
  // o and g written here
  auto osc_step_tup = choose_osc_stepsize(info, xi, hosc_ini, epsilon_h);
  auto hosc = std::get<0>(osc_step_tup);
  // NOTE: Calling choose_osc_stepsize will update these values
  auto &&omega_n = std::get<1>(osc_step_tup);
  auto &&gamma_n = std::get<2>(osc_step_tup);
  Scalar xcurrent = xi;
  Scalar wnext = wi;
  using matrixc_t = matrix_t<complex_t>;
  matrixc_t y_eval;
  matrixc_t dy_eval;
  arena_matrix<vectorc_t> un(info.alloc_, omega_n.size(), 1);
  arena_matrix<vectorc_t> d_un(info.alloc_, omega_n.size(), 1);
  std::pair<complex_t, complex_t> a_pair;
  Scalar phase{0.0};
  bool success = false;
  bool steptype = true;
  Scalar err;
  int cheb_N = 0;
  if ((direction * hosc > direction * hslo * 5.0)
      && (direction * hosc * wnext / (2.0 * pi<Scalar>()) > 1.0)) {
    if (hard_stop) {
      if (direction * (xcurrent + hosc) > direction * xf) {
        hosc = xf - xcurrent;
        auto xp_scaled = scale(info.xp().array(), xcurrent, hosc).eval();
        omega_n = omega(info, xp_scaled);
        gamma_n = gamma(info, xp_scaled);
      }
      if (direction * (xcurrent + hslo) > direction * xf) {
        hslo = xf - xcurrent;
      }
    }
    // o and g read here
    std::tie(success, y, dy, err, phase, un, d_un, a_pair)
        = osc_step<dense_output>(info, omega_n, gamma_n, xcurrent, hosc, yprev,
                                 dyprev, eps);
    steptype = 1;
  }
  while (!success) {
    std::tie(success, y, dy, err, y_eval, dy_eval, cheb_N)
        = nonosc_step(info, xcurrent, hslo, yprev, dyprev, eps);
    steptype = 0;
    if (!success) {
      hslo *= Scalar{0.5};
    }
    if (direction * hslo < 1e-16) {
      throw std::domain_error("Stepsize became to small error");
    }
  }
  auto h = steptype ? hosc : hslo;
  if constexpr (dense_output) {
    Eigen::Index dense_size = 0;
    Eigen::Index dense_start = 0;
    // Assuming x_eval is sorted we just want start and size
    std::tie(dense_start, dense_size)
        = get_slice(x_eval, direction * xcurrent, direction * (xcurrent + h));
    yeval = Eigen::Matrix<complex_t, -1, 1>(dense_size);
    dyeval = Eigen::Matrix<complex_t, -1, 1>(dense_size);
    if (dense_size != 0) {
      auto x_eval_map
          = Eigen::Map<vectord_t>(x_eval.data() + dense_start, dense_size);
      auto y_eval_map
          = Eigen::Map<vectorc_t>(yeval.data() + dense_start, dense_size);
      auto dy_eval_map
          = Eigen::Map<vectorc_t>(dyeval.data() + dense_start, dense_size);
      if (steptype) {
        auto x_eval_scaled
            = eval(info.alloc_,
                   (2.0 / h * (x_eval_map.array() - xcurrent) - 1.0).matrix());
        auto Linterp = interpolate(info.xn(), x_eval_scaled, info.alloc_);
        auto fdense = eval(info.alloc_, (Linterp * un).array().exp().matrix());
        y_eval_map = a_pair.first * fdense + a_pair.second * fdense.conjugate();
        auto du_dense = eval(info.alloc_, (Linterp * d_un));
        dy_eval_map
            = a_pair.first * (du_dense.array() * fdense.array())
              + a_pair.second * (du_dense.array() * fdense.array()).conjugate();
      } else {
        auto xc_scaled = eval(
            info.alloc_,
            scale(std::get<2>(info.chebyshev_[cheb_N]), xcurrent, h).matrix());
        auto Linterp = interpolate(xc_scaled, x_eval_map, info.alloc_);
        y_eval_map = Linterp * y_eval;
      }
    }
  }
  // Finish appending and ending conditions
  ys = y;
  dys = dy;
  xs = xcurrent + h;
  phases = phase;
  steptypes = steptype;
  successes = success;
  Scalar dwnext;
  Scalar gnext;
  Scalar dgnext;
  if (steptype) {
    wnext = omega_n[0];
    gnext = gamma_n[0];
    dwnext = 2.0 / h * info.Dn().row(0).dot(omega_n);
    dgnext = 2.0 / h * info.Dn().row(0).dot(gamma_n);
  } else {
    wnext = omega(info, xcurrent + h);
    gnext = gamma(info, xcurrent + h);
    auto xn_scaled = scale(info.xn().array(), xcurrent, h).eval();
    dwnext = 2.0 / h * info.Dn().row(0).dot(omega(info, xn_scaled).matrix());
    dgnext = 2.0 / h * info.Dn().row(0).dot(gamma(info, (xn_scaled).matrix()));
  }
  xcurrent += h;
  if (direction * xcurrent < direction * xf) {
    hslo_ini = direction * std::min(Scalar{1e8}, std::abs(Scalar{1.0} / wnext));
    hosc_ini = direction
               * std::min(std::min(Scalar{1e8}, std::abs(wnext / dwnext)),
                          std::abs(gnext / dgnext));
    if (hard_stop) {
      if (direction * (xcurrent + hosc_ini) > direction * xf) {
        hosc_ini = xf - xcurrent;
      }
      if (direction * (xcurrent + hslo_ini) > direction * xf) {
        hslo_ini = xf - xcurrent;
      }
    }
    // o and g written here
    osc_step_tup = choose_osc_stepsize(info, xcurrent, hosc_ini, epsilon_h);
    hosc = std::get<0>(osc_step_tup);
    hslo = choose_nonosc_stepsize(info, xcurrent, hslo_ini, Scalar{0.2});
    yprev = y;
    dyprev = dy;
  }
  info.alloc_.recover_memory();
  return std::make_tuple(xs, ys, dys, hosc, hslo, successes, phases, steptypes,
                         yeval);
}

}  // namespace riccati

#endif
