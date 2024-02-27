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
 * @param alloc An allocator for the Eigen objects.
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
template <typename SolverInfo, typename Scalar, typename YScalar,
          typename Allocator>
inline auto nonosc_step(SolverInfo &&info, Scalar x0, Scalar h, YScalar y0,
                        YScalar dy0, Scalar epsres, Allocator &&alloc) {
  using complex_t = std::complex<Scalar>;

  Scalar maxerr = 10 * epsres;
  auto N = info.nini_;
  auto Nmax = info.nmax_;
  auto cheby = spectral_chebyshev(info, x0, h, y0, dy0, 0, alloc);
  auto yprev = std::get<0>(cheby);
  auto dyprev = std::get<1>(cheby);
  auto xprev = std::get<2>(cheby);
  int iter = 0;
  while (maxerr > epsres) {
    iter++;
    N *= 2;
    if (N > Nmax) {
      return std::make_tuple(false, complex_t(0.0, 0.0), complex_t(0.0, 0.0),
                             maxerr, yprev, dyprev);
    }
    auto cheb_num = static_cast<int>(std::log2(N / info.nini_));
    auto cheby2 = spectral_chebyshev(info, x0, h, y0, dy0, cheb_num, alloc);
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
  return std::make_tuple(true, yprev(0), dyprev(0), maxerr, yprev, dyprev);
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
 * @tparam Allocator An allocator for the Eigen objects
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
 * @param alloc An allocator for the Eigen objects.
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
template <typename SolverInfo, typename OmegaVec, typename GammaVec,
          typename Scalar, typename YScalar, typename Allocator>
inline auto osc_step(SolverInfo &&info, OmegaVec &&omega_s, GammaVec &&gamma_s,
                     Scalar x0, Scalar h, YScalar y0, YScalar dy0,
                     Scalar epsres, Allocator &&alloc) {
  using complex_t = std::complex<Scalar>;
  using vectorc_t = vector_t<complex_t>;
  bool success = true;
  auto &&Dn = info.Dn();
  auto y = eval(alloc, complex_t(0.0, 1.0) * omega_s);
  auto delta = [&](const auto &r, const auto &y) {
    return (-r.array() / (2.0 * (y.array() + gamma_s.array()))).matrix().eval();
  };
  auto R = [&](const auto &d) {
    return 2.0 / h * (Dn * d) + d.array().square().matrix();
  };
  auto Ry = (complex_t(0.0, 1.0) * 2.0
             * (1.0 / h * (Dn * omega_s) + gamma_s.cwiseProduct(omega_s)))
                .eval();
  Scalar maxerr = Ry.array().abs().maxCoeff();

  arena_matrix<vectorc_t> deltay(alloc, Ry.size(), 1);
  Scalar prev_err = std::numeric_limits<Scalar>::infinity();
  while (maxerr > epsres) {
    deltay = delta(Ry, y);
    y += deltay;
    Ry = R(deltay);
    maxerr = Ry.array().abs().maxCoeff();
    if (maxerr >= prev_err) {
      success = false;
      break;
    }
    prev_err = maxerr;
  }
  if (info.denseout_) {
    auto u1 = eval(alloc, h / 2.0 * (info.integration_matrix_ * y));
    auto f1 = eval(alloc, (u1).array().exp().matrix());
    auto f2 = eval(alloc, f1.conjugate());
    auto du2 = eval(alloc, y.conjugate());
    auto ap_top = (dy0 - y0 * du2(du2.size() - 1));
    auto ap_bottom = (y(y.size() - 1) - du2(du2.size() - 1));
    auto ap = ap_top / ap_bottom;
    auto am = (dy0 - y0 * y(y.size() - 1))
              / (du2(du2.size() - 1) - y(y.size() - 1));
    auto y1 = eval(alloc, ap * f1 + am * f2);
    auto dy1
        = eval(alloc, ap * y.cwiseProduct(f1) + am * du2.cwiseProduct(f2));
    Scalar phase = std::imag(f1(0));
    return std::make_tuple(success, y1(0), dy1(0), maxerr, phase, u1,
                           std::make_pair(ap, am));
  } else {
    complex_t f1 = std::exp(h / 2.0 * (info.quadwts_.dot(y)));
    auto f2 = std::conj(f1);
    auto du2 = y.conjugate().eval();
    auto ap_top = (dy0 - y0 * du2(du2.size() - 1));
    auto ap_bottom = (y(y.size() - 1) - du2(du2.size() - 1));
    auto ap = ap_top / ap_bottom;
    auto am = (dy0 - y0 * y(y.size() - 1))
              / (du2(du2.size() - 1) - y(y.size() - 1));
    auto y1 = (ap * f1 + am * f2);
    auto dy1 = (ap * y * f1 + am * du2 * f2).eval();
    Scalar phase = std::imag(f1);
    return std::make_tuple(success, y1, dy1(0), maxerr, phase,
                           arena_matrix<vectorc_t>(alloc, y.size(), 0),
                           std::make_pair(ap, am));
  }
}

}  // namespace riccati

#endif
