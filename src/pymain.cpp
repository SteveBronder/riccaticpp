
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/eigen/matrix.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <riccati/evolve.hpp>
#include <riccati/solver.hpp>
#include <Eigen/Dense>
#include <functional>
#include <iostream>
#include <complex>
#include <utility>
#include <type_traits>

namespace py = pybind11;

namespace riccati {
template <bool B>
using bool_constant = std::integral_constant<bool, B>;
/**
 * Checks if a type's pointer is convertible to a templated base type's pointer.
 * If the arbitrary function
 * ```
 * std::true_type f(const Base<Derived>*)
 * ```
 * is well formed for input `std::declval<Derived*>() this has a member
 *  value equal to `true`, otherwise the value is false.
 * @tparam Base The templated base type for valid pointer conversion.
 * @tparam Derived The type to check
 * @ingroup type_trait
 */
template <template <typename> class Base, typename Derived>
struct is_base_pointer_convertible {
  static std::false_type f(const void*);
  template <typename OtherDerived>
  static std::true_type f(const Base<OtherDerived>*);
  enum {
    value
    = decltype(f(std::declval<std::remove_reference_t<Derived>*>()))::value
  };
};
template <typename T>
struct is_eigen
    : bool_constant<is_base_pointer_convertible<Eigen::EigenBase, T>::value> {};

template <typename T>
struct is_scalar : bool_constant<std::is_floating_point<std::decay_t<T>>::value
                                 || std::is_integral<std::decay_t<T>>::value> {
};

template <
    typename SolverInfo, typename Scalar,
    std::enable_if_t<std::is_same<typename std::decay_t<SolverInfo>::funtype,
                                  pybind11::object>::value>* = nullptr,
    std::enable_if_t<is_scalar<Scalar>::value>* = nullptr>
inline auto gamma(SolverInfo&& info, const Scalar& x) {
  return info.gamma_fun_(x).template cast<Scalar>();
}

template <
    typename SolverInfo, typename Scalar,
    std::enable_if_t<std::is_same<typename std::decay_t<SolverInfo>::funtype,
                                  pybind11::object>::value>* = nullptr,
    std::enable_if_t<is_scalar<Scalar>::value>* = nullptr>
inline auto omega(SolverInfo&& info, const Scalar& x) {
  return info.omega_fun_(x).template cast<Scalar>();
}
template <
    typename SolverInfo, typename Scalar,
    std::enable_if_t<std::is_same<typename std::decay_t<SolverInfo>::funtype,
                                  pybind11::object>::value>* = nullptr,
    std::enable_if_t<is_eigen<Scalar>::value>* = nullptr>
inline auto gamma(SolverInfo&& info, const Scalar& x) {
  return info.gamma_fun_(x.eval())
      .template cast<typename Scalar::PlainObject>();
}

template <
    typename SolverInfo, typename Scalar,
    std::enable_if_t<std::is_same<typename std::decay_t<SolverInfo>::funtype,
                                  pybind11::object>::value>* = nullptr,
    std::enable_if_t<is_eigen<Scalar>::value>* = nullptr>
inline auto omega(SolverInfo&& info, const Scalar& x) {
  return info.omega_fun_(x.eval())
      .template cast<typename Scalar::PlainObject>();
}

template <typename SolverInfo, typename Scalar>
inline auto evolve(SolverInfo& info, Scalar xi, Scalar xf,
                   std::complex<Scalar> yi, std::complex<Scalar> dyi,
                   Scalar eps, Scalar epsilon_h, Scalar init_stepsize,
                   bool hard_stop = false) {
  Eigen::Matrix<double, 0, 0> not_used;
  return evolve(info, xi, xf, yi, dyi, eps, epsilon_h, init_stepsize, not_used,
                hard_stop);
}

template <typename SolverInfo, typename Scalar>
inline auto osc_evolve(SolverInfo& info, Scalar xi, Scalar xf,
                       std::complex<Scalar> yi, std::complex<Scalar> dyi,
                       Scalar eps, Scalar epsilon_h, Scalar init_stepsize,
                       bool hard_stop = false) {
  Eigen::Matrix<double, 0, 0> not_used;
  return osc_evolve(info, xi, xf, yi, dyi, eps, epsilon_h, init_stepsize,
                    not_used, hard_stop);
}

template <typename SolverInfo, typename Scalar>
inline auto nonosc_evolve(SolverInfo& info, Scalar xi, Scalar xf,
                          std::complex<Scalar> yi, std::complex<Scalar> dyi,
                          Scalar eps, Scalar epsilon_h, Scalar init_stepsize,
                          bool hard_stop = false) {
  Eigen::Matrix<double, 0, 0> not_used;
  return nonosc_evolve(info, xi, xf, yi, dyi, eps, epsilon_h, init_stepsize,
                       not_used, hard_stop);
}

template <typename SolverInfo, typename Scalar>
inline auto step_(SolverInfo& info, Scalar xi, Scalar xf,
                  std::complex<Scalar> yi, std::complex<Scalar> dyi, Scalar eps,
                  Scalar epsilon_h, Scalar init_stepsize,
                  bool hard_stop = false) {
  Eigen::VectorXd not_used;
  return step(info, xi, xf, yi, dyi, eps, epsilon_h, init_stepsize, not_used,
              hard_stop);
}
template <typename SolverInfo, typename FloatingPoint>
inline auto choose_osc_stepsize_(SolverInfo& info, FloatingPoint x0,
                                 FloatingPoint h, FloatingPoint epsilon_h) {
  return choose_osc_stepsize(info, x0, h, epsilon_h);
  info.alloc_.recover_memory();
}

template <typename SolverInfo, typename FloatingPoint>
inline FloatingPoint choose_nonosc_stepsize_(SolverInfo& info, FloatingPoint x0,
                                             FloatingPoint h,
                                             FloatingPoint epsilon_h) {
  return choose_nonosc_stepsize(info, x0, h, epsilon_h);
  info.alloc_.recover_memory();
}
using init_f64_i64
    = riccati::SolverInfo<py::object, py::object, double, int64_t>;

template <typename Mat>
auto hard_copy_arena(arena_matrix<Mat>&& x) {
  return Mat(x);
}

template <typename Alt>
auto hard_copy_arena(Alt&& x) {
  return std::move(x);
}

template <typename Tuple>
auto hard_copy_arena(std::tuple<Tuple>&& tup) {
  return std::apply(
      [](auto&&... args) {
        return std::make_tuple(
            hard_copy_arena(std::forward<decltype(args)>(args))...);
      },
      std::move(tup));
}
}  // namespace riccati

PYBIND11_MODULE(pyriccaticpp, m) {
  m.doc() = "Riccati solver module";
  py::class_<riccati::init_f64_i64>(m, "Init").def(
      py::init<py::object, py::object, int64_t, int64_t, int64_t, int64_t>(),
      R"pbdoc(
          """
          Construct a new SolverInfo object.

          Parameters
          ----------
          omega_fun : callable
              Frequency function. Must be able to take in and return scalars and vectors.
          gamma_fun : callable
              Friction function. Must be able to take in and return scalars and vectors.
          nini : int
              Minimum number of Chebyshev nodes to use inside Chebyshev collocation steps.
          nmax : int
              Maximum number of Chebyshev nodes to use inside Chebyshev collocation steps.
          n : int
              (Number of Chebyshev nodes - 1) to use inside Chebyshev collocation steps.
          p : int
              (Number of Chebyshev nodes - 1) to use for computing Riccati steps.
          """
            )pbdoc");

  m.def(
      "step",
      [](py::object& info, double xi, double xf, std::complex<double> yi,
         std::complex<double> dyi, double eps, double epsilon_h,
         double init_stepsize, py::object x_eval, bool hard_stop) {
        if (py::isinstance<riccati::init_f64_i64>(info)) {
          auto info_ = info.cast<riccati::init_f64_i64>();
          if (x_eval.is_none()) {
            auto ret = riccati::step_<riccati::init_f64_i64, double>(
                info_, xi, xf, yi, dyi, eps, epsilon_h, init_stepsize,
                hard_stop);
            info_.alloc_.recover_memory();
            return ret;
          } else {
            auto ret = riccati::step<riccati::init_f64_i64, double>(
                info_, xi, xf, yi, dyi, eps, epsilon_h, init_stepsize,
                x_eval.cast<Eigen::VectorXd>(), hard_stop);
            info_.alloc_.recover_memory();
            return ret;
          }
        } else {
          throw std::invalid_argument("Invalid SolverInfo object.");
        }
      },
      py::arg("info"), py::arg("xi"), py::arg("xf"), py::arg("yi"),
      py::arg("dyi"), py::arg("eps"), py::arg("epsilon_h"),
      py::arg("init_stepsize") = 0.01, py::arg("x_eval") = py::none(),
      py::arg("hard_stop") = false, R"pbdoc(
    """
    Solves the differential equation y'' + 2gy' + w^2y = 0 over a given interval.

    This function solves the differential equation on the interval (xi, xf), starting from the initial conditions 
    y(xi) = yi and y'(xi) = dyi. It keeps the residual of the ODE below eps, and returns an interpolated solution 
    (dense output) at the points specified in x_eval.

    Parameters
    ----------
    info : SolverInfo
        SolverInfo object containing necessary information for the solver, such as differentiation matrices, etc.
    xi : float
        Starting value of the independent variable.
    xf : float
        Ending value of the independent variable.
    yi : complex
        Initial value of the dependent variable at xi.
    dyi : complex
        Initial derivative of the dependent variable at xi.
    eps : float
        Relative tolerance for the local error of both Riccati and Chebyshev type steps.
    epsilon_h : float
        Relative tolerance for choosing the stepsize of Riccati steps.
    init_stepsize : float
        Initial stepsize for the integration.
    x_eval : numpy.ndarray[numpy.float64], optional
        List of x-values where the solution is to be interpolated (dense output) and returned.
    hard_stop : bool, optional
        If True, forces the solver to have a potentially smaller last stepsize to stop exactly at xf. Default is False.

    Returns
    -------
    tuple[list[float], list[complex], list[complex], list[int], list[float], list[int]]
        A tuple containing multiple elements representing the results of the ODE solving process:
        - list[float]: x-values at which the solution was evaluated or interpolated, including points in x_eval if dense output was requested.
        - list[complex]: The solution y(x) of the differential equation at each x-value.
        - list[complex]: The derivative of the solution, y'(x), at each x-value.
        - list[int]: Indicates the success status of the solver at each step (1 for success, 0 for failure).
        - list[int]: Indicates the type of step taken at each point (1 for oscillatory step, 0 for non-oscillatory step).
        - list[float]: The phase angle at each step of the solution process (relevant for oscillatory solutions).
    """
          )pbdoc");

  m.def(
      "evolve",
      [](py::object& info, double xi, double xf, std::complex<double> yi,
         std::complex<double> dyi, double eps, double epsilon_h,
         double init_stepsize, py::object x_eval, bool hard_stop) {
        if (py::isinstance<riccati::init_f64_i64>(info)) {
          auto info_ = info.cast<riccati::init_f64_i64>();
          if (x_eval.is_none()) {
            auto ret = riccati::evolve(info_, xi, xf, yi, dyi, eps, epsilon_h,
                                       init_stepsize, hard_stop);
            info_.alloc_.recover_memory();
            return ret;
          } else {
            auto ret = riccati::evolve(
                info_, xi, xf, yi, dyi, eps, epsilon_h, init_stepsize,
                x_eval.cast<Eigen::VectorXd>(), hard_stop);
            info_.alloc_.recover_memory();
            return ret;
          }
        } else {
          throw std::invalid_argument("Invalid SolverInfo object.");
        }
      },
      py::arg("info"), py::arg("xi"), py::arg("xf"), py::arg("yi"),
      py::arg("dyi"), py::arg("eps"), py::arg("epsilon_h"),
      py::arg("init_stepsize") = 0.01, py::arg("x_eval") = py::none(),
      py::arg("hard_stop") = false, R"pbdoc(
    """
    Solves the differential equation y'' + 2gy' + w^2y = 0 over a given interval.

    This function solves the differential equation on the interval (xi, xf), starting from the initial conditions 
    y(xi) = yi and y'(xi) = dyi. It keeps the residual of the ODE below eps, and returns an interpolated solution 
    (dense output) at the points specified in x_eval.

    Parameters
    ----------
    info : SolverInfo
        SolverInfo object containing necessary information for the solver, such as differentiation matrices, etc.
    xi : float
        Starting value of the independent variable.
    xf : float
        Ending value of the independent variable.
    yi : complex
        Initial value of the dependent variable at xi.
    dyi : complex
        Initial derivative of the dependent variable at xi.
    eps : float
        Relative tolerance for the local error of both Riccati and Chebyshev type steps.
    epsilon_h : float
        Relative tolerance for choosing the stepsize of Riccati steps.
    init_stepsize : float
        Initial stepsize for the integration.
    x_eval : numpy.ndarray[numpy.float64], optional
        List of x-values where the solution is to be interpolated (dense output) and returned.
    hard_stop : bool, optional
        If True, forces the solver to have a potentially smaller last stepsize to stop exactly at xf. Default is False.

    Returns
    -------
    tuple[list[float], list[complex], list[complex], list[int], list[float], list[int], numpy.ndarray[numpy.complex128[m, 1]]]
        A tuple containing multiple elements representing the results of the ODE solving process:
        - list[float]: x-values at which the solution was evaluated or interpolated, including points in x_eval if dense output was requested.
        - list[complex]: The solution y(x) of the differential equation at each x-value.
        - list[complex]: The derivative of the solution, y'(x), at each x-value.
        - list[int]: Indicates the success status of the solver at each step (1 for success, 0 for failure).
        - list[int]: Indicates the type of step taken at each point (1 for oscillatory step, 0 for non-oscillatory step).
        - list[float]: The phase angle at each step of the solution process (relevant for oscillatory solutions).
        - numpy.ndarray[numpy.complex128[m, 1]]: Interpolated solution at the specified x_eval.
    """
          )pbdoc");

  m.def(
      "osc_evolve",
      [](py::object& info, double xi, double xf, std::complex<double> yi,
         std::complex<double> dyi, double eps, double epsilon_h,
         double init_stepsize, py::object x_eval, bool hard_stop) {
        if (py::isinstance<riccati::init_f64_i64>(info)) {
          auto info_ = info.cast<riccati::init_f64_i64>();
          if (x_eval.is_none()) {
            auto ret = riccati::hard_copy_arena(
                riccati::osc_evolve(info_, xi, xf, yi, dyi, eps, epsilon_h,
                                    init_stepsize, hard_stop));
            info_.alloc_.recover_memory();
            return ret;
          } else {
            auto ret = riccati::hard_copy_arena(riccati::osc_evolve(
                info_, xi, xf, yi, dyi, eps, epsilon_h, init_stepsize,
                x_eval.cast<Eigen::VectorXd>(), hard_stop));
            info_.alloc_.recover_memory();
            return ret;
          }
        } else {
          throw std::invalid_argument("Invalid SolverInfo object.");
        }
      },
      py::arg("info"), py::arg("xi"), py::arg("xf"), py::arg("yi"),
      py::arg("dyi"), py::arg("eps"), py::arg("epsilon_h"),
      py::arg("init_stepsize") = 0.01, py::arg("x_eval") = py::none(),
      py::arg("hard_stop") = false, R"pbdoc()pbdoc");

  m.def(
      "nonosc_evolve",
      [](py::object& info, double xi, double xf, std::complex<double> yi,
         std::complex<double> dyi, double eps, double epsilon_h,
         double init_stepsize, py::object x_eval, bool hard_stop) {
        if (py::isinstance<riccati::init_f64_i64>(info)) {
          auto info_ = info.cast<riccati::init_f64_i64>();
          if (x_eval.is_none()) {
            auto ret = riccati::hard_copy_arena(
                riccati::nonosc_evolve(info_, xi, xf, yi, dyi, eps, epsilon_h,
                                       init_stepsize, hard_stop));
            info_.alloc_.recover_memory();
            return ret;
          } else {
            auto ret = riccati::hard_copy_arena(riccati::nonosc_evolve(
                info_, xi, xf, yi, dyi, eps, epsilon_h, init_stepsize,
                x_eval.cast<Eigen::VectorXd>(), hard_stop));
            info_.alloc_.recover_memory();
            return ret;
          }
        } else {
          throw std::invalid_argument("Invalid SolverInfo object.");
        }
      },
      py::arg("info"), py::arg("xi"), py::arg("xf"), py::arg("yi"),
      py::arg("dyi"), py::arg("eps"), py::arg("epsilon_h"),
      py::arg("init_stepsize") = 0.01, py::arg("x_eval") = py::none(),
      py::arg("hard_stop") = false, R"pbdoc()pbdoc");

  m.def(
      "choose_osc_stepsize",
      [](py::object& info, double x0, double h, double epsilon_h) {
        if (py::isinstance<riccati::init_f64_i64>(info)) {
          auto info_ = info.cast<riccati::init_f64_i64>();
          auto ret = riccati::hard_copy_arena(
              riccati::choose_osc_stepsize(info_, x0, h, epsilon_h));
          info_.alloc_.recover_memory();
          return ret;
        } else {
          throw std::invalid_argument("Invalid SolverInfo object.");
        }
      },
      py::arg("info"), py::arg("x0"), py::arg("h"), py::arg("epsilon_h"),
      R"pbdoc(
                """
    Chooses an appropriate step size for the Riccati step based on the accuracy of Chebyshev interpolation of w(x) and g(x).

    This function determines an optimal step size `h` over which the functions `w(x)` and `g(x)` can be represented with 
    sufficient accuracy by evaluating their values at `p+1` Chebyshev nodes. It performs interpolation to `p` points 
    halfway between these nodes and compares the interpolated values with the actual values of `w(x)` and `g(x)`. 
    If the largest relative error in `w` or `g` exceeds the tolerance `epsilon_h`, the step size `h` is reduced. 
    This process ensures that the Chebyshev interpolation of `w(x)` and `g(x)` over the step [`x0`, `x0+h`] has a 
    relative error no larger than `epsilon_h`.

    Parameters
    ----------
    info : SolverInfo
        Object containing pre-computed information and methods for evaluating functions `w(x)` and `g(x)`, 
        as well as interpolation matrices and node positions.
    x0 : float
        The current value of the independent variable.
    h : float
        The initial estimate of the step size.
    epsilon_h : float
        Tolerance parameter defining the maximum relative error allowed in the Chebyshev interpolation of `w(x)` 
        and `g(x)` over the proposed step.

    Returns
    -------
    float
        The refined step size over which the Chebyshev interpolation of `w(x)` and `g(x)` satisfies the relative error tolerance `epsilon_h`.
    """
          )pbdoc");
  m.def(
      "choose_nonosc_stepsize",
      [](py::object& info, double x0, double h, double epsilon_h) {
        if (py::isinstance<riccati::init_f64_i64>(info)) {
          auto info_ = info.cast<riccati::init_f64_i64>();
          auto ret
              = riccati::choose_nonosc_stepsize_<riccati::init_f64_i64, double>(
                  info_, x0, h, epsilon_h);
          info_.alloc_.recover_memory();
          return ret;
        } else {
          throw std::invalid_argument("Invalid SolverInfo object.");
        }
      },
      py::arg("info"), py::arg("x0"), py::arg("h"), py::arg("epsilon_h"),
      R"pbdoc(
    """
    Chooses the stepsize for spectral Chebyshev steps based on the variation of 1/w, 
    the approximate timescale over which the solution changes.

    This function evaluates the change in 1/w over the suggested interval h. 
    If 1/w changes by a fraction of ±epsilon_h or more, the interval is halved; otherwise, it is accepted.

    Parameters
    ----------
    info : SolverInfo
        A SolverInfo-like object, used to retrieve SolverInfo.xp, which contains the (p+1) Chebyshev nodes 
        used for interpolation to determine the stepsize.
    x0 : float
        The current value of the independent variable.
    h : float
        Initial estimate of the stepsize.
    epsilon_h : float
        Tolerance parameter defining how much 1/w(x) is allowed to change over the course of the step.

    Returns
    -------
    float
        Refined stepsize over which 1/w(x) does not change by more than epsilon_h/w(x).
    """            
          )pbdoc");
}
