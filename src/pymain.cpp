
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/eigen/matrix.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
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

template <typename T>
struct is_scalar
    : bool_constant<
          std::is_floating_point_v<std::decay_t<
              T>> || std::is_integral_v<std::decay_t<T>> || is_complex_v<T>> {};

template <typename T>
constexpr bool is_scalar_v = is_scalar<T>::value;

template <
    typename SolverInfo, typename Scalar,
    std::enable_if_t<std::is_same<typename std::decay_t<SolverInfo>::gamma_type,
                                  pybind11::object>::value>* = nullptr,
    std::enable_if_t<is_scalar_v<Scalar>>* = nullptr>
RICCATI_ALWAYS_INLINE auto gamma(SolverInfo&& info, const Scalar& x) {
  using return_scalar_t = typename std::decay_t<SolverInfo>::GammaReturn;
  return info.gamma_fun_(x).template cast<return_scalar_t>();
}

template <
    typename SolverInfo, typename Scalar,
    std::enable_if_t<std::is_same<typename std::decay_t<SolverInfo>::omega_type,
                                  pybind11::object>::value>* = nullptr,
    std::enable_if_t<is_scalar_v<Scalar>>* = nullptr>
RICCATI_ALWAYS_INLINE auto omega(SolverInfo&& info, const Scalar& x) {
  using return_scalar_t = typename std::decay_t<SolverInfo>::OmegaReturn;
  return info.omega_fun_(x).template cast<return_scalar_t>();
}
template <
    typename SolverInfo, typename EigVec,
    std::enable_if_t<std::is_same<typename std::decay_t<SolverInfo>::gamma_type,
                                  pybind11::object>::value>* = nullptr,
    std::enable_if_t<is_eigen_v<EigVec>>* = nullptr>
RICCATI_ALWAYS_INLINE auto gamma(SolverInfo&& info, const EigVec& x) {
  using return_scalar_t = typename std::decay_t<SolverInfo>::GammaReturn;
  using return_t = Eigen::Matrix<return_scalar_t, -1, 1>;
  return info.gamma_fun_(x.eval()).template cast<return_t>().eval();
}

template <
    typename SolverInfo, typename EigVec,
    std::enable_if_t<std::is_same<typename std::decay_t<SolverInfo>::omega_type,
                                  pybind11::object>::value>* = nullptr,
    std::enable_if_t<is_eigen_v<EigVec>>* = nullptr>
RICCATI_ALWAYS_INLINE auto omega(SolverInfo&& info, const EigVec& x) {
  using return_scalar_t = typename std::decay_t<SolverInfo>::OmegaReturn;
  using return_t = Eigen::Matrix<return_scalar_t, -1, 1>;
  return info.omega_fun_(x.eval()).template cast<return_t>().eval();
}

template <typename SolverInfo, typename Scalar>
inline auto evolve(SolverInfo& info, Scalar xi, Scalar xf,
                   std::complex<Scalar> yi, std::complex<Scalar> dyi,
                   Scalar eps, Scalar epsilon_h, Scalar init_stepsize,
                   bool hard_stop = false,
                   riccati::LogLevel log_level = riccati::LogLevel::ERROR) {
  Eigen::Matrix<double, 0, 0> not_used;
  return evolve(info, xi, xf, yi, dyi, eps, epsilon_h, init_stepsize, not_used,
                hard_stop, log_level);
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
struct py_print {
  template <typename... Args>
  py_print& operator<<(Args&&... args) {
    py::print(std::forward<Args>(args)...);
    return *this;
  }
};
// Define the SolverInfo type for Python
// Omega: double and Gamma: double
using init_of64_gf64_f64_i64 = riccati::SolverInfo<
    py::object, py::object, double, int64_t,
    riccati::arena_allocator<double, riccati::arena_alloc>,
    riccati::SharedLogger<riccati::py_print>, double, double>;

// Omega: complex and Gamma: double
using init_oc64_gf64_f64_i64 = riccati::SolverInfo<
    py::object, py::object, double, int64_t,
    riccati::arena_allocator<double, riccati::arena_alloc>,
    riccati::SharedLogger<riccati::py_print>, std::complex<double>, double>;

// Omega: double and Gamma: complex
using init_of64_gc64_f64_i64 = riccati::SolverInfo<
    py::object, py::object, double, int64_t,
    riccati::arena_allocator<double, riccati::arena_alloc>,
    riccati::SharedLogger<riccati::py_print>, double, std::complex<double>>;

// Omega: complex and Gamma: complex
using init_oc64_gc64_f64_i64 = riccati::SolverInfo<
    py::object, py::object, double, int64_t,
    riccati::arena_allocator<double, riccati::arena_alloc>,
    riccati::SharedLogger<riccati::py_print>, std::complex<double>,
    std::complex<double>>;


template <typename T>
RICCATI_ALWAYS_INLINE auto hard_copy_arena(T&& x) {
  if constexpr (is_arena_matrix_v<T>) {
    return typename std::decay_t<T>::PlainObject(x);
  } else if constexpr (is_tuple_v<T>) {
    return std::apply(
        [](auto&&... args) {
          return std::make_tuple(
              hard_copy_arena(std::forward<decltype(args)>(args))...);
        },
        std::forward<T>(x));
  } else {
    return std::forward<T>(x);
  }
}

#ifndef RICCATI_INIT_DOCS
#define RICCATI_INIT_DOCS \
  R"pbdoc( \
          """\
          Construct a new SolverInfo object. \
                                             \
          Parameters \
          ---------- \
          omega_fun : callable \
              Frequency function. Must be able to take in and return scalars and vectors. \
          gamma_fun : callable \
              Friction function. Must be able to take in and return scalars and vectors. \
          nini : int \
              Minimum number of Chebyshev nodes to use inside Chebyshev collocation steps. \
          nmax : int \
              Maximum number of Chebyshev nodes to use inside Chebyshev collocation steps. \
          n : int \
              (Number of Chebyshev nodes - 1) to use inside Chebyshev collocation steps. \
          p : int \
              (Number of Chebyshev nodes - 1) to use for computing Riccati steps. \
          """ \
            )pbdoc"
#endif

template <typename F, typename Solver, typename XEval, typename... Args>
RICCATI_ALWAYS_INLINE auto evolve_call(F&& f, Solver&& info, XEval&& x_eval, Args&&... args) {
  if (x_eval.is_none()) {
    return riccati::hard_copy_arena(std::forward<F>(f)(
        info, Eigen::Matrix<double, 0, 0>{}, std::forward<Args>(args)...));
  } else {
    return riccati::hard_copy_arena(
        std::forward<F>(f)(info, x_eval.template cast<Eigen::VectorXd>(),
                           std::forward<Args>(args)...));
  }
}

template <typename F, typename... Args>
RICCATI_ALWAYS_INLINE auto info_caster(F&& f, py::object info, Args&&... args) {
  if (py::isinstance<riccati::init_of64_gf64_f64_i64>(info)) {
    auto info_ = info.cast<riccati::init_of64_gf64_f64_i64>();
    auto ret = std::forward<F>(f)(info_, std::forward<Args>(args)...);
    info_.alloc_.recover_memory();
    return ret;
  } else if (py::isinstance<riccati::init_oc64_gf64_f64_i64>(info)) {
    auto info_ = info.cast<riccati::init_oc64_gf64_f64_i64>();
    auto ret = std::forward<F>(f)(info_, std::forward<Args>(args)...);
    info_.alloc_.recover_memory();
    return ret;
  } else if (py::isinstance<riccati::init_of64_gc64_f64_i64>(info)) {
    auto info_ = info.cast<riccati::init_of64_gc64_f64_i64>();
    auto ret = std::forward<F>(f)(info_, std::forward<Args>(args)...);
    info_.alloc_.recover_memory();
    return ret;
  } else if (py::isinstance<riccati::init_oc64_gc64_f64_i64>(info)) {
    auto info_ = info.cast<riccati::init_oc64_gc64_f64_i64>();
    auto ret = std::forward<F>(f)(info_, std::forward<Args>(args)...);
    info_.alloc_.recover_memory();
    return ret;
  } else {
    throw std::invalid_argument("Invalid SolverInfo object.");
  }
}

}  // namespace riccati

PYBIND11_MODULE(pyriccaticpp, m) {
  m.doc() = "Riccati solver module";
  py::enum_<riccati::LogLevel>(m, "LogLevel", py::arithmetic(),
                               "Log levels for eveolve")
      .value("ERROR", riccati::LogLevel::ERROR, "Report Only Errors")
      .value("WARNING", riccati::LogLevel::WARNING, "Report up to warnings")
      .value("INFO", riccati::LogLevel::INFO, "Report all information")
      .value("DEBUG", riccati::LogLevel::DEBUG,
             "DEV ONLY: Report all information and debug info")
      .export_values();
  // Expose LogInfo
  py::enum_<riccati::LogInfo>(m, "LogInfo", "Information types for logging")
      .value("CHEBNODES", riccati::LogInfo::CHEBNODES)
      .value("CHEBSTEP", riccati::LogInfo::CHEBSTEP)
      .value("CHEBITS", riccati::LogInfo::CHEBITS)
      .value("LS", riccati::LogInfo::LS)
      .value("RICCSTEP", riccati::LogInfo::RICCSTEP)
      .export_values();
  // Omega: double, Gamma: double
  py::class_<riccati::init_of64_gf64_f64_i64>(m, "Init_OF64_GF64")
      .def(py::init<py::object, py::object, int64_t, int64_t, int64_t,
                    int64_t>(),
           RICCATI_INIT_DOCS);

  // Omega: complex, Gamma: double
  py::class_<riccati::init_oc64_gf64_f64_i64>(m, "Init_OC64_GF64")
      .def(py::init<py::object, py::object, int64_t, int64_t, int64_t,
                    int64_t>(),
           RICCATI_INIT_DOCS);

  // Omega: double, Gamma: complex
  py::class_<riccati::init_of64_gc64_f64_i64>(m, "Init_OF64_GC64")
      .def(py::init<py::object, py::object, int64_t, int64_t, int64_t,
                    int64_t>(),
           RICCATI_INIT_DOCS);

  // Omega: complex, Gamma: complex
  py::class_<riccati::init_oc64_gc64_f64_i64>(m, "Init_OC64_GC64")
      .def(py::init<py::object, py::object, int64_t, int64_t, int64_t,
                    int64_t>(),
           RICCATI_INIT_DOCS);
  /**
   * This is very silly looking, but essentially we have two levels of dynamic
   *  dispatch we need to do. One for the info type, which can switch values
   *  based on the return type of omega and gamma (double and complex) and
   *  another for whether we are doing dense evaluations. Placing each of
   *  the choices inside of their own functions makes the logic clean, but
   *  means we need some indirection in our calling site when calling them.
   *  I need to think of a nicer way to do this...
   */
  m.def(
      "evolve",
      [](py::object& info, double xi, double xf, std::complex<double> yi,
         std::complex<double> dyi, double eps, double epsilon_h,
         double init_stepsize, py::object x_eval, bool hard_stop,
         riccati::LogLevel log_level) {
        return riccati::info_caster(
            [](auto&& info, auto&& x_eval, auto&&... args) {
              return riccati::evolve_call(
                  [](auto&& info, auto&& x_eval, double xi, double xf,
                     std::complex<double> yi, std::complex<double> dyi,
                     double eps, double epsilon_h, double init_stepsize,
                     bool hard_stop, riccati::LogLevel log_level) {
                    return riccati::evolve(info, xi, xf, yi, dyi, eps,
                                           epsilon_h, init_stepsize, x_eval,
                                           hard_stop, log_level);
                  },
                  info, x_eval, args...);
            },
            info, x_eval, xi, xf, yi, dyi, eps, epsilon_h, init_stepsize,
            hard_stop, log_level);
      },
      py::arg("info"), py::arg("xi"), py::arg("xf"), py::arg("yi"),
      py::arg("dyi"), py::arg("eps"), py::arg("epsilon_h"),
      py::arg("init_stepsize") = 0.01, py::arg("x_eval") = py::none(),
      py::arg("hard_stop") = false,
      py::arg("log_level") = riccati::LogLevel::ERROR,
      R"pbdoc(
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
        return riccati::info_caster(
            [](auto&& info, auto&& x_eval, auto&&... args) {
              return riccati::evolve_call(
                  [](auto&& info, auto&& x_eval, double xi, double xf,
                     std::complex<double> yi, std::complex<double> dyi,
                     double eps, double epsilon_h, double init_stepsize,
                     bool hard_stop) {
                    return riccati::osc_evolve(info, xi, xf, yi, dyi, eps,
                                           epsilon_h, init_stepsize, x_eval,
                                           hard_stop);
                  },
                  info, x_eval, args...);
            },
            info, x_eval, xi, xf, yi, dyi, eps, epsilon_h, init_stepsize,
            hard_stop);
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
        return riccati::info_caster(
            [](auto&& info, auto&& x_eval, auto&&... args) {
              return riccati::evolve_call(
                  [](auto&& info, auto&& x_eval, double xi, double xf,
                     std::complex<double> yi, std::complex<double> dyi,
                     double eps, double epsilon_h, double init_stepsize,
                     bool hard_stop) {
                    return riccati::nonosc_evolve(info, xi, xf, yi, dyi, eps,
                                           epsilon_h, init_stepsize, x_eval);
                  },
                  info, x_eval, args...);
            },
            info, x_eval, xi, xf, yi, dyi, eps, epsilon_h, init_stepsize,
            hard_stop);
      },
      py::arg("info"), py::arg("xi"), py::arg("xf"), py::arg("yi"),
      py::arg("dyi"), py::arg("eps"), py::arg("epsilon_h"),
      py::arg("init_stepsize") = 0.01, py::arg("x_eval") = py::none(),
      py::arg("hard_stop") = false, R"pbdoc()pbdoc");

  m.def(
      "choose_osc_stepsize",
      [](py::object& info, double x0, double h, double epsilon_h) {
        return riccati::info_caster(
            [](auto& info, double x0, double h, double epsilon_h) {
              return std::get<0>(
                  riccati::choose_osc_stepsize(info, x0, h, epsilon_h));
            },
            info, x0, h, epsilon_h);
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
        return riccati::info_caster(
            [](auto& info, double x0, double h, double epsilon_h) {
              return riccati::choose_nonosc_stepsize_<decltype(info), double>(
                  info, x0, h, epsilon_h);
            },
            info, x0, h, epsilon_h);
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
