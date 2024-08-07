
#include <riccati/evolve.hpp>
#include <riccati/solver.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen/matrix.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <functional>
#include <iostream>
#include <complex>
#include <utility>
#include <type_traits>

namespace py = pybind11;


namespace riccati {
  using nondense_init_f64_i64 = riccati::SolverInfo<py::object, py::object, double, int64_t, false>;
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
  static std::false_type f(const void *);
  template <typename OtherDerived>
  static std::true_type f(const Base<OtherDerived> *);
  enum {
    value
    = decltype(f(std::declval<std::remove_reference_t<Derived> *>()))::value
  };
};
template <typename T>
struct is_eigen
    : bool_constant<is_base_pointer_convertible<Eigen::EigenBase, T>::value> {};

template <typename T>
struct is_scalar :
    bool_constant<std::is_floating_point<std::decay_t<T>>::value || std::is_integral<std::decay_t<T>>::value> {};

  template <typename SolverInfo, typename Scalar,
  std::enable_if_t<std::is_same<typename std::decay_t<SolverInfo>::funtype, pybind11::object>::value>* = nullptr,
  std::enable_if_t<is_scalar<Scalar>::value>* = nullptr>
inline auto gamma(SolverInfo&& info, const Scalar& x) {
  return info.gamma_fun_(x).template cast<Scalar>();
}

template <typename SolverInfo,  typename Scalar,
  std::enable_if_t<std::is_same<typename std::decay_t<SolverInfo>::funtype, pybind11::object>::value>* = nullptr,
  std::enable_if_t<is_scalar<Scalar>::value>* = nullptr>
inline auto omega(SolverInfo&& info, const Scalar& x) {
  return info.omega_fun_(x).template cast<Scalar>();
}
  template <typename SolverInfo, typename Scalar,
  std::enable_if_t<std::is_same<typename std::decay_t<SolverInfo>::funtype, pybind11::object>::value>* = nullptr,
  std::enable_if_t<is_eigen<Scalar>::value>* = nullptr>
inline auto gamma(SolverInfo&& info, const Scalar& x) {
  return info.gamma_fun_(x.eval()).template cast<typename Scalar::PlainObject>();
}

template <typename SolverInfo,  typename Scalar,
  std::enable_if_t<std::is_same<typename std::decay_t<SolverInfo>::funtype, pybind11::object>::value>* = nullptr,
  std::enable_if_t<is_eigen<Scalar>::value>* = nullptr>
inline auto omega(SolverInfo&& info, const Scalar& x) {
  return info.omega_fun_(x.eval()).template cast<typename Scalar::PlainObject>();
}

template <typename SolverInfo, typename Scalar>
inline auto evolve(SolverInfo &info, Scalar xi, Scalar xf,
                   std::complex<Scalar> yi, std::complex<Scalar> dyi,
                   Scalar eps, Scalar epsilon_h, Scalar init_stepsize,
                   bool hard_stop = false) {
                   Eigen::VectorXd not_used;
                   return evolve(info, xi, xf, yi, dyi, eps, epsilon_h, init_stepsize, not_used, hard_stop);
                   }
}

PYBIND11_MODULE(pyriccaticpp, m) {
    py::class_<riccati::nondense_init_f64_i64>(m, "Init")
        .def(py::init<py::object, py::object, int64_t, int64_t, int64_t, int64_t>());
    m.def("evolve", &riccati::evolve<riccati::nondense_init_f64_i64, double>, 
          py::arg("info"), py::arg("xi"), py::arg("xf"), py::arg("yi"), py::arg("dyi"),
          py::arg("eps"), py::arg("epsilon_h"), py::arg("init_stepsize"),
          py::arg("hard_stop") = false);
}
