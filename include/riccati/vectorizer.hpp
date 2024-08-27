#ifndef STAN_MATH_REV_CORE_VECTORIZER_HPP
#define STAN_MATH_REV_CORE_VECTORIZER_HPP

#include <riccati/macros.hpp>
#include <riccati/memory.hpp>
#include <riccati/utils.hpp>
#include <Eigen/Dense>
#include <type_traits>

namespace riccati {


/**
 * @brief Vectorizes a scalar function for use with Eigen matrices.
 * @note Instead of using this class directly use the `vectorize` function.
 * @tparam F The function type, typically a lambda or function object.
 * 
 */
template <typename F>
struct Vectorizer {
F func_;

template <typename FF, require_same<FF, F>* = nullptr>
explicit Vectorizer(FF&& func) : func_(std::forward<FF>(func)) {}

template <typename T, typename F_ = F, require_floating_point<T>* = nullptr,
  require_not_same<F, pybind11::object>* = nullptr>
inline auto operator()(T x) {
  return func_(x);
}

template <typename T, require_not_floating_point<T>* = nullptr,
  require_not_same<F, pybind11::object>* = nullptr>
inline auto operator()(T&& x) {
  return x.array()
      .unaryExpr([&func = func_](auto&& x) { return func(x); })
      .matrix()
      .eval();
}

template <typename T, typename F_ = F, require_floating_point<T>* = nullptr,
  require_same<F_, pybind11::object>* = nullptr>
inline auto operator()(T x) {
  return func_(x).template cast<T>();
}

template <typename T, typename F_ = F, require_not_floating_point<T>* = nullptr,
  require_same<F_, pybind11::object>* = nullptr>
inline auto operator()(T&& x) {
  return x.array()
      .unaryExpr([&func = func_](auto&& y) { return func(y).template cast<T>(); })
      .matrix()
      .eval();
}

};

/**
 * @brief Vectorizes a scalar function for use with Eigen matrices.
 * @tparam F A type with a valid `operator()` method.
 * @param func The function to vectorize.
 * 
 */
template <typename F>
inline auto vectorize(F&& func) {
  return Vectorizer<F>(std::forward<F>(func));
}

}  // namespace riccati



#endif