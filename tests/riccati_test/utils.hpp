#ifndef RICCATI_TESTS_CPP_UTILS_HPP
#define RICCATI_TESTS_CPP_UTILS_HPP

#include <boost/math/special_functions/airy.hpp>
#include <riccati/memory.hpp>
#include <gtest/gtest.h>
#include <type_traits>

struct Riccati : public testing::Test {
  riccati::arena_alloc* arena{new riccati::arena_alloc{}};
  riccati::arena_allocator<double, riccati::arena_alloc> allocator{arena};
  void SetUp() override {
    // make sure memory's clean before starting each test
  }
  void TearDown() override { allocator.recover_memory(); }
  ~Riccati() { delete arena; }
};

namespace riccati {
namespace test {
template <typename T>
using require_not_floating_point
    = std::enable_if_t<!std::is_floating_point<std::decay_t<T>>::value>;

template <typename T>
using require_floating_point
    = std::enable_if_t<std::is_floating_point<std::decay_t<T>>::value>;

namespace internal {
template <typename T>
struct is_complex_impl : std::false_type {};
template <typename T>
struct is_complex_impl<std::complex<T>> : std::true_type {};
}  // namespace internal

template <typename T>
struct is_complex : internal::is_complex_impl<std::decay_t<T>> {};

template <typename T>
using require_floating_point_or_complex
    = std::enable_if_t<std::is_floating_point<std::decay_t<T>>::value
                       || is_complex<std::decay_t<T>>::value>;

template <typename T>
using require_not_floating_point_or_complex
    = std::enable_if_t<!std::is_floating_point<std::decay_t<T>>::value
                       && !is_complex<std::decay_t<T>>::value>;

template <typename T, require_floating_point_or_complex<T>* = nullptr>
inline auto eval(T x) {
  return x;
}

template <typename T, require_not_floating_point_or_complex<T>* = nullptr>
inline auto eval(T&& x) {
  return x.eval();
}

template <typename T, require_floating_point<T>* = nullptr>
inline auto airy_ai(T x) {
  return boost::math::airy_ai(x);
}

template <typename T, require_not_floating_point<T>* = nullptr>
inline auto airy_ai(T&& x) {
  return x.array()
      .unaryExpr([](auto&& x) { return boost::math::airy_ai(x); })
      .matrix()
      .eval();
}

template <typename T, require_floating_point<T>* = nullptr>
inline auto airy_bi(T x) {
  return boost::math::airy_bi(x);
}

template <typename T, require_not_floating_point<T>* = nullptr>
inline auto airy_bi(T&& x) {
  return x.array()
      .unaryExpr([](auto&& x) { return boost::math::airy_bi(x); })
      .matrix()
      .eval();
}

template <typename T, require_floating_point<T>* = nullptr>
inline auto airy_ai_prime(T x) {
  return boost::math::airy_ai_prime(x);
}

template <typename T, require_not_floating_point<T>* = nullptr>
inline auto airy_ai_prime(T&& x) {
  return x.array()
      .unaryExpr([](auto&& x) { return boost::math::airy_ai_prime(x); })
      .matrix()
      .eval();
}

template <typename T, require_floating_point<T>* = nullptr>
inline auto airy_bi_prime(T x) {
  return boost::math::airy_bi_prime(x);
}

template <typename T, require_not_floating_point<T>* = nullptr>
inline auto airy_bi_prime(T&& x) {
  return x.array()
      .unaryExpr([](auto&& x) { return boost::math::airy_bi_prime(x); })
      .matrix()
      .eval();
}


template <typename T>
inline auto airy_i(T&& xi) {
  return eval(airy_ai(-xi)
              + std::complex<value_type_t<T>>(0.0, 1.0) * airy_bi(-xi));
}

template <typename T>
inline auto airy_i_prime(T&& xi) {
  return eval(-airy_ai_prime(-xi)
              - std::complex<value_type_t<T>>(0.0, 1.0) * airy_bi_prime(-xi));
}

}  // namespace test
}  // namespace riccati

#endif
