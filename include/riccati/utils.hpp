#ifndef INCLUDE_riccati_UTILS_HPP
#define INCLUDE_riccati_UTILS_HPP

#include <Eigen/Dense>
#include <type_traits>
#ifdef RICCATI_DEBUG
#include <iostream>
#include <iomanip>
#endif
namespace riccati {

/**
 * @brief Scales and shifts a vector of Chebyshev nodes.
 *
 * This function takes a vector of Chebyshev nodes and scales it to fit a
 * specific interval. Each element `x(i)` of the input vector is transformed to
 * fit into the new interval starting at `x0` with a width of `h`, effectively
 * mapping the standard Chebyshev interval `[-1, 1]` to `[x0, x0 + h]`. The
 * transformation used is `x(i) -> x0 + h/2 + h/2 * x(i)`. This is commonly used
 * to adjust Chebyshev nodes or the results of spectral methods to the specific
 * interval of interest in numerical computations.
 *
 * @tparam Scalar The scalar type, typically a floating-point type like double
 * or float.
 * @tparam Vector The Eigen vector type, typically Eigen::VectorXd or similar.
 * @param x Vector (forwarded reference) - The input vector, typically
 * containing Chebyshev nodes or similar quantities.
 * @param x0 Scalar - The start of the interval to which the nodes should be
 * scaled.
 * @param h Scalar - The width of the interval to which the nodes should be
 * scaled.
 * @return Returns a new Eigen vector of the same type as `x`, with each element
 * scaled and shifted to the new interval `[x0, x0 + h]`.
 */
template <typename Scalar, typename Vector>
inline auto scale(Vector&& x, Scalar x0, Scalar h) {
  return (x0 + h / 2.0 + h / 2.0 * x.array()).matrix();
}

template <typename Scalar>
using matrix_t = Eigen::Matrix<Scalar, -1, -1>;
template <typename Scalar>
using vector_t = Eigen::Matrix<Scalar, -1, 1>;

template <typename Scalar>
using row_vector_t = Eigen::Matrix<Scalar, 1, -1>;

template <typename Scalar>
using array2d_t = Eigen::Matrix<Scalar, -1, -1>;
template <typename Scalar>
using array1d_t = Eigen::Matrix<Scalar, -1, 1>;
template <typename Scalar>
using row_array1d_t = Eigen::Matrix<Scalar, 1, -1>;

template <typename T>
inline constexpr T pi() {
  return static_cast<T>(3.141592653589793238462643383279);
}

inline double eval(double x) { return x; }
template <typename T>
inline std::complex<T>& eval(std::complex<T>& x) {
  return x;
}
template <typename T>
inline std::complex<T> eval(std::complex<T>&& x) {
  return x;
}

template <typename T>
inline auto eval(T&& x) {
  return x.eval();
}

template <typename T, Eigen::Index R, Eigen::Index C>
inline Eigen::Matrix<T, R, C> eval(Eigen::Matrix<T, R, C>&& x) {
  return x;
}

template <typename T, Eigen::Index R, Eigen::Index C>
inline auto& eval(Eigen::Matrix<T, R, C>& x) {
  return x;
}

template <typename T, Eigen::Index R, Eigen::Index C>
inline const auto& eval(const Eigen::Matrix<T, R, C>& x) {
  return x;
}

template <typename T, Eigen::Index R, Eigen::Index C>
inline Eigen::Array<T, R, C> eval(Eigen::Array<T, R, C>&& x) {
  return x;
}

template <typename T, Eigen::Index R, Eigen::Index C>
inline auto& eval(Eigen::Array<T, R, C>& x) {
  return x;
}

template <typename T, Eigen::Index R, Eigen::Index C>
inline const auto& eval(const Eigen::Array<T, R, C>& x) {
  return x;
}

template <typename T, typename Scalar>
auto get_slice(T&& x_eval, Scalar start, Scalar end) {
  Eigen::Index i = 0;
  Eigen::Index dense_start = 0;
  for (; i < x_eval.size(); ++i) {
    if ((x_eval[i] >= start && x_eval[i] <= end)) {
      dense_start = i;
      break;
    }
  }
  Eigen::Index dense_size = 0;
  for (; i < x_eval.size(); ++i) {
    if ((x_eval[i] >= start && x_eval[i] <= end)) {
      dense_size++;
    } else {
      break;
    }
  }
  return std::make_pair(dense_start, dense_size);
}

template <typename T, int R, int C>
inline void print(const char* name, const Eigen::Matrix<T, R, C>& x) {
#ifdef RICCATI_DEBUG
  std::cout << name << "(" << x.rows() << ", " << x.cols() << ")" << std::endl;
  std::cout << x << std::endl;
#endif
}

template <typename T, int R, int C>
inline void print(const char* name, const Eigen::Array<T, R, C>& x) {
#ifdef RICCATI_DEBUG
  std::cout << name << "(" << x.rows() << ", " << x.cols() << ")" << std::endl;
  std::cout << x << std::endl;
#endif
}

template <
    typename T,
    std::enable_if_t<std::is_floating_point<std::decay_t<T>>::value
                     || std::is_integral<std::decay_t<T>>::value>* = nullptr>
inline void print(const char* name, T&& x) {
#ifdef RICCATI_DEBUG
  std::cout << name << ": " << std::setprecision(16) << x << std::endl;
#endif
}

template <
    typename T,
    std::enable_if_t<std::is_floating_point<std::decay_t<T>>::value
                     || std::is_integral<std::decay_t<T>>::value>* = nullptr>
inline void print(const char* name, const std::complex<T>& x) {
#ifdef RICCATI_DEBUG
  std::cout << name << ": (" << std::setprecision(16) << x.real() << ", "
            << x.imag() << ")" << std::endl;
#endif
}

}  // namespace riccati

#endif
