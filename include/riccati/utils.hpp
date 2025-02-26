#ifndef INCLUDE_RICCATI_UTILS_HPP
#define INCLUDE_RICCATI_UTILS_HPP
#include <riccati/macros.hpp>
#include <Eigen/Dense>
#include <type_traits>
#define RICCATI_DEBUG
#ifdef RICCATI_DEBUG
#include <iostream>
#include <iomanip>
#endif
#include <chrono>
#include <ctime>
#include <stdio.h>
namespace pybind11 {
class object;
}

namespace riccati {

template <typename T>
constexpr Eigen::Index compile_size_v
    = std::decay_t<T>::RowsAtCompileTime * std::decay_t<T>::ColsAtCompileTime;

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
RICCATI_ALWAYS_INLINE auto scale(Vector&& x, Scalar x0, Scalar h) {
  return (x0 + (h / 2.0) + (h / 2.0) * x.array()).matrix();
}

/**
 *
 */
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

template <typename Scalar>
using promote_complex_t = std::conditional_t<std::is_floating_point_v<std::decay_t<Scalar>>,
                                             std::complex<std::decay_t<Scalar>>, std::decay_t<Scalar>>;

template <typename T>
inline constexpr T pi() {
  return static_cast<T>(3.141592653589793238462643383279);
}

RICCATI_ALWAYS_INLINE double eval(double x) noexcept { return x; }
template <typename T>
RICCATI_ALWAYS_INLINE std::complex<T>& eval(std::complex<T>& x) noexcept {
  return x;
}
template <typename T>
RICCATI_ALWAYS_INLINE std::complex<T> eval(std::complex<T>&& x) {
  return x;
}

template <typename T>
RICCATI_ALWAYS_INLINE auto eval(T&& x) {
  return x.eval();
}

template <typename T, Eigen::Index R, Eigen::Index C>
RICCATI_ALWAYS_INLINE Eigen::Matrix<T, R, C> eval(Eigen::Matrix<T, R, C>&& x) {
  return std::move(x);
}

template <typename T, Eigen::Index R, Eigen::Index C>
RICCATI_ALWAYS_INLINE auto& eval(Eigen::Matrix<T, R, C>& x) {
  return x;
}

template <typename T, Eigen::Index R, Eigen::Index C>
RICCATI_ALWAYS_INLINE const auto& eval(const Eigen::Matrix<T, R, C>& x) {
  return x;
}

template <typename T, Eigen::Index R, Eigen::Index C>
RICCATI_ALWAYS_INLINE Eigen::Array<T, R, C> eval(Eigen::Array<T, R, C>&& x) {
  return std::move(x);
}

template <typename T, Eigen::Index R, Eigen::Index C>
RICCATI_ALWAYS_INLINE auto& eval(Eigen::Array<T, R, C>& x) {
  return x;
}

template <typename T, Eigen::Index R, Eigen::Index C>
RICCATI_ALWAYS_INLINE const auto& eval(const Eigen::Array<T, R, C>& x) {
  return x;
}

template <typename T, typename Scalar>
RICCATI_ALWAYS_INLINE auto get_slice(T&& x_eval, Scalar start, Scalar end) {
  Eigen::Index i = 0;
  Eigen::Index dense_start = 0;
  if (start > end) {
    std::swap(start, end);
  }
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

template <template <typename> class Base, typename Derived>
inline constexpr bool is_base_pointer_convertible_v = is_base_pointer_convertible<Base, Derived>::value;

template <typename T>
struct is_eigen
    : std::bool_constant<is_base_pointer_convertible_v<Eigen::EigenBase, std::decay_t<T>>> {};

template <typename T>
inline constexpr bool is_eigen_v = is_eigen<T>::value;

template <typename T>
using require_not_floating_point
    = std::enable_if_t<!std::is_floating_point_v<std::decay_t<T>>>;

template <typename T>
using require_floating_point
    = std::enable_if_t<std::is_floating_point_v<std::decay_t<T>>>;

template <typename T1, typename T2>
using require_same
    = std::enable_if_t<std::is_same_v<std::decay_t<T1>, std::decay_t<T2>>>;

template <typename T1, typename T2>
using require_not_same = std::enable_if_t<!std::is_same_v<std::decay_t<T1>, std::decay_t<T2>>>;

template <typename MatrixType>
class arena_matrix;

namespace internal {
template <typename T, typename Enable = void>
struct value_type_impl {
  static_assert(1,"Should never be used!");
  using type = T;
};
template <typename T>
struct value_type_impl<T, std::enable_if_t<std::is_arithmetic_v<T>>> {
  using type = T;
};

template <typename T>
struct value_type_impl<T, std::enable_if_t<is_eigen_v<T>>> {
  using type = typename std::decay_t<T>::Scalar;
};

}  // namespace internal

template <typename T>
using value_type_t = typename internal::value_type_impl<std::decay_t<T>>::type;

namespace internal {
template <typename T>
struct is_complex_impl : std::false_type {};
template <typename T>
struct is_complex_impl<std::complex<T>> : std::true_type {};
}  // namespace internal

template <typename T>
struct is_complex : internal::is_complex_impl<std::decay_t<T>> {};

template <typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

template <typename T>
using require_floating_point_or_complex
    = std::enable_if_t<std::is_floating_point<std::decay_t<T>>::value
                       || is_complex<std::decay_t<T>>::value>;

template <typename T>
using require_not_floating_point_or_complex
    = std::enable_if_t<!std::is_floating_point<std::decay_t<T>>::value
                       && !is_complex<std::decay_t<T>>::value>;

template <typename T>
using require_eigen
    = std::enable_if_t<is_eigen_v<std::decay_t<T>>>;


namespace internal {
template <typename>
struct is_pair : std::false_type {};

template <typename T, typename U>
struct is_pair<std::pair<T, U>> : std::true_type {};

}

template <typename T>
struct is_pair : internal::is_pair<std::decay_t<T>> {};

template <typename T>
inline constexpr bool is_pair_v = is_pair<std::decay_t<T>>::value;

template <typename T, require_floating_point_or_complex<T>* = nullptr>
RICCATI_ALWAYS_INLINE auto sin(T x) {
  return std::sin(x);
}

template <typename T, require_eigen<T>* = nullptr>
RICCATI_ALWAYS_INLINE auto sin(T&& x) {
  return x.sin();
}

template <typename T, require_floating_point_or_complex<T>* = nullptr>
RICCATI_ALWAYS_INLINE auto cos(T x) {
  return std::cos(x);
}

template <typename T, require_eigen<T>* = nullptr>
RICCATI_ALWAYS_INLINE auto cos(T&& x) {
  return x.cos();
}

template <typename T, require_floating_point_or_complex<T>* = nullptr>
RICCATI_ALWAYS_INLINE auto sqrt(T x) {
  return std::sqrt(x);
}

template <typename T, require_eigen<T>* = nullptr>
RICCATI_ALWAYS_INLINE auto sqrt(T&& x) {
  return x.sqrt();
}

template <typename T, require_floating_point_or_complex<T>* = nullptr>
RICCATI_ALWAYS_INLINE auto square(T x) {
  return x * x;
}

template <typename T, require_eigen<T>* = nullptr>
RICCATI_ALWAYS_INLINE auto square(T&& x) {
  return x.square();
}

template <typename T, require_floating_point_or_complex<T>* = nullptr>
RICCATI_ALWAYS_INLINE auto array(T x) {
  return x;
}

template <typename T, require_eigen<T>* = nullptr>
RICCATI_ALWAYS_INLINE auto array(T&& x) {
  return x.array();
}

template <typename T, require_floating_point_or_complex<T>* = nullptr>
RICCATI_ALWAYS_INLINE auto matrix(T x) {
  return x;
}

template <typename T, require_eigen<T>* = nullptr>
RICCATI_ALWAYS_INLINE auto matrix(T&& x) {
  return x.matrix();
}

template <typename T, require_floating_point_or_complex<T>* = nullptr>
RICCATI_ALWAYS_INLINE constexpr T zero_like(T x) {
  return static_cast<T>(0.0);
}

template <typename T, require_eigen<T>* = nullptr>
RICCATI_ALWAYS_INLINE auto zero_like(const T& x) {
  return std::decay_t<typename T::PlainObject>::Zero(x.rows(), x.cols());
}

template <typename T1, typename T2, require_floating_point<T1>* = nullptr>
RICCATI_ALWAYS_INLINE auto pow(T1 x, T2 y) {
  return std::pow(x, y);
}

template <typename T1, typename T2, require_eigen<T1>* = nullptr>
RICCATI_ALWAYS_INLINE auto pow(T1&& x, T2 y) {
  return x.array().pow(y);
}

template <typename T1, require_floating_point_or_complex<T1>* = nullptr>
RICCATI_ALWAYS_INLINE auto real(T1 x) {
  return std::real(x);
}

template <typename T1, require_eigen<T1>* = nullptr>
RICCATI_ALWAYS_INLINE auto real(T1&& x) {
  return x.real();
}

template <typename T, require_floating_point_or_complex<T>* = nullptr>
RICCATI_ALWAYS_INLINE auto to_complex(T x) {
  if constexpr (is_complex_v<value_type_t<T>>) {
    return x;
  } else {
    return std::complex(x);
  }
}

template <typename T, require_eigen<T>* = nullptr>
RICCATI_ALWAYS_INLINE auto to_complex(T&& x) {
  if constexpr (is_complex_v<value_type_t<T>>) {
    return std::forward<T>(x);
  } else {
    return x.template cast<std::complex<value_type_t<T>>>();
  }
}


template <typename T, int R, int C>
inline void print(const char* name, const Eigen::Matrix<T, R, C>& x) {
#ifdef RICCATI_DEBUG
  std::cout << name << "(" << x.rows() << ", " << x.cols() << ")" << std::endl;
  if (R == 1 || C == 1) {
    Eigen::IOFormat numpyFormat(Eigen::FullPrecision, Eigen::DontAlignCols,
                                ", ", ", ", "", "", "np.array([", "])");
    std::cout << x.transpose().eval().format(numpyFormat) << std::endl;
  } else {
    Eigen::IOFormat numpyFormat(Eigen::FullPrecision, Eigen::DontAlignCols,
                                ", ", ", ", "[", "]", "np.array([", "])");
    std::cout << x.format(numpyFormat) << std::endl;
  }
#endif
}

template <typename T, int R, int C>
inline void print(const char* name, const Eigen::Array<T, R, C>& x) {
#ifdef RICCATI_DEBUG
  std::cout << name << "(" << x.rows() << ", " << x.cols() << ")" << std::endl;
  if (R == 1 || C == 1) {
    Eigen::IOFormat numpyFormat(Eigen::FullPrecision, Eigen::DontAlignCols,
                                ", ", ", ", "", "", "np.array([", "])");
    std::cout << x.transpose().eval().format(numpyFormat) << std::endl;
  } else {
    Eigen::IOFormat numpyFormat(Eigen::FullPrecision, Eigen::DontAlignCols,
                                ", ", ", ", "[", "]", "np.array([", "])");
    std::cout << x.format(numpyFormat) << std::endl;
  }
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

template <typename T>
inline void print(const char* name, const std::vector<T>& x) {
  for (int i = 0; i < x.size(); ++i) {
    print((std::string(name) + std::string("[") + std::to_string(i) + "]")
              .c_str(),
          x[i]);
  }
}

RICCATI_ALWAYS_INLINE void local_time(const time_t* timer, struct tm* buf) noexcept {
#ifdef _WIN32
  // Windows switches the order of the arguments?
  localtime_s(buf, timer);
#else
  localtime_r(timer, buf);
#endif
}

/* Get the current time with microseconds */
RICCATI_ALWAYS_INLINE std::string time_mi() noexcept {
    auto now = std::chrono::system_clock::now();
    time_t epoch = std::chrono::system_clock::to_time_t(now);
    struct tm tms{};
    ::riccati::local_time(&epoch, &tms);
    auto fractional_seconds = now - std::chrono::system_clock::from_time_t(epoch);
    int micros = std::chrono::duration_cast<std::chrono::microseconds>(fractional_seconds).count();
    // Format the time string
    char buf[sizeof "[9999-12-31 29:59:59.999999]"];
    size_t nb = strftime(buf, sizeof(buf), "[%Y-%m-%d %H:%M:%S", &tms);
    nb += snprintf(&buf[nb], sizeof(buf) - nb, ".%06d]", micros);
    // Return the formatted string
    return std::string(buf, nb);
}


}  // namespace riccati

#endif
