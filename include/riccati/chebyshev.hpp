#ifndef INCLUDE_RICCATI_CHEBYSHEV_HPP
#define INCLUDE_RICCATI_CHEBYSHEV_HPP

#include <riccati/arena_matrix.hpp>
#include <riccati/logger.hpp>
#include <riccati/memory.hpp>
#include <riccati/utils.hpp>
#include <unsupported/Eigen/FFT>

namespace riccati {

namespace internal {
template <bool Fwd, typename T>
RICCATI_ALWAYS_INLINE auto fft(T&& x) {
  using Scalar = typename std::decay_t<T>::Scalar;
  Eigen::FFT<Scalar> fft;
  using T_t = std::decay_t<T>;
  typename T_t::PlainObject res(x.rows(), x.cols());
  for (Eigen::Index j = 0; j < x.cols(); ++j) {
    if (Fwd) {
      res.col(j) = fft.fwd(Eigen::Matrix<std::complex<Scalar>, -1, 1>(x.col(j)))
                       .real();
    } else {
      res.col(j) = fft.inv(Eigen::Matrix<std::complex<Scalar>, -1, 1>(x.col(j)))
                       .real();
    }
  }
  return res;
}
}  // namespace internal

/**
 * @brief Convert the Chebyshev coefficient representation of a set of
 * polynomials `P_j` to their values at Chebyshev nodes of the second kind.
 *
 * This function computes the values of a set of polynomials at Chebyshev nodes
 * of the second kind. The input is a matrix of coefficients `C`, where each
 * column represents a polynomial. The output is a matrix `V`, where `V(i,j)` is
 * the value of the j-th polynomial at the i-th Chebyshev node. The relationship
 * is given by:
 *
 * \f[
 * V_{ij} = P_j(x_i) = \sum_{k=0}^{n} C_{kj}T_k(x_i)
 * \f]
 *
 * This implementation is adapted from the `coeff2vals` function in the chebfun
 * package
 * (https://github.com/chebfun/chebfun/blob/master/%40chebtech2/coeffs2vals.m).
 *
 * @tparam Mat Eigen matrix type
 * @param coeffs A matrix of size (n+1, m), where the
 * (i, j)th element represents the projection of the j-th input polynomial onto
 * the i-th Chebyshev polynomial.
 * @return A matrix of size (n+1, m), where the (i,
 * j)th element represents the j-th input polynomial evaluated at the i-th
 * Chebyshev node.
 */
template <typename Mat>
RICCATI_ALWAYS_INLINE auto coeffs_to_cheby_nodes(Mat&& coeffs) {
  using Scalar = typename std::decay_t<Mat>::Scalar;
  const auto n = coeffs.rows();
  using Mat_t = matrix_t<Scalar>;
  if (n <= 1) {
    return Mat_t(coeffs);
  } else {
    Mat_t fwd_values(n + n - 2, coeffs.cols());
    fwd_values.topRows(n) = coeffs;
    fwd_values.block(1, 0, n - 2, n) /= 2.0;
    fwd_values.bottomRows(n - 1)
        = fwd_values.topRows(n - 1).rowwise().reverse();
    return Mat_t(internal::fft<true>(fwd_values).topRows(n).eval());
  }
}

/**
 * @brief Convert a matrix of values of `m` polynomials evaluated at `n+1`
 * Chebyshev nodes of the second kind to their interpolating Chebyshev
 * coefficients.
 *
 * This function computes the Chebyshev coefficients for a set of polynomials.
 * The input is a matrix `V`, where each column contains the values of a
 * polynomial at Chebyshev nodes. The output is a matrix `C`, where `C(i, j)` is
 * the coefficient of the i-th Chebyshev polynomial for the j-th input
 * polynomial. The relationship is given by:
 *
 * \f[
 * F_j(x) = \sum_{k=0}^{n} C_{kj}T_k(x)
 * \f]
 *
 * which interpolates the values `[V_{0j}, V_{1j}, ..., V_{nj}]` for `j =
 * 0...(m-1)`.
 *
 * This implementation is adapted from the `vals2coeffs` function in the chebfun
 * package
 * (https://github.com/chebfun/chebfun/blob/master/%40chebtech2/vals2coeffs.m).
 *
 * @tparam Mat Eigen matrix type
 * @param values A matrix of size (n+1, m), where the
 * (i, j)th element is the value of the j-th polynomial evaluated at the i-th
 * Chebyshev node.
 * @return A matrix of size (n+1, m), where the (i,
 * j)th element is the coefficient of the i-th Chebyshev polynomial for
 * interpolating the j-th input polynomial.
 */
template <typename Mat>
RICCATI_ALWAYS_INLINE auto cheby_nodes_to_coeffs(Mat&& values) {
  using Scalar = typename std::decay_t<Mat>::Scalar;
  using Mat_t = matrix_t<Scalar>;
  const auto n = values.rows();
  if (n <= 1) {
    return Mat_t(Mat_t::Zero(values.rows(), values.cols()));
  } else {
    Mat_t rev_values(n + n - 2, values.cols());
    rev_values.topRows(n) = values;
    rev_values.bottomRows(n - 1)
        = rev_values.topRows(n - 1).rowwise().reverse();
    auto rev_ret = Mat_t(internal::fft<false>(rev_values).topRows(n)).eval();
    rev_ret.block(1, 0, n - 2, n).array() *= 2.0;
    return rev_ret;
  }
}

/**
 * Returns both the Chebyshev coefficients and the Chebyshev nodes of the second
 * kind for a set of polynomials.
 *
 * @tparam Mat Eigen matrix type
 * @param values A matrix of size (n+1, m), where the
 * (i, j)th element is the value of the j-th polynomial evaluated at the i-th
 * Chebyshev node.
 */
template <typename Mat>
RICCATI_ALWAYS_INLINE auto coeffs_and_cheby_nodes(Mat&& values) {
  using Scalar = typename std::decay_t<Mat>::Scalar;
  using Mat_t = matrix_t<Scalar>;
  const auto n = values.rows();
  if (n <= 1) {
    return std::make_pair(Mat_t(values),
                          Mat_t(Mat_t::Zero(values.rows(), values.cols())));
  } else {
    Mat_t fwd_values(n + n - 2, values.cols());
    fwd_values.topRows(n) = values;
    fwd_values.block(1, 0, n - 2, n) /= 2.0;
    fwd_values.bottomRows(n - 1)
        = fwd_values.topRows(n - 1).rowwise().reverse();
    auto fwd_val = Mat_t(internal::fft<true>(fwd_values).topRows(n).eval());
    Mat_t rev_values(n + n - 2, values.cols());
    rev_values.topRows(n) = values;
    rev_values.bottomRows(n - 1)
        = rev_values.topRows(n - 1).rowwise().reverse();
    auto rev_ret = Mat_t(internal::fft<false>(rev_values).topRows(n)).eval();
    rev_ret.block(1, 0, n - 2, n).array() *= 2.0;
    return std::make_pair(fwd_val, rev_ret);
  }
}

/**
 * @brief Constructs a Chebyshev integration matrix.
 *
 * This function computes the Chebyshev integration matrix, which maps the
 * values of a function at `n` Chebyshev nodes of the second kind (ordered from
 * +1 to -1) to the values of the integral of the interpolating polynomial at
 * those nodes. The integral is computed on the interval defined by the
 * Chebyshev nodes, with the last value of the integral (at the start of the
 * interval) being set to zero. This implementation is adapted from the
 * `cumsummat` function in the chebfun package
 * (https://github.com/chebfun/chebfun/blob/master/%40chebcolloc2/chebcolloc2.m).
 *
 * @tparam Scalar The scalar type of the Chebyshev nodes and integration matrix
 * @tparam Integral The integral type of the number of Chebyshev nodes
 * @param n Number of Chebyshev nodes the integrand is evaluated at. The
 * nodes are ordered from +1 to -1.
 * @return Integration matrix of size (n, n). This
 * matrix maps the values of the integrand at the n Chebyshev nodes to the
 * values of the definite integral on the interval, up to each of the Chebyshev
 * nodes (the last value being zero by definition).
 */
template <typename Scalar, typename Integral>
RICCATI_ALWAYS_INLINE auto integration_matrix(Integral n) {
  auto ident = matrix_t<Scalar>::Identity(n, n).eval();
  auto coeffs_pair = coeffs_and_cheby_nodes(ident);
  auto&& T = coeffs_pair.first;
  auto&& T_inverse = coeffs_pair.second;
  n--;
  auto k = vector_t<Scalar>::LinSpaced(n, 1.0, n).eval();
  auto k2 = eval(2 * (k.array() - 1));
  k2.coeffRef(0) = 1.0;
  // B = np.diag(1 / (2 * k), -1) - np.diag(1 / k2, 1)
  matrix_t<Scalar> B = matrix_t<Scalar>::Zero(n + 1, n + 1);
  B.diagonal(-1).array() = 1.0 / (2.0 * k).array();
  B.diagonal(1).array() = -1.0 / k2.array();
  vector_t<Scalar> v = vector_t<Scalar>::Ones(n);
  for (Integral i = 1; i < n; i += 2) {
    v.coeffRef(i) = -1;
  }
  auto tmp = (v.asDiagonal() * B.block(1, 0, n, n + 1)).eval();
  B.row(0) = (tmp).colwise().sum();
  B.col(0) *= 2.0;
  auto Q = matrix_t<Scalar>(T * B * T_inverse);
  Q.bottomRows(1).setZero();
  return Q;
}

/**
 * @brief Calculates Clenshaw-Curtis quadrature weights.
 *
 * This function computes the Clenshaw-Curtis quadrature weights, which map
 * function evaluations at `n+1` Chebyshev nodes of the second kind (ordered
 * from +1 to -1) to the value of the definite integral of the interpolating
 * function on the same interval. The method is based on the Clenshaw-Curtis
 * quadrature formula, as described in Trefethen's "Spectral Methods in MATLAB"
 * (Chapter 12, `clencurt.m`).
 *
 * @tparam Scalar The scalar type of the quadrature weights
 * @tparam Integral The integral type of the number of Chebyshev nodes
 * @param n The number of Chebyshev nodes minus one, for which the
 * quadrature weights are to be computed.
 * @return A vector of size (n+1), containing the
 * quadrature weights.
 *
 * @note See the below for more information
 * Trefethen, Lloyd N. Spectral methods in MATLAB. Society for industrial and
 * applied mathematics, 2000.
 */
template <typename Scalar, typename Integral>
RICCATI_ALWAYS_INLINE auto quad_weights(Integral n) {
  vector_t<Scalar> w = vector_t<Scalar>::Zero(n + 1);
  if (n == 0) {
    return w;
  } else {
    auto a = vector_t<Scalar>::LinSpaced(n + 1, 0, pi<Scalar>()).eval();
    auto v = vector_t<Scalar>::Ones(n - 1).eval();
    // TODO: Smarter way to do this
    if (n % 2 == 0) {  // Check if n is even
      w[0] = 1.0 / (std::pow(n, 2) - 1);
      w[n] = w[0];
      for (int k = 1; k < static_cast<int>(std::floor(n / 2.0)); ++k) {
        v.array() -= 2.0
                     * ((2.0 * static_cast<Scalar>(k) * a.segment(1, n - 1))
                            .array()
                            .cos())
                           .array()
                     / (4.0 * k * k - 1);
      }
      v.array()
          -= ((n * a.segment(1, n - 1)).array().cos()) / (std::pow(n, 2) - 1);
    } else {  // If n is odd
      w[0] = 1.0 / std::pow(n, 2);
      w[n] = w[0];
      const auto max_val = static_cast<int>(std::floor((n + 1) / 2));
      for (int k = 1; k < max_val; ++k) {
        v.array() -= 2.0 * (2.0 * k * a.segment(1, n - 1).array()).cos()
                     / (4.0 * k * k - 1.0);
      }
    }
    w.segment(1, n - 1) = (2.0 * v / n).array();  // Set weights
    return w;
  }
}

/**
 * @brief Computes the Chebyshev differentiation matrix and Chebyshev nodes.
 *
 * This function calculates the Chebyshev differentiation matrix `D` of size
 * (n+1, n+1) and `n+1` Chebyshev nodes `x` for the standard 1D interval [-1,
 * 1]. The differentiation matrix `D` can be used to approximate the derivative
 * of a function sampled at the Chebyshev nodes. The nodes are computed
 * according to the formula:
 *
 * \f[
 * x_p = \cos \left( \frac{\pi p}{n} \right), \quad p = 0, 1, \ldots, n.
 * \f]
 *
 * @tparam Scalar The scalar type of the Chebyshev nodes and differentiation
 * @tparam Integral The integral type of the number of Chebyshev nodes
 * @param n int - The number of Chebyshev nodes minus one.
 * @return std::pair<matrix_t<Scalar>, vector_t<Scalar>> - A pair consisting of:
 *         1. The differentiation matrix `D` of size
 * (n+1, n+1).
 *         2. vector_t<Scalar> (real) - The vector of Chebyshev nodes `x` of
 * size (n+1), ordered in descending order from 1 to -1.
 */
template <typename Scalar, typename Integral>
RICCATI_ALWAYS_INLINE auto chebyshev(Integral n) {
  // Case when n == 0
  if (n == 0) {
    matrix_t<Scalar> D{{1}};
    vector_t<Scalar> x{{1}};
    return std::make_pair(D, x);
  } else {
    // Create the vector of Chebyshev nodes
    vector_t<Scalar> x
        = vector_t<Scalar>::LinSpaced(n + 1, 0.0, pi<Scalar>()).array().cos();
    vector_t<Scalar> b = vector_t<Scalar>::Ones(n + 1);
    b(0) = 2;
    b(n) = 2;
    vector_t<Scalar> d = vector_t<Scalar>::Ones(n + 1);
    for (int i = 1; i <= n; i += 2) {
      d(i) = -1;
    }
    auto c = b.array() * d.array();
    auto X = x * Eigen::RowVectorXd::Ones(n + 1);
    matrix_t<Scalar> D
        = (c.matrix() * (1.0 / c).matrix().transpose().matrix()).array()
          / ((X - X.transpose()).array()
             + matrix_t<Scalar>::Identity(n + 1, n + 1).array());
    D.diagonal() -= D.rowwise().sum();
    return std::make_pair(D, x);
  }
}

/**
 * @brief Creates an interpolation matrix from an array of source nodes to
 * target nodes.
 *
 * This function constructs an interpolation matrix that maps function values
 * known at source nodes `s` to estimated values at target nodes `t`. The
 * computation is based on the Vandermonde matrix approach and the resulting
 * matrix `L` applies the interpolation. The method is adapted from the
 * implementation provided `here`
 * (https://github.com/ahbarnett/BIE3D/blob/master/utils/interpmat_1d.m).
 *
 * @tparam Vec1 An Eigen vector type
 * @tparam Vec2 An Eigen vector type
 * @tparam Allocator An allocator type
 * @param s A vector specifying the source nodes, at
 * which the function values are known.
 * @param t A vector specifying the target nodes, at
 * which the function values are to be interpolated.
 * @param alloc An allocator for the Eigen objects.
 * @return The interpolation matrix `L`. If `s` has
 * size `p` and `t` has size `q`, then `L` has size (q, p). `L` takes function
 * values at source points `s` and yields the function evaluated at target
 * points `t`.
 */
template <typename Vec1, typename Vec2, typename Allocator>
RICCATI_ALWAYS_INLINE auto interpolate(Vec1&& s, Vec2&& t, Allocator&& alloc) {
  const auto r = s.size();
  const auto q = t.size();
  auto V
      = eval(alloc, matrix_t<typename std::decay_t<Vec1>::Scalar>::Ones(r, r));
  auto R
      = eval(alloc, matrix_t<typename std::decay_t<Vec1>::Scalar>::Ones(q, r));
  for (std::size_t i = 1; i < static_cast<std::size_t>(r); ++i) {
    V.col(i).array() = V.col(i - 1).array() * s.array();
    R.col(i).array() = R.col(i - 1).array() * t.array();
  }
  /*
  return V.transpose().jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                .solve(R.transpose())
                .transpose()
                .eval();
  */
  // Eigen::PartialPivLU<std::decay_t<decltype(V)>> lu(V.transpose());
  return V.transpose().partialPivLu().solve(R.transpose()).transpose().eval();
}

/**
 * @brief Applies a spectral collocation method based on Chebyshev nodes for
 * solving differential equations.
 *
 * This function utilizes a spectral collocation method based on Chebyshev nodes
 * to solve a differential equation over an interval from `x = x0` to `x = x0 +
 * h`. The solution starts from the initial conditions `y(x0) = y0` and `y'(x0)
 * = dy0`. In each iteration of the spectral collocation method, the number of
 * Chebyshev nodes used increases, starting from `info.nini` and doubling until
 * `info.nmax` is reached or the desired tolerance is met. The `niter` parameter
 * keeps track of the number of iterations and is used to retrieve pre-computed
 * differentiation matrices and Chebyshev nodes from the `info` object.
 *
 * @tparam SolverInfo A type containing pre-computed information for the solver,
 * like differentiation matrices and Chebyshev nodes.
 * @tparam Scalar The scalar type of the independent variable
 * @tparam YScalar The scalar type of the dependent variable
 * @tparam Integral The integral type of the number of iterations
 * @tparam Allocator An allocator type
 * @param info SolverInfo object - Contains pre-computed information for the
 * solver, like differentiation matrices and Chebyshev nodes.
 * @param x0 The starting value of the independent variable.
 * @param h Step size for the spectral method.
 * @param y0 Initial value of the dependent variable at `x0`.
 * @param dy0 Initial derivative of the dependent variable at `x0`.
 * @param niter Counter for the number of iterations of the spectral
 * collocation step performed.
 * @param alloc An allocator for the Eigen objects.
 * @return A tuple containing:
 *         1. Eigen::Vector<std::complex<Scalar>, Eigen::Dynamic, 1> - Numerical
 * estimate of the solution at the end of the step, at `x0 + h`.
 *         2. Eigen::Vector<std::complex<Scalar>, Eigen::Dynamic, 1> - Numerical
 * estimate of the derivative of the solution at the end of the step, at `x0 +
 * h`.
 *         3. vector_t<Scalar> (real) - Chebyshev nodes used for the current
 * iteration of the spectral collocation method, scaled to lie in the interval
 * `[x0, x0 + h]`.
 */
template <typename SolverInfo, typename Scalar, typename YScalar,
          typename Integral>
RICCATI_ALWAYS_INLINE auto spectral_chebyshev(SolverInfo&& info, Scalar x0,
                                              Scalar h, YScalar y0, YScalar dy0,
                                              Integral niter) {
  using complex_t = promote_complex_t<Scalar>;
  using vectorc_t = vector_t<complex_t>;
  auto x_scaled = eval(
      info.alloc_, riccati::scale(std::get<2>(info.chebyshev_[niter]), x0, h));
  auto&& D = info.Dn(niter);
  auto ws = omega(info, x_scaled);
  auto gs = gamma(info, x_scaled);
  auto D2 = eval(info.alloc_, ((D * D) + h * (gs.asDiagonal() * D))
                                  + ((ws * h / 2.0).array().square())
                                        .matrix()
                                        .asDiagonal()
                                        .toDenseMatrix());
  const auto n = std::round(std::get<0>(info.chebyshev_[niter]));
  auto D2ic = eval(info.alloc_, matrix_t<complex_t>::Zero(n + 3, n + 1));
  D2ic.topRows(n + 1) = D2;
  D2ic.row(n + 1) = D.row(D.rows() - 1);
  auto ic = eval(info.alloc_, vectorc_t::Zero(n + 1));
  ic.coeffRef(n) = complex_t{1.0, 0.0};
  D2ic.row(n + 2) = ic;
  auto rhs = eval(info.alloc_, vectorc_t::Zero(n + 3));
  rhs.coeffRef(n + 1) = dy0 * h / 2.0;
  rhs.coeffRef(n + 2) = y0;
  auto y1 = eval(info.alloc_, D2ic.colPivHouseholderQr().solve(rhs));
  auto dy1 = eval(info.alloc_, 2.0 / h * (D * y1));
  return std::make_tuple(std::move(y1), std::move(dy1), std::move(x_scaled));
}

}  // namespace riccati

#endif
