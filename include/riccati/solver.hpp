#ifndef INCLUDE_RICCATI_SOLVER_HPP
#define INCLUDE_RICCATI_SOLVER_HPP

#include <riccati/arena_matrix.hpp>
#include <riccati/logger.hpp>
#include <riccati/memory.hpp>
#include <riccati/chebyshev.hpp>
#include <riccati/utils.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <functional>
#include <complex>
#include <cmath>
#include <vector>

namespace riccati {

template <typename SolverInfo, typename Scalar,
          require_floating_point<value_type_t<Scalar>>* = nullptr,
          require_not_same<typename std::decay_t<SolverInfo>::gamma_type,
                           pybind11::object>* = nullptr>
inline auto gamma(SolverInfo&& info, const Scalar& x) {
  return info.gamma_fun_(x);
}

template <
    typename SolverInfo, typename Scalar,
    require_floating_point<value_type_t<Scalar>>* = nullptr,
    std::enable_if_t<!std::is_same<typename std::decay_t<SolverInfo>::omega_type,
                                   pybind11::object>::value>* = nullptr>
inline auto omega(SolverInfo&& info, const Scalar& x) {
  return info.omega_fun_(x);
}

/**
 * @brief Class containing information about the solver.
 * @tparam OmegaFun Type of the frequency function. Must have a valid `operator()`
 *  whose input size and type is the same as its output size and type.
 * @tparam GammaFun Type of the friction function. Must have a valid `operator()`
 *  whose input size and type is the same as its output size and type.
 * @tparam Scalar_ Numeric scalar type, typically float or double.
 * @tparam Integral_ Numeric integral type, typically int or long.
 * @tparam Allocator Type of the allocator for the arena memory pool.
 * @tparam Logger Type of the logger for the solver. Must inherit from
 * @ref riccati::LoggerBase.
 */
template <typename OmegaFun, typename GammaFun, typename Scalar_,
          typename Integral_,
          typename Allocator = arena_allocator<Scalar_, arena_alloc>,
          typename Logger = ::riccati::EmptyLogger,
          typename OmegaReturn_ = double,
          typename GammaReturn_ = double>
class SolverInfo {
 public:
  using Scalar = Scalar_;
  using Integral = Integral_;
  using complex_t = std::complex<Scalar>;
  using matrixd_t = matrix_t<Scalar>;
  using vectorc_t = vector_t<complex_t>;
  using vectord_t = vector_t<Scalar>;
  using gamma_type = std::decay_t<GammaFun>;
  using omega_type = std::decay_t<OmegaFun>;
  using OmegaReturn = OmegaReturn_;
  using GammaReturn = GammaReturn_;
  // Frequency function
  OmegaFun omega_fun_;
  // Friction function
  GammaFun gamma_fun_;

  Allocator alloc_;
  // Number of nodes and diff matrices
  Integral n_nodes_{0};
  // Differentiation matrices and Vectors of Chebyshev nodes
  std::vector<std::tuple<Integral, matrixd_t, vectord_t>> chebyshev_;
  // Lengths of node vectors
  Integral n_idx_;
  Integral p_idx_;
  /**
   * Values of the independent variable evaluated at `p` points
   * over the interval [x, x + `h`] lying in between Chebyshev nodes, where x is
   * the value of the independent variable at the start of the current step and
   * `h` is the current stepsize. The in-between points :math:`\\tilde{x}_p` are
   * defined by
   * \f[
   * \\tilde{x}_p = \cos\left( \\frac{(2k + 1)\pi}{2p} \\right), \quad k = 0, 1,
   * \ldots p-1.
   * \f]
   */
  vectord_t xp_interp_;

  matrixd_t L_;
  // Clenshaw-Curtis quadrature weights
  vectord_t quadwts_;

  matrixd_t integration_matrix_;
  /**
   * Minimum and maximum (number of Chebyshev nodes - 1) to use inside
   * Chebyshev collocation steps. The step will use `nmax` nodes or the
   * minimum number of nodes necessary to achieve the required local error,
   * whichever is smaller. If `nmax` > 2`nini`, collocation steps will be
   * attempted with :math:`2^i` `nini` nodes at the ith iteration.
   */
  Integral nini_;
  Integral nmax_;
  // (Number of Chebyshev nodes - 1) to use for computing Riccati steps.
  Integral n_;
  // (Number of Chebyshev nodes - 1) to use for estimating Riccati stepsizes.
  Integral p_;
  Logger logger_;
 private:
  inline auto build_chebyshev(Integral nini, Integral n_nodes, Integral n,
                              Integral p) {
    std::vector<std::tuple<Integral, matrixd_t, vectord_t>> res;
    res.reserve(n_nodes + 1);
    // Compute Chebyshev nodes and differentiation matrices
    bool n_found = false;
    bool p_found = false;
    std::vector<int> cheb_nodes;
    cheb_nodes.reserve(n_nodes + 2);
    for (Integral i = 0; i <= n_nodes; ++i) {
      auto it_v = nini * std::pow(2, i);
      cheb_nodes.push_back(it_v);
      n_found |= (it_v == n);
      p_found |= (it_v == p);
    }
    if (!n_found) {
      cheb_nodes.push_back(n);
    }
    if (!p_found && n != p) {
      cheb_nodes.push_back(p);
    }
    std::sort(cheb_nodes.begin(), cheb_nodes.end());
    for (auto&& it_v : cheb_nodes) {
      auto cheb_v = chebyshev<Scalar>(it_v);
      res.emplace_back(it_v, std::move(cheb_v.first), std::move(cheb_v.second));
    }
    return res;
  }

 public:
  /**
   * @brief Construct a new Solver Info object
   * @tparam OmegaFun_ Type of the frequency function. Must be able to take in
   * and return scalars and vectors.
   * @tparam GammaFun_ Type of the friction function. Must be able to take in
   * and return scalars and vectors.
   * @param omega_fun Frequency function
   * @param gamma_fun Friction function
   * @param nini Minimum number of Chebyshev nodes to use inside Chebyshev
   * collocation steps.
   * @param nmax Maximum number of Chebyshev nodes to use inside Chebyshev
   * collocation steps.
   * @param n (Number of Chebyshev nodes - 1) to use inside Chebyshev
   * collocation steps.
   * @param p (Number of Chebyshev nodes - 1) to use for computing Riccati
   * steps.
   */
  template <typename OmegaFun_, typename GammaFun_, typename Allocator_>
  SolverInfo(OmegaFun_&& omega_fun, GammaFun_&& gamma_fun, Allocator_&& alloc,
             Logger&& logger,
             Integral nini, Integral nmax, Integral n, Integral p)
      : omega_fun_(std::forward<OmegaFun_>(omega_fun)),
        gamma_fun_(std::forward<GammaFun_>(gamma_fun)),
        alloc_(std::forward<Allocator_>(alloc)),
        n_nodes_(log2(nmax / nini) + 1),
        chebyshev_(build_chebyshev(nini, n_nodes_, n, p)),
        n_idx_(std::distance(
            chebyshev_.begin(),
            std::find_if(chebyshev_.begin(), chebyshev_.end(),
                         [n](auto& x) { return std::get<0>(x) == n; }))),
        p_idx_(std::distance(
            chebyshev_.begin(),
            std::find_if(chebyshev_.begin(), chebyshev_.end(),
                         [p](auto& x) { return std::get<0>(x) == p; }))),
        xp_interp_((vector_t<Scalar>::LinSpaced(
                        p, pi<Scalar>() / (2.0 * p),
                        pi<Scalar>() * (1.0 - (1.0 / (2.0 * p))))
                        .array())
                       .cos()
                       .matrix()),
        L_(interpolate(this->xp(), xp_interp_, dummy_allocator{})),
        quadwts_(quad_weights<Scalar>(n)),
        integration_matrix_(integration_matrix<Scalar>(n + 1)),
        nini_(nini),
        nmax_(nmax),
        n_(n),
        p_(p),
        logger_(std::move(logger)) {
        }

  template <typename OmegaFun_, typename GammaFun_>
  SolverInfo(OmegaFun_&& omega_fun, GammaFun_&& gamma_fun, Integral nini,
             Integral nmax, Integral n, Integral p)
      : omega_fun_(std::forward<OmegaFun_>(omega_fun)),
        gamma_fun_(std::forward<GammaFun_>(gamma_fun)),
        alloc_(),
        n_nodes_(log2(nmax / nini) + 1),
        chebyshev_(build_chebyshev(nini, n_nodes_, n, p)),
        n_idx_(std::distance(
            chebyshev_.begin(),
            std::find_if(chebyshev_.begin(), chebyshev_.end(),
                         [n](auto& x) { return std::get<0>(x) == n; }))),
        p_idx_(std::distance(
            chebyshev_.begin(),
            std::find_if(chebyshev_.begin(), chebyshev_.end(),
                         [p](auto& x) { return std::get<0>(x) == p; }))),
        xp_interp_((vector_t<Scalar>::LinSpaced(
                        p, pi<Scalar>() / (2.0 * p),
                        pi<Scalar>() * (1.0 - (1.0 / (2.0 * p))))
                        .array())
                       .cos()
                       .matrix()),
        L_(interpolate(this->xp(), xp_interp_, dummy_allocator{})),
        quadwts_(quad_weights<Scalar>(n)),
        integration_matrix_(integration_matrix<Scalar>(n + 1)),
        nini_(nini),
        nmax_(nmax),
        n_(n),
        p_(p),
        logger_() {}
  template <typename OmegaFun_, typename GammaFun_, typename Allocator_>
  SolverInfo(OmegaFun_&& omega_fun, GammaFun_&& gamma_fun, Allocator_&& alloc, Integral nini,
             Integral nmax, Integral n, Integral p)
      : omega_fun_(std::forward<OmegaFun_>(omega_fun)),
        gamma_fun_(std::forward<GammaFun_>(gamma_fun)),
        alloc_(std::forward<Allocator_>(alloc)),
        n_nodes_(log2(nmax / nini) + 1),
        chebyshev_(build_chebyshev(nini, n_nodes_, n, p)),
        n_idx_(std::distance(
            chebyshev_.begin(),
            std::find_if(chebyshev_.begin(), chebyshev_.end(),
                         [n](auto& x) { return std::get<0>(x) == n; }))),
        p_idx_(std::distance(
            chebyshev_.begin(),
            std::find_if(chebyshev_.begin(), chebyshev_.end(),
                         [p](auto& x) { return std::get<0>(x) == p; }))),
        xp_interp_((vector_t<Scalar>::LinSpaced(
                        p, pi<Scalar>() / (2.0 * p),
                        pi<Scalar>() * (1.0 - (1.0 / (2.0 * p))))
                        .array())
                       .cos()
                       .matrix()),
        L_(interpolate(this->xp(), xp_interp_, dummy_allocator{})),
        quadwts_(quad_weights<Scalar>(n)),
        integration_matrix_(integration_matrix<Scalar>(n + 1)),
        nini_(nini),
        nmax_(nmax),
        n_(n),
        p_(p),
        logger_() {}
  auto info() const {
    return std::array<std::pair<LogInfo, std::size_t>, 5>{
      std::make_pair(LogInfo::CHEBNODES, chebyshev_.size()),
      std::make_pair(LogInfo::CHEBSTEP, 0), std::make_pair(LogInfo::CHEBITS, 0),
      std::make_pair(LogInfo::LS, 1), std::make_pair(LogInfo::RICCSTEP, 0)};
  }
  void mem_info() {
#ifdef RICCATI_DEBUG
    auto* mem = alloc_.alloc_;
    std::cout << "mem info: " << std::endl;
    std::cout << "blocks_ size: " << mem->blocks_.size() << std::endl;
    std::cout << "sizes_ size: " << mem->sizes_.size() << std::endl;
    std::cout << "cur_block_: " << mem->cur_block_ << std::endl;
    std::cout << "cur_block_end_: " << mem->cur_block_end_ << std::endl;
    std::cout << "next_loc_: " << mem->next_loc_ << std::endl;
#endif
  }

  RICCATI_ALWAYS_INLINE const auto& Dn() const noexcept {
    return std::get<1>(chebyshev_[n_idx_]);
  }
  RICCATI_ALWAYS_INLINE const auto& Dn(std::size_t idx) const noexcept {
    return std::get<1>(chebyshev_[idx]);
  }

  RICCATI_ALWAYS_INLINE const auto cheby_size() const noexcept {
    return chebyshev_.size();
  }


  /**
   * Values of the independent variable evaluated at (`n` + 1) Chebyshev
   * nodes over the interval [x, x + `h`], where x is the value of the
   * independent variable at the start of the current step and `h` is the
   * current stepsize.
   */
  RICCATI_ALWAYS_INLINE const auto& xn() const noexcept {
    return std::get<2>(chebyshev_[n_idx_]);
  }

  /**
   * Values of the independent variable evaluated at (`p` + 1) Chebyshev
   * nodes over the interval [x, x + `h`], where x is the value of the
   * independent variable at the start of the current step and `h` is the
   * current stepsize.
   */
  RICCATI_ALWAYS_INLINE const auto& xp() const noexcept {
    return std::get<2>(chebyshev_[p_idx_]);
  }

  RICCATI_ALWAYS_INLINE const auto& xp_interp() const noexcept {
    return xp_interp_;
  }

  /**
   * Interpolation matrix of size (`p`+1, `p`), used for interpolating
   * function between the nodes `xp` and `xpinterp` (for computing Riccati
   * stepsizes).
   */
  RICCATI_ALWAYS_INLINE const auto& L() const noexcept { return L_; }

  RICCATI_ALWAYS_INLINE Logger& logger() noexcept { return logger_; }
};

/**
 * @brief Construct a new Solver Info object
 * @tparam Scalar A scalar type used for calculations
 * @tparam OmegaFun Type of the frequency function. Must be able to take in and
 *  return scalars and vectors.
 * @tparam GammaFun Type of the friction function. Must be able to take in and
 *  return scalars and vectors.
 * @tparam Integral An integral type
 * @param omega_fun Frequency function
 * @param gamma_fun Friction function
 * @param nini Minimum number of Chebyshev nodes to use inside Chebyshev
 * collocation steps.
 * @param nmax Maximum number of Chebyshev nodes to use inside Chebyshev
 * collocation steps.
 * @param n (Number of Chebyshev nodes - 1) to use inside Chebyshev
 * collocation steps.
 * @param p (Number of Chebyshev nodes - 1) to use for computing Riccati
 * steps.
 */
template <typename Scalar, typename OmegaFun, typename Allocator,
          typename GammaFun, typename Integral, typename Logger = ::riccati::EmptyLogger>
inline auto make_solver(OmegaFun&& omega_fun, GammaFun&& gamma_fun,
                        Allocator&& alloc, Integral nini, Integral nmax,
                        Integral n, Integral p, Logger&& logger = Logger{}) {
  return SolverInfo<std::decay_t<OmegaFun>, std::decay_t<GammaFun>, Scalar,
                    Integral, std::decay_t<Allocator>, std::decay_t<Logger>>(
      std::forward<OmegaFun>(omega_fun), std::forward<GammaFun>(gamma_fun),
      std::forward<Allocator>(alloc), std::move(logger), nini, nmax, n, p);
}

}  // namespace riccati

#endif
