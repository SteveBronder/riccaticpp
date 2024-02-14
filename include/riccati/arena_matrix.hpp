#ifndef STAN_MATH_REV_CORE_ARENA_MATRIX_HPP
#define STAN_MATH_REV_CORE_ARENA_MATRIX_HPP

#include <riccati/macros.hpp>
#include <riccati/memory.hpp>
#include <riccati/utils.hpp>
#include <Eigen/Dense>
#include <type_traits>

namespace riccati {

/**
 * Equivalent to `Eigen::Matrix`, except that the data is stored on AD stack.
 * That makes these objects triviali destructible and usable in `vari`s.
 *
 * @tparam MatrixType Eigen matrix type this works as (`MatrixXd`, `VectorXd`
 * ...)
 */
template <typename MatrixType>
class arena_matrix : public Eigen::Map<MatrixType> {
 public:
  using Scalar = typename std::decay_t<MatrixType>::Scalar;
  using Base = Eigen::Map<MatrixType>;
  using PlainObject = std::decay_t<MatrixType>;
  typedef typename Eigen::internal::remove_all<Base>::type NestedExpression;
  static constexpr int RowsAtCompileTime = MatrixType::RowsAtCompileTime;
  static constexpr int ColsAtCompileTime = MatrixType::ColsAtCompileTime;
  using allocator_t = arena_allocator<Scalar, arena_alloc>;
  allocator_t allocator_;
  /**
   * Default constructor.
   */
  template <typename T>
  arena_matrix(arena_allocator<T, arena_alloc>& allocator)
      : Base::Map(nullptr,
                  RowsAtCompileTime == Eigen::Dynamic ? 0 : RowsAtCompileTime,
                  ColsAtCompileTime == Eigen::Dynamic ? 0 : ColsAtCompileTime),
        allocator_(allocator) {}

  /**
   * Constructs `arena_matrix` with given number of rows and columns.
   * @param rows number of rows
   * @param cols number of columns
   */
  template <typename T>
  arena_matrix(arena_allocator<T, arena_alloc>& allocator, Eigen::Index rows,
               Eigen::Index cols)
      : Base::Map(allocator.template allocate<Scalar>(rows * cols), rows, cols),
        allocator_(allocator) {}

  /**
   * Constructs `arena_matrix` with given size. This only works if
   * `MatrixType` is row or col vector.
   * @param size number of elements
   */
  template <typename T>
  arena_matrix(arena_allocator<T, arena_alloc>& allocator, Eigen::Index size)
      : Base::Map(allocator_.template allocate<Scalar>(size), size),
        allocator_(allocator) {}

  /**
   * Constructs `arena_matrix` from an expression.
   * @param other expression
   */
  template <typename T, typename Expr>
  arena_matrix(arena_allocator<T, arena_alloc>& allocator,
               const Expr& other)  // NOLINT
      : Base::Map(
          allocator.template allocate<Scalar>(other.size()),
          (RowsAtCompileTime == 1 && Expr::ColsAtCompileTime == 1)
                  || (ColsAtCompileTime == 1 && Expr::RowsAtCompileTime == 1)
              ? other.cols()
              : other.rows(),
          (RowsAtCompileTime == 1 && Expr::ColsAtCompileTime == 1)
                  || (ColsAtCompileTime == 1 && Expr::RowsAtCompileTime == 1)
              ? other.rows()
              : other.cols()),
        allocator_(allocator) {
    (*this).noalias() = other;
  }

  /**
   * Constructs `arena_matrix` from an expression. This makes an assumption that
   * any other `Eigen::Map` also contains memory allocated in the arena.
   * @param other expression
   */
  arena_matrix(const Base& other)  // NOLINT
      : Base::Map(other) {}

  /**
   * Copy constructor.
   * @param other matrix to copy from
   */
  arena_matrix(const arena_matrix<MatrixType>& other)
      : Base::Map(const_cast<Scalar*>(other.data()), other.rows(),
                  other.cols()),
        allocator_(other.allocator_) {}

  // without this using, compiler prefers combination of implicit construction
  // and copy assignment to the inherited operator when assigned an expression
  using Base::operator=;

  /**
   * Copy assignment operator.
   * @param other matrix to copy from
   * @return `*this`
   */
  arena_matrix& operator=(const arena_matrix<MatrixType>& other) {
    // placement new changes what data map points to - there is no allocation
    new (this)
        Base(const_cast<Scalar*>(other.data()), other.rows(), other.cols());
    this->allocator_ = other.allocator_;
    return *this;
  }

  /**
   * Assignment operator for assigning an expression.
   * @param a expression to evaluate into this
   * @return `*this`
   */
  template <typename T>
  arena_matrix& operator=(const T& a) {
    // do we need to transpose?
    if ((RowsAtCompileTime == 1 && T::ColsAtCompileTime == 1)
        || (ColsAtCompileTime == 1 && T::RowsAtCompileTime == 1)) {
      // placement new changes what data map points to - there is no allocation
      new (this) Base(allocator_.template allocate<Scalar>(a.size()), a.cols(),
                      a.rows());

    } else {
      new (this) Base(allocator_.template allocate<Scalar>(a.size()), a.rows(),
                      a.cols());
    }
    Base::operator=(a);
    return *this;
  }
};

template <typename T, typename Expr>
inline auto to_arena(arena_allocator<T, arena_alloc>& arena,
                     const Expr& expr) noexcept {
  return arena_matrix<typename std::decay_t<Expr>::PlainObject>(arena, expr);
}

template <typename T, typename Expr>
RICCATI_ALWAYS_INLINE auto eval(arena_allocator<T, arena_alloc>& arena,
                     const Expr& expr) noexcept {
  return arena_matrix<typename std::decay_t<Expr>::PlainObject>(arena, expr);
}


template <typename Expr>
inline auto to_arena(dummy_allocator& arena, const Expr& expr) noexcept {
  return eval(expr);
}

template <typename T>
inline void print(const char* name, const arena_matrix<T>& x) {
#ifdef RICCATI_DEBUG
  std::cout << name << "(" << x.rows() << ", " << x.cols() << ")" << std::endl;
  std::cout << x << std::endl;
#endif
}

}  // namespace riccati

namespace Eigen {
namespace internal {

template <typename T>
struct traits<riccati::arena_matrix<T>> : traits<Eigen::Map<T>> {
  using base = traits<Eigen::Map<T>>;
  using XprKind = typename Eigen::internal::traits<std::decay_t<T>>::XprKind;
  using Scalar = typename std::decay_t<T>::Scalar;
  enum {
    PlainObjectTypeInnerSize = base::PlainObjectTypeInnerSize,
    InnerStrideAtCompileTime = base::InnerStrideAtCompileTime,
    OuterStrideAtCompileTime = base::OuterStrideAtCompileTime,
    Alignment = base::Alignment,
    Flags = base::Flags
  };
};

}  // namespace internal
}  // namespace Eigen

#endif
