#include <riccati/riccati.hpp>
#include <boost/math/special_functions/airy.hpp>
#include <iostream>

using namespace riccati;
template <typename T>
inline auto airy_i_scalar(T&& xi) {
  return boost::math::airy_ai(-xi) + 
          std::complex<riccati::value_type_t<T>>(0.0, 1.0) * 
          boost::math::airy_bi(-xi);
}
template <typename T>
inline auto airy_i_prime_scalar(T&& xi) {
  return eval(-boost::math::airy_ai_prime(-xi)
              - std::complex<value_type_t<T>>(0.0, 1.0) * boost::math::airy_bi_prime(-xi));
}

int main() {
  using namespace riccati;
  // Build omega and gamma functions
  auto omega_scalar
      = [](auto&& x) { return std::sqrt(x); };
  auto gamma_scalar = [](auto&& x) { return 0.0; };
  // Vectorize the functions
  auto omega_fun = vectorize(omega_scalar);
  auto gamma_fun = vectorize(gamma_scalar);
  // Create arena allocator and solver info
  riccati::arena_alloc* arena{new riccati::arena_alloc{}};
  riccati::arena_allocator<double, riccati::arena_alloc> allocator{arena};

  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 8,
                                           32, 32, 32);
  // Initial conditions
  auto xi = 1e0;
  auto xf = 1e6;
  auto eps = 1e-12;
  auto epsh = 1e-13;
  auto yi = airy_i_scalar(xi);
  auto dyi = airy_i_prime_scalar(xi);
  Eigen::Index Neval = 1e3;
  riccati::vector_t<double> x_eval
      = riccati::vector_t<double>::LinSpaced(Neval, xi, 1e2);
  // Evolve the solution
  auto res = riccati::evolve(info, xi, xf, yi, dyi, eps, epsh, 0.5, x_eval);
  // Check the results
  auto airy_i = riccati::vectorize([](auto x) {return airy_i_scalar(x);});
  auto airy_i_prime = riccati::vectorize([](auto x) {return airy_i_prime_scalar(x);});
  auto ytrue = airy_i(x_eval).matrix().eval();
  auto dytrue = airy_i_prime(x_eval).matrix().eval();
  auto y_err
      = ((std::get<6>(res) - ytrue).array() / ytrue.array()).abs().eval();
  auto dy_err
      = ((std::get<7>(res) - dytrue).array() / dytrue.array()).abs().eval();
  std::cout << "Max y interp error: " << y_err.maxCoeff() << std::endl;
  std::cout << "Max dy interp error: " << dy_err.maxCoeff() << std::endl;
  return 0;
}
