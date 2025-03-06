
#include <riccati/evolve.hpp>
#include <riccati/solver.hpp>
#include <riccati/vectorizer.hpp>
#include <riccati_test/utils.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <string>

TEST_F(Riccati, evolve_burst_dense_output) {
  using namespace riccati;
  constexpr double m = 1e6;
  auto omega_fun = [m](auto&& x) {
    return eval(matrix(riccati::sqrt(static_cast<double>(std::pow(m, 2)) - 1.0)
                       / (1 + riccati::pow(array(x), 2.0))));
  };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 16,
                                           35, 32, 32);
  constexpr double xi = -m;
  constexpr double xf = m;
  auto burst_y = [m](auto&& x) {
    return std::sqrt(1 + x * x) / m
           * (std::cos(m * std::atan(x))
              + std::complex<double>(0.0, 1.0) * std::sin(m * std::atan(x)));
  };
  auto yi = burst_y(xi);
  auto burst_dy = [m](auto&& x) {
    return (1.0 / std::sqrt(1.0 + x * x) / m
            * ((x + std::complex<double>(0.0, 1.0) * m)
                   * std::cos(m * std::atan(x))
               + (-m + std::complex<double>(0.0, 1.0) * x)
                     * std::sin(m * std::atan(x))));
  };
  auto dyi = burst_dy(xi);
  constexpr auto eps = 1e-12;
  constexpr auto epsh = 1e-13;
  Eigen::Index Neval = 1e3;
  riccati::vector_t<double> x_eval
      = riccati::vector_t<double>::LinSpaced(Neval, xi, xf);
  auto res = riccati::evolve(info, xi, xf, yi, dyi, eps, epsh, 0.1, x_eval);
  auto x_steps = Eigen::Map<Eigen::VectorXd>(std::get<0>(res).data(),
                                             std::get<0>(res).size());
  auto ytrue = x_steps.unaryExpr(burst_y).eval();
  auto y_steps = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
      std::get<1>(res).data(), std::get<1>(res).size());
  auto y_err
      = (((ytrue - y_steps).array()).abs() / (ytrue.array()).abs()).eval();
  EXPECT_LE(y_err.maxCoeff(), 1e-8);
}
