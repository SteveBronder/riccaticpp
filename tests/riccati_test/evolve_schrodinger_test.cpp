
#include <riccati/evolve.hpp>
#include <riccati/solver.hpp>
#include <riccati/vectorizer.hpp>
#include <riccati_test/utils.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <string>
// TODO: Test more boundaries
TEST_F(Riccati, evolve_nondense_fwd_schrodinger) {
  using namespace riccati;
  double current_energy = 417.056;
  double l = 1.0;
  double m = 0.5;
  auto potential = [l](auto&& x) {
    auto&& x_arr = x;
    auto ret = eval((x_arr * x_arr) + l * (x_arr * x_arr * x_arr * x_arr));
    std::cout << "potential: \n" << ret << std::endl;
    return ret;
  };
  auto omega_fun
      = [current_energy, potential, m](auto&& x) {
        return eval(matrix(riccati::sqrt(2.0 * m * (std::complex(current_energy) - potential(array(x))))));
        };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto cout_ptr = std::unique_ptr<std::ostream, deleter_noop>(&std::cout, deleter_noop{});
  DefaultLogger<std::ostream, deleter_noop> logger{std::move(cout_ptr)};
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 16,
                                           35, 35, 35, std::move(logger));
  const double left_boundary = -std::pow(current_energy, 0.25) - 2.0;
  const double right_boundary = -left_boundary;
  const double midpoint = 0.5;
  constexpr auto eps = 1e-12;
  constexpr auto epsh = 1e-13;
  auto yi = std::complex(0.0, 0.0);
  auto dyi = std::complex(1e-3, 0.0);
  Eigen::Matrix<double, 0, 0> x_eval;
  auto init_step_ = choose_osc_stepsize(info, left_boundary, midpoint - left_boundary, epsh);
  auto init_step = std::get<0>(init_step_);
  std::cout << "left: " << left_boundary << "\nright: " << right_boundary << "\nmidpoint: " << midpoint << std::endl;
  std::cout << "init_step: " << init_step << std::endl;
  auto res
      = riccati::evolve(info, left_boundary, midpoint, yi, dyi, eps, epsh, 1.0, x_eval, true, LogLevel::INFO);
  auto x_steps = Eigen::Map<Eigen::VectorXd>(std::get<0>(res).data(),
                                             std::get<0>(res).size());
  auto y_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
      std::get<1>(res).data(), std::get<1>(res).size());
  std::cout << y_est << std::endl;
}
