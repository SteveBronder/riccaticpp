
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
  auto potential = [l](auto&& x_arr) {
    return (x_arr * x_arr) + l * (x_arr * x_arr * x_arr * x_arr);
  };
  auto omega_fun
      = [current_energy, potential, m](auto&& x) {
        return eval(matrix(riccati::sqrt(2.0 * m * (std::complex(current_energy) - potential(array(x))))));
        };
  auto gamma_fun = [](auto&& x) { return to_complex(zero_like(x)); };
  //auto cout_ptr = std::unique_ptr<std::ostream, deleter_noop>(&std::cout, deleter_noop{});
  //DefaultLogger<std::ostream, deleter_noop> logger{std::move(cout_ptr)};
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 16,
                                           35, 35, 35);//, std::move(logger));
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
  auto left_res
      = riccati::evolve(info, left_boundary, midpoint, yi, dyi, eps, epsh, init_step, x_eval, true, LogLevel::WARNING);
  auto left_y_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
      std::get<1>(left_res).data(), std::get<1>(left_res).size());
  auto left_dy_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
      std::get<2>(left_res).data(), std::get<2>(left_res).size());
  init_step_ = choose_osc_stepsize(info, right_boundary, right_boundary - midpoint, epsh);
  init_step = -std::get<0>(init_step_);
  auto right_res
      = riccati::evolve(info, right_boundary, midpoint, yi, dyi, eps, epsh, init_step, x_eval, true, LogLevel::WARNING);
  auto right_y_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
      std::get<1>(right_res).data(), std::get<1>(right_res).size());
  auto right_dy_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
      std::get<2>(right_res).data(), std::get<2>(right_res).size());
  auto energy_diff = std::abs(left_dy_est.tail(1)[0] / left_y_est.tail(1)[0] -
    right_dy_est.tail(1)[0] / right_y_est.tail(1)[0]);
  EXPECT_LE(energy_diff, 4e-3);
}
