
#include <riccati/evolve.hpp>
#include <riccati/solver.hpp>
#include <riccati/vectorizer.hpp>
#include <riccati_test/utils.hpp>
#include <boost/math/tools/minima.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <string>

/*
 * Tests a path of the python riccati running through Brent with
 * eps 1e-12 and epsh 1e-13
 */
TEST_F(Riccati, evolve_schrodinger_nondense_fwd_path_optimize) {
  constexpr std::array energy_arr{
      21933.819660112502, 21936.180339887498, 21932.360679775,
      21932.79926820149,  21932.92560277868,  21932.717324526293,
      21932.79823117818,  21932.77380316041,  21932.752230241815,
      21932.783055066633, 21932.784813008315, 21932.782726417554,
      21932.783722645006, 21932.78413912673,  21932.78339399592,
      21932.783722645006};
  constexpr std::array energy_target{
      360.61859818087714,   3027.7780665966357, 142.72696896676825,
      5.159737196477863,    47.0246242296563,   22.04977431633256,
      4.8158692225837285,   3.287104498701524,  10.448477574824096,
      0.21748306005861195,  0.3656714725908614, 0.3265078696388173,
      0.003973495107516101, 0.1421312060404034, 0.10504908512007205,
      0.003973495107516101};
  using namespace riccati;
  constexpr double l = 1.0;
  constexpr double m = 0.5;
  auto potential = [l](auto&& x_arr) {
    auto x_square = riccati::square(x_arr);
    return eval(x_square + l * riccati::square(x_square));
  };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  // auto cout_ptr = std::unique_ptr<std::ostream, deleter_noop>(&std::cout,
  // deleter_noop{}); DefaultLogger<std::ostream, deleter_noop>
  // logger{std::move(cout_ptr)};
  constexpr auto eps = 1e-5;
  constexpr auto epsh = 1e-6;
  auto yi = std::complex(0.0, 0.0);
  auto dyi = std::complex(1e-3, 0.0);
  Eigen::Matrix<double, 0, 0> x_eval;
  for (long unsigned int iter = 0; iter < energy_target.size(); iter++) {
    auto current_energy = energy_arr[iter];
    auto target_energy_diff = energy_target[iter];
    auto omega_fun = [current_energy, potential, m](auto&& x) {
      return eval(matrix(riccati::sqrt(
          2.0 * m * (std::complex(current_energy) - potential(array(x))))));
    };
    auto cout_ptr = std::unique_ptr<std::ostream, deleter_noop>(&std::cout,
                                                                deleter_noop{});
    DefaultLogger<std::ostream, deleter_noop> logger{std::move(cout_ptr)};
    auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator,
                                             16, 35, 35, 35, std::move(logger));
    const double left_boundary = -std::pow(current_energy, 0.25) - 2.0;
    const double right_boundary = -left_boundary;
    constexpr double midpoint = 0.5;
    auto init_step = 0.1;
    auto left_res
        = riccati::evolve(info, left_boundary, midpoint, yi, dyi, eps, epsh,
                          init_step, x_eval, true, LogLevel::WARNING);
    auto left_y_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
        std::get<1>(left_res).data(), std::get<1>(left_res).size());
    auto left_dy_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
        std::get<2>(left_res).data(), std::get<2>(left_res).size());
    Eigen::Matrix<double, -1, 5> resp(left_y_est.size(), 5);
    resp.col(0) = Eigen::Map<Eigen::Matrix<double, -1, 1>>(
        std::get<0>(left_res).data() + 1, std::get<0>(left_res).size());
    resp.col(1) = left_y_est.real();
    resp.col(2) = left_y_est.imag();
    resp.col(3) = left_dy_est.real();
    resp.col(4) = left_dy_est.imag();
    init_step = -init_step;
    auto right_res
        = riccati::evolve(info, right_boundary, midpoint, yi, dyi, eps, epsh,
                          init_step, x_eval, true, LogLevel::WARNING);
    auto right_y_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
        std::get<1>(right_res).data(), std::get<1>(right_res).size());
    auto right_dy_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
        std::get<2>(right_res).data(), std::get<2>(right_res).size());
    Eigen::Matrix<double, -1, 5> resp_right(right_y_est.size(), 5);
    resp_right.col(0) = Eigen::Map<Eigen::Matrix<double, -1, 1>>(
        std::get<0>(right_res).data() + 1, std::get<0>(right_res).size());
    resp_right.col(1) = right_y_est.real();
    resp_right.col(2) = right_y_est.imag();
    resp_right.col(3) = right_dy_est.real();
    resp_right.col(4) = right_dy_est.imag();
    auto psi_l = left_y_est.tail(1)[0];
    auto dpsi_l = left_dy_est.tail(1)[0];
    auto psi_r = right_y_est.tail(1)[0];
    auto dpsi_r = right_dy_est.tail(1)[0];
    auto energy_diff = std::abs((dpsi_l / psi_l) - (dpsi_r / psi_r));
    auto abs_target_energy_diff = std::abs(energy_diff - target_energy_diff);
    EXPECT_LE(abs_target_energy_diff, 1e-4);
  }
}

TEST_F(Riccati, evolve_schrodinger_nondense_fwd_full_optimize) {
  auto energy_difference = [&](auto current_energy) {
    using namespace riccati;
    constexpr double l = 1.0;
    constexpr double m = 0.5;
    auto potential = [l](auto&& x_arr) {
      return (x_arr * x_arr) + l * (x_arr * x_arr * x_arr * x_arr);
    };
    auto gamma_fun = [](auto&& x) { return zero_like(x); };
    // auto cout_ptr = std::unique_ptr<std::ostream, deleter_noop>(&std::cout,
    // deleter_noop{}); DefaultLogger<std::ostream, deleter_noop>
    // logger{std::move(cout_ptr)};
    constexpr auto eps = 1e-5;
    constexpr auto epsh = 1e-6;
    std::cout << std::setprecision(30);
    auto yi = std::complex(0.0, 0.0);
    auto dyi = std::complex(1e-3, 0.0);
    Eigen::Matrix<double, 0, 0> x_eval;
    auto omega_fun = [current_energy, potential, m](auto&& x) {
      return eval(matrix(riccati::sqrt(
          2.0 * m * (std::complex(current_energy) - potential(array(x))))));
    };
    auto info
        = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 16, 35,
                                       32, 32);  //, std::move(logger));
    const double left_boundary = -std::pow(current_energy, 0.25) - 2.0;
    const double right_boundary = -left_boundary;
    constexpr double midpoint = 0.5;
    auto init_step = riccati::choose_nonosc_stepsize(
        info, left_boundary, midpoint - left_boundary, epsh);
    auto left_res
        = riccati::evolve(info, left_boundary, midpoint, yi, dyi, eps, epsh,
                          init_step, x_eval, true, LogLevel::WARNING);
    auto left_y_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
        std::get<1>(left_res).data(), std::get<1>(left_res).size());
    auto left_dy_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
        std::get<2>(left_res).data(), std::get<2>(left_res).size());
    init_step = -init_step;
    auto right_res
        = riccati::evolve(info, right_boundary, midpoint, yi, dyi, eps, epsh,
                          init_step, x_eval, true, LogLevel::WARNING);
    auto right_y_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
        std::get<1>(right_res).data(), std::get<1>(right_res).size());
    auto right_dy_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
        std::get<2>(right_res).data(), std::get<2>(right_res).size());
    auto psi_l = left_y_est.tail(1)[0];
    auto dpsi_l = left_dy_est.tail(1)[0];
    auto psi_r = right_y_est.tail(1)[0];
    auto dpsi_r = right_dy_est.tail(1)[0];
    auto energy_diff = std::abs((dpsi_l / psi_l) - (dpsi_r / psi_r));
    return energy_diff;
  };
  constexpr std::array bounds{
      std::make_pair(416.5, 417.5), std::make_pair(1035.0, 1037.0),
      std::make_pair(21930.0, 21939.0), std::make_pair(471100.0, 471110.0)};
  constexpr std::array reference_energy{417.056, 1035.544, 21932.783,
                                        471103.777};
  int sentinal = 0;
  for (auto& bound : bounds) {
    auto solve_ans = boost::math::tools::brent_find_minima(
        energy_difference, bound.first, bound.second,
        std::numeric_limits<double>::digits / 2);
    auto abs_target_diff
        = std::abs(solve_ans.first - reference_energy[sentinal]);
    sentinal++;
    // Brent is just not very good. For 471103.777, if it kept iterating
    // It would find a closer minimum but it stops short.
    EXPECT_LE(abs_target_diff, 8e-3);
  }
}
