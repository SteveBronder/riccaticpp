
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
// TODO: Test more boundaries
/*
TEST_F(Riccati, evolve_nondense_fwd_schrodinger) {
  using namespace riccati;
  double current_energy = 417.056;
  double l = 1.0;
  double m = 0.5;
  std::cout << std::setprecision(14);
  auto potential = [l](auto&& x_arr) {
    return (x_arr * x_arr) +  (l * x_arr * x_arr * x_arr * x_arr);
  };
  auto omega_fun
      = [current_energy, potential, m](auto&& x) {
        return eval(matrix(riccati::sqrt(2.0 * m * (std::complex(current_energy) - potential(array(x))))));
        };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto cout_ptr = std::unique_ptr<std::ostream, deleter_noop>(&std::cout, deleter_noop{});
  DefaultLogger<std::ostream, deleter_noop> logger{std::move(cout_ptr)};
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 16,
                                           35, 32, 32, std::move(logger));
  const double left_boundary = -std::pow(current_energy, 0.25) - 2.0;
  const double right_boundary = -left_boundary;
  constexpr double midpoint = 0.5;
  constexpr auto eps = 1e-5;
  constexpr auto epsh = 1e-6;
  auto yi = std::complex(1e-3, 0.0);
  auto dyi = std::complex(1e-3, 0.0);
  Eigen::Matrix<double, 0, 0> x_eval;
  auto init_step_ = choose_osc_stepsize(info, left_boundary, midpoint - left_boundary, epsh);
//  auto init_step = std::get<0>(init_step_);
  auto init_step = 0.1;
  std::cout << "===LEFT SOLVE===\n";
  std::cout << "Info: \n";
  std::cout << "\t left_boundary: " << left_boundary << "\n";
  std::cout << "\t midpoint: " << midpoint << "\n";
  std::cout << "\t yi: " << yi << "\n";
  std::cout << "\t dyi: " << dyi << "\n";
  std::cout << "\t eps: " << eps << "\n";
  std::cout << "\t epsh: " << epsh << "\n";
  std::cout << "\t init_step: " << init_step << "\n";
  std::cout << "\t l: " << l << std::endl;
  std::cout << "\t m: " << m << std::endl;
  auto left_res
      = riccati::evolve(info, left_boundary, midpoint, yi, dyi, eps, epsh, init_step, x_eval, true, LogLevel::INFO);
  auto left_y_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
      std::get<1>(left_res).data(), std::get<1>(left_res).size());
  auto left_dy_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
      std::get<2>(left_res).data(), std::get<2>(left_res).size());
  init_step_ = choose_osc_stepsize(info, right_boundary, -(right_boundary - midpoint), epsh);
  //init_step = std::get<0>(init_step_);
  init_step = 0.1;
  if (init_step > 0) {
    init_step = -init_step;
  }
  std::cout << "===RIGHT SOLVE===\n";
  std::cout << "Info: \n";
  std::cout << "\t right_boundary: " << right_boundary << "\n";
  std::cout << "\t midpoint: " << midpoint << "\n";
  std::cout << "\t yi: " << yi << "\n";
  std::cout << "\t dyi: " << dyi << "\n";
  std::cout << "\t eps: " << eps << "\n";
  std::cout << "\t epsh: " << epsh << "\n";
  std::cout << "\t init_step: " << init_step << "\n";
  std::cout << "\t l: " << l << std::endl;
  std::cout << "\t m: " << m << std::endl;
  auto right_res
      = riccati::evolve(info, right_boundary, midpoint, yi, dyi, eps, epsh, init_step, x_eval, true, LogLevel::INFO);
  auto right_y_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
      std::get<1>(right_res).data(), std::get<1>(right_res).size());
  auto right_dy_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
      std::get<2>(right_res).data(), std::get<2>(right_res).size());
  auto energy_diff = std::abs(left_dy_est.tail(1)[0] / left_y_est.tail(1)[0] -
    right_dy_est.tail(1)[0] / right_y_est.tail(1)[0]);
    std::cout << "Energy Diff: " << energy_diff << "\n";
  EXPECT_LE(energy_diff, 4e-3);
}
*/
TEST_F(Riccati, evolve_nondense_fwd_left_optimize_schrodinger) {
  using namespace riccati;
  double l = 1.0;
  double m = 0.5;
  auto potential = [l](auto&& x_arr) {
    return (x_arr * x_arr) + l * (x_arr * x_arr * x_arr * x_arr);
  };
  std::cout << std::setprecision(18);
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  //auto cout_ptr = std::unique_ptr<std::ostream, deleter_noop>(&std::cout, deleter_noop{});
  //DefaultLogger<std::ostream, deleter_noop> logger{std::move(cout_ptr)};
  constexpr auto eps = 1e-5;
  constexpr auto epsh = 1e-6;
  auto yi = std::complex(0.0, 0.0);
  auto dyi = std::complex(1e-3, 0.0);
  Eigen::Matrix<double, 0, 0> x_eval;
  auto current_energy = 21936.1803400516510009765625;
  auto omega_fun
      = [current_energy, potential, m](auto&& x) {
        return eval(matrix(riccati::sqrt(2.0 * m * (std::complex(current_energy) - potential(array(x))))));
        };
  std::cout << std::setprecision(30);
  auto cout_ptr = std::unique_ptr<std::ostream, deleter_noop>(&std::cout, deleter_noop{});
  DefaultLogger<std::ostream, deleter_noop> logger{std::move(cout_ptr)};
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 16,
                                          35, 32, 32, std::move(logger));
  const double left_boundary = -std::pow(current_energy, 0.25) - 2.0;
  const double right_boundary = -left_boundary;
  constexpr double midpoint = 0.5;
  auto init_step = 0.1;
  std::cout << "===LEFT SOLVE===\n";
  std::cout << "Info: \n";
  std::cout << "\t left_boundary: " << left_boundary << "\n";
  std::cout << "\t midpoint: " << midpoint << "\n";
  std::cout << "\t right_boundary: " << right_boundary << "\n";
  std::cout << "\t yi: " << yi << "\n";
  std::cout << "\t dyi: " << dyi << "\n";
  std::cout << "\t eps: " << eps << "\n";
  std::cout << "\t epsh: " << epsh << "\n";
  std::cout << "\t init_step: " << init_step << "\n";
  std::cout << "\t l: " << l << std::endl;
  std::cout << "\t m: " << m << std::endl;
  std::cout << "START SOLVE\n";
  auto left_res
      = riccati::evolve(info, left_boundary, midpoint, yi, dyi, eps, epsh, init_step, x_eval, true, LogLevel::WARNING);
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
      = riccati::evolve(info, right_boundary, midpoint, yi, dyi, eps, epsh, init_step, x_eval, true, LogLevel::WARNING);
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
  std::cout << "left est: y and dy\n " << resp << "\n";
  std::cout << "right est: y and dy\n " << resp_right << "\n";
  auto psi_l = left_y_est.real().tail(1)[0];
  auto dpsi_l = left_dy_est.real().tail(1)[0];
  auto psi_r = right_y_est.real().tail(1)[0];
  auto dpsi_r = right_dy_est.real().tail(1)[0];
  std::cout << "Energy: " << current_energy << "\n";
  std::cout <<
    "\tpsi_l: " << psi_l <<
    "\n\tdpsi_l: " << dpsi_l <<
    "\n\tpsi_r: " << psi_r <<
    "\n\tdpsi_r: " << dpsi_r << "\n";
  auto energy_diff = std::abs((dpsi_l / psi_l) - (dpsi_r / psi_r));
  auto target_energy_diff = 360.61861864208686;
  std::cout << "\tenergy diff: " << energy_diff << "\n";
  std::cout << "ref energy: " << target_energy_diff << "\n";
  std::cout << "ref energy diff: " << std::abs(energy_diff - target_energy_diff) << "\n";
}

TEST_F(Riccati, evolve_nondense_fwd_full_optimize_schrodinger) {
  using namespace riccati;
  double l = 1.0;
  double m = 0.5;
  auto potential = [l](auto&& x_arr) {
    return (x_arr * x_arr) + l * (x_arr * x_arr * x_arr * x_arr);
  };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  //auto cout_ptr = std::unique_ptr<std::ostream, deleter_noop>(&std::cout, deleter_noop{});
  //DefaultLogger<std::ostream, deleter_noop> logger{std::move(cout_ptr)};
  constexpr auto eps = 1e-5;
  constexpr auto epsh = 1e-6;
    std::cout << std::setprecision(30);

  auto yi = std::complex(0.0, 0.0);
  auto dyi = std::complex(1e-3, 0.0);
  Eigen::Matrix<double, 0, 0> x_eval;
  auto energy_difference = [&](auto current_energy) {
    auto omega_fun
        = [current_energy, potential, m](auto&& x) {
          return eval(matrix(riccati::sqrt(2.0 * m * (std::complex(current_energy) - potential(array(x))))));
          };
    auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 16,
                                            35, 32, 32);//, std::move(logger));
    const double left_boundary = -std::pow(current_energy, 0.25) - 2.0;
    const double right_boundary = -left_boundary;
    constexpr double midpoint = 0.5;
    auto init_step = 0.01;
    auto left_res
        = riccati::evolve(info, left_boundary, midpoint, yi, dyi, eps, epsh, init_step, x_eval, true, LogLevel::WARNING);
    auto left_y_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
        std::get<1>(left_res).data(), std::get<1>(left_res).size());
    auto left_dy_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
        std::get<2>(left_res).data(), std::get<2>(left_res).size());
//    init_step_ = choose_osc_stepsize(info, right_boundary, right_boundary - midpoint, epsh);
    init_step = -init_step;
    auto right_res
        = riccati::evolve(info, right_boundary, midpoint, yi, dyi, eps, epsh, init_step, x_eval, true, LogLevel::WARNING);
    auto right_y_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
        std::get<1>(right_res).data(), std::get<1>(right_res).size());
    auto right_dy_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
        std::get<2>(right_res).data(), std::get<2>(right_res).size());
    auto psi_l = left_y_est.real().tail(1)[0];
    auto dpsi_l = left_dy_est.real().tail(1)[0];
    auto psi_r = right_y_est.real().tail(1)[0];
    auto dpsi_r = right_dy_est.real().tail(1)[0];
    std::cout << "Energy: " << current_energy << "\n";
    std::cout <<
      "\tpsi_l: " << psi_l <<
      "\n\tdpsi_l: " << dpsi_l <<
      "\n\tpsi_r: " << psi_r <<
      "\n\tdpsi_r: " << dpsi_r << "\n";
    auto energy_diff = std::abs((dpsi_l / psi_l) - (dpsi_r / psi_r));
      std::cout << "\tenergy diff: " << energy_diff << "\n";
    return energy_diff;
  };
  std::array bounds{//std::make_pair(416.5, 417.5), std::make_pair(1035.0, 1037.0),
  std::make_pair(21930.0, 21940.0)};//, std::make_pair(471100.0, 471110.0)};
  std::array reference_energy{//417.056, 1035.544,
  21932.783};//, 471103.777};
  int sentinal = 0;
  for (auto& bound : bounds) {
    boost::math::tools::eps_tolerance<double> tol(std::numeric_limits<double>::digits - 2);
    std::uintmax_t max_steps = 1500;
    using boost::math::policies::policy;
    using boost::math::policies::digits10;
    using my_pol_5 = policy<digits10<12>>;
    auto solve_ans = boost::math::tools::brent_find_minima(
      energy_difference, bound.first, bound.second, std::numeric_limits<double>::digits);
    std::cout << std::setprecision(17) << "energy at minimum = " << solve_ans.first
      << ", f(" << solve_ans.first << ") = " << solve_ans.second << " diff: " <<
     (solve_ans.first - reference_energy[sentinal]) << std::endl;
     sentinal++;
  }
}

