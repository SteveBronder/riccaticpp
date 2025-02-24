
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

TEST_F(Riccati, evolve_airy_nondense_fwd_bounds) {
  using namespace riccati;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 16,
                                           35, 32, 32);
  constexpr double xi = 0.0;
  constexpr double xf = 1e6;
  constexpr auto eps = 1e-12;
  constexpr auto epsh = 1e-13;
  auto yi = riccati::test::airy_i(xi);
  auto dyi = riccati::test::airy_i_prime(xi);
  Eigen::Matrix<double, 0, 0> x_eval;
  auto res
      = riccati::evolve(info, xi, xf, yi, dyi, eps, epsh, 1.0, x_eval, true);
  auto x_steps = Eigen::Map<Eigen::VectorXd>(std::get<0>(res).data(),
                                             std::get<0>(res).size());
  auto ytrue = riccati::test::airy_i(x_steps.array()).matrix().eval();
  auto y_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
      std::get<1>(res).data(), std::get<1>(res).size());
  auto y_err = ((y_est - ytrue).array() / ytrue.array()).abs().eval();
  for (int i = 0; i < y_err.size(); ++i) {
    EXPECT_LE(y_err[i], 9e-6) << "i = " << i << " y = " << ytrue[i]
                              << " y_est = " << std::get<6>(res)[i];
  }
  EXPECT_LE(y_err.maxCoeff(), 9e-6);
}

TEST_F(Riccati, evolve_airy_evolve_dense_fwd) {
  using namespace riccati;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 16,
                                           32, 32, 32);
  double xi = 1e2;
  constexpr double xf = 1e6;
  constexpr auto eps = 1e-12;
  constexpr auto epsh = 1e-13;
  auto yi = riccati::test::airy_i(xi);
  auto dyi = riccati::test::airy_i_prime(xi);
  Eigen::Index Neval = 1e3;
  riccati::vector_t<double> x_eval
      = riccati::vector_t<double>::LinSpaced(Neval, xi, xf);
  auto ytrue = riccati::test::airy_i(x_eval.array()).matrix().eval();
  auto hi = 2.0 * xi;
  hi = std::get<0>(choose_osc_stepsize(info, xi, hi, epsh));
  bool x_validated = false;
  while (xi < xf) {
    auto res
        = riccati::osc_evolve(info, xi, xf, yi, dyi, eps, epsh, hi, x_eval);
    if (!std::get<0>(res)) {
      break;
    } else {
      xi = std::get<1>(res);
      hi = std::get<2>(res);
      yi = std::get<1>(std::get<3>(res));
      dyi = std::get<2>(std::get<3>(res));
    }
    auto airy_true = riccati::test::airy_i(xi);
    auto airy_est = yi;
    auto err = std::abs((airy_true - airy_est) / airy_true);
    EXPECT_LE(err, 3e-7);
    auto start_y = std::get<6>(res);
    auto size_y = std::get<7>(res);
    if (size_y > 0) {
      x_validated = true;
      auto y_true_slice = ytrue.segment(start_y, size_y);
      auto y_err
          = ((std::get<4>(res) - y_true_slice).array() / y_true_slice.array())
                .abs()
                .eval();
      for (Eigen::Index i = 0; i < y_err.size(); ++i) {
        EXPECT_LE(y_err[i], 2e-6);
      }
    }
    allocator.recover_memory();
  }
  if (!x_validated) {
    FAIL() << "Dense evaluation was never completed!";
  }
}

TEST_F(Riccati, evolve_airy_nonosc_dense_output) {
  using namespace riccati;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 16,
                                           32, 32, 32);
  double xi = 1.0;
  constexpr double xf = 4e1;
  constexpr auto eps = 1e-12;
  constexpr auto epsh = 1e-13;
  auto yi = riccati::test::airy_i(xi);
  auto dyi = riccati::test::airy_i_prime(xi);
  Eigen::Index Neval = 1e3;
  riccati::vector_t<double> x_eval
      = riccati::vector_t<double>::LinSpaced(Neval, xi, xf);
  auto ytrue
      = (riccati::test::airy_ai(-x_eval)
         + std::complex<double>(0.0, 1.0) * riccati::test::airy_bi(-x_eval))
            .eval();
  auto hi = choose_nonosc_stepsize(info, xi, xf - xi, epsh);
  bool x_validated = false;
  while (xi < xf) {
    auto res
        = riccati::nonosc_evolve(info, xi, xf, yi, dyi, eps, epsh, hi, x_eval);
    if (!std::get<0>(res)) {
      break;
    } else {
      xi = std::get<1>(res);
      hi = std::get<2>(res);
      yi = std::get<1>(std::get<3>(res));
      dyi = std::get<2>(std::get<3>(res));
    }
    auto airy_true = riccati::test::airy_i(xi);
    auto airy_est = yi;
    auto err = std::abs((airy_true - airy_est) / airy_true);
    EXPECT_LE(err, 5e-9);
    auto start_y = std::get<6>(res);
    auto size_y = std::get<7>(res);
    if (size_y > 0) {
      x_validated = true;
      auto y_true_slice = ytrue.segment(start_y, size_y);
      auto&& y_est = std::get<4>(res);
      auto y_err = ((y_true_slice - y_est).array() / y_true_slice.array())
                       .abs()
                       .eval();
      for (int i = 0; i < y_err.size(); ++i) {
        EXPECT_LE(y_err[i], 4e-4) << "iter: " << i;
      }
    }
    allocator.recover_memory();
  }
  if (!x_validated) {
    FAIL() << "Dense evaluation was never completed!";
  }
}

TEST_F(Riccati, evolve_airy_dense_output) {
  using namespace riccati;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 8,
                                           32, 32, 32);
  constexpr double xi = 1e0;
  constexpr double xf = 1e6;
  constexpr auto eps = 1e-12;
  constexpr auto epsh = 1e-13;
  auto yi = riccati::test::airy_i(xi);
  auto dyi = riccati::test::airy_i_prime(xi);
  Eigen::Index Neval = 1e3;
  riccati::vector_t<double> x_eval
      = riccati::vector_t<double>::LinSpaced(Neval, xi, 1e2);
  auto ytrue = riccati::test::airy_i(x_eval.array()).matrix().eval();
  auto res = riccati::evolve(info, xi, xf, yi, dyi, eps, epsh, 0.1, x_eval);
  auto y_err
      = ((std::get<6>(res) - ytrue).array() / ytrue.array()).abs().eval();
  for (int i = 0; i < y_err.size(); ++i) {
    EXPECT_LE(y_err[i], 9e-6)
        << "i = " << i << " x = " << x_eval[i] << " y = " << ytrue[i]
        << " y_est = " << std::get<6>(res)[i];
  }
  EXPECT_LE(y_err.maxCoeff(), 9e-6);
}

TEST_F(Riccati, evolve_airy_nondense_output) {
  using namespace riccati;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 8,
                                           32, 32, 32);
  constexpr double xi = 1e0;
  constexpr double xf = 1e6;
  constexpr auto eps = 1e-12;
  constexpr auto epsh = 1e-13;
  auto yi = riccati::test::airy_i(xi);
  auto dyi = riccati::test::airy_i_prime(xi);
  // Eigen::Index Neval = 1e3;
  Eigen::Matrix<double, 0, 0> x_eval_dummy;
  auto res
      = riccati::evolve(info, xi, xf, yi, dyi, eps, epsh, 0.1, x_eval_dummy);
  auto y_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
      std::get<1>(res).data(), std::get<1>(res).size());
  Eigen::Map<Eigen::VectorXd> x_eval(std::get<0>(res).data(),
                                     std::get<0>(res).size());
  auto ytrue = riccati::test::airy_i(x_eval.array()).matrix().eval();
  auto y_err = ((y_est - ytrue).array() / ytrue.array()).abs().eval();
  for (int i = 0; i < y_err.size(); ++i) {
    EXPECT_LE(y_err[i], 9e-6) << "i = " << i << " x = " << x_eval[i]
                              << " y = " << ytrue[i] << " y_est = " << y_est[i];
  }
  EXPECT_LE(y_err.maxCoeff(), 9e-6);
}

TEST_F(Riccati, evolve_airy_nondense_reverse_hardstop) {
  using namespace riccati;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 16,
                                           32, 32, 32);
  constexpr double xi = 1e6;
  constexpr double xf = 0;
  constexpr auto eps = 1e-12;
  constexpr auto epsh = 1e-13;
  auto yi = riccati::test::airy_i(xi);
  auto dyi = riccati::test::airy_i_prime(xi);
  Eigen::Matrix<double, 0, 0> x_eval;
  auto res
      = riccati::evolve(info, xi, xf, yi, dyi, eps, epsh, -0.1, x_eval, true);
  auto x_steps = Eigen::Map<Eigen::VectorXd>(std::get<0>(res).data(),
                                             std::get<0>(res).size());
  auto ytrue = riccati::test::airy_i(x_steps.array()).matrix().eval();
  auto y_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
      std::get<1>(res).data(), std::get<1>(res).size());
  auto y_err = ((y_est - ytrue).array() / ytrue.array()).abs().eval();
  for (int i = 0; i < y_err.size(); ++i) {
    EXPECT_LE(y_err[i], 9e-6) << "i = " << i << " y = " << ytrue[i]
                              << " y_est = " << y_est[i];
  }
  EXPECT_LE(y_err.maxCoeff(), 9e-6);
}



TEST_F(Riccati, evolve_airy_nondense_fwd_hardstop) {
  using namespace riccati;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 16,
                                           32, 20, 20);
  constexpr double xi = 1e0;
  constexpr double xf = 1e6;
  constexpr auto eps = 1e-12;
  constexpr auto epsh = 1e-13;
  auto yi = riccati::test::airy_i(xi);
  auto dyi = riccati::test::airy_i_prime(xi);
  Eigen::Matrix<double, 0, 0> x_eval;
  auto res
      = riccati::evolve(info, xi, xf, yi, dyi, eps, epsh, 0.1, x_eval, true);
  auto x_steps = Eigen::Map<Eigen::VectorXd>(std::get<0>(res).data(),
                                             std::get<0>(res).size());
  auto ytrue = riccati::test::airy_i(x_steps.array()).matrix().eval();
  auto y_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
      std::get<1>(res).data(), std::get<1>(res).size());
  auto y_err = ((y_est - ytrue).array() / ytrue.array()).abs().eval();
  for (int i = 0; i < y_err.size(); ++i) {
    EXPECT_LE(y_err[i], 9e-6) << "i = " << i << " y = " << ytrue[i]
                              << " y_est = " << std::get<6>(res)[i];
  }
  EXPECT_LE(y_err.maxCoeff(), 9e-6);
}

