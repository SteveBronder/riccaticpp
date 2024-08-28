
#include <riccati/evolve.hpp>
#include <riccati/solver.hpp>
#include <riccati/vectorizer.hpp>
#include <riccati_test/utils.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <string>

TEST_F(Riccati, bremer_nondense_output) {
    using namespace riccati;

    Eigen::Matrix<double, -1, -1> bremer_table{{
      {1e1,0.2913132934408612e0,7e-14},
      {1e2,0.5294889561602804e0,5e-13},
      {1e3,-0.6028749132401260e0,3e-12},
      {1e4,-0.4813631690625038e0,5e-11},
      {1e5,0.6558931145821987e0,3e-10},
      {1e6,-0.4829009413372087e0,5e-9},
      {1e7,-0.6634949630196019e0,4e-8}}};
    std::vector<double> lambda_arr = {10, 100, 1000, 10000, 100000, 1000000, 10000000};
    double xi = -1.0;
    double xf = 1.0;
    std::vector<double> epss = {1e-12, 1e-8};
    std::vector<double> epshs = {1e-13, 1e-9};
    std::vector<int> ns = {32, 20};

    for (int j = 0; j < bremer_table.rows(); ++j) {
        double lambda_scalar = bremer_table(j, 0);
        for (size_t i = 0; i < epss.size(); ++i) {
            double eps = epss[i];
            double epsh = epshs[i];
            int n = ns[i];
            // Find the corresponding reference value for ytrue and err
            int index = j;
            double ytrue = bremer_table(index, 1);
//            double errref = bremer_table(index, 2);
            // Define omega and gamma functions
            auto omega_fun = [lambda_scalar](auto&& x) {
                return eval(matrix(lambda_scalar * sqrt(1.0 - square(array(x)) * cos(3.0 * array(x)))));
            };
            auto gamma_fun = [](auto&& x) { return zero_like(x); };
            // Initialize solver
            auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 8, std::max(32, n), n, n);
            // Choose initial step size
            auto init_step = choose_nonosc_stepsize(info, xi, 1.0, epsh);
            // Perform the evolution
            Eigen::Matrix<double, 0, 0> x_eval;
            auto res = evolve(info, xi, xf, std::complex<double>(0.0), std::complex<double>(lambda_scalar), eps, epsh, init_step, x_eval, true);
            // Get the final value of y
            auto y_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(std::get<1>(res).data(), std::get<1>(res).size()).eval();
            // Calculate error
            double yerr = std::abs((ytrue - y_est[y_est.size() - 1]) / ytrue);
            // See Fig 5 from https://arxiv.org/pdf/2212.06924
            double err_val = eps == 1e-12 ? eps * lambda_scalar : eps * lambda_scalar * 1e-4;
            EXPECT_LE(yerr, err_val) << 
            "\nLambda: " << lambda_scalar << 
            "\nn: " << n <<
            "\nepsh: " << epsh <<
            "\neps: " << eps << 
            "\nytrue: " << ytrue <<
            "\ny_est: " << y_est[y_est.size() - 1].real() <<
            "\nyerr: " << yerr << 
            "\nerr_val: " << err_val;
        }
    }
}

TEST_F(Riccati, osc_evolve_dense_output) {
  using namespace riccati;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 16,
                                           32, 32, 32);
  auto xi = 1e2;
  auto xf = 1e6;
  auto eps = 1e-12;
  auto epsh = 1e-13;
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

TEST_F(Riccati, nonosc_evolve_dense_output) {
  using namespace riccati;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 16,
                                           32, 32, 32);
  auto xi = 1e0;
  auto xf = 4e1;
  auto eps = 1e-12;
  auto epsh = 0.2;
  auto yi = riccati::test::airy_i(xi);
  auto dyi = riccati::test::airy_i_prime(xi);
  Eigen::Index Neval = 1e3;
  riccati::vector_t<double> x_eval
      = riccati::vector_t<double>::LinSpaced(Neval, xi, xf);
  auto ytrue
      = (riccati::test::airy_ai(-x_eval)
         + std::complex<double>(0.0, 1.0) * riccati::test::airy_bi(-x_eval))
            .eval();
  auto hi = 1.0 / omega_fun(xi);
  hi = choose_nonosc_stepsize(info, xi, hi, epsh);
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

TEST_F(Riccati, evolve_dense_output_airy) {
  using namespace riccati;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 8,
                                           32, 32, 32);
  auto xi = 1e0;
  auto xf = 1e6;
  auto eps = 1e-12;
  auto epsh = 1e-13;
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

TEST_F(Riccati, evolve_nondense_output_airy) {
  using namespace riccati;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 8,
                                           32, 32, 32);
  auto xi = 1e0;
  auto xf = 1e6;
  auto eps = 1e-12;
  auto epsh = 1e-13;
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

TEST_F(Riccati, evolve_dense_output_burst) {
  using namespace riccati;
  constexpr int m = 1e6;
  auto omega_fun = [m](auto&& x) {
    return eval(matrix(riccati::sqrt(static_cast<double>(std::pow(m, 2)) - 1.0)
                       / (1 + riccati::pow(array(x), 2.0))));
  };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 16,
                                           32, 32, 32);
  constexpr double xi = -m;
  constexpr double xf = m;
  auto burst_y = [m = static_cast<double>(m)](auto&& x) {
    return std::sqrt(1 + x * x) / m
           * (std::cos(m * std::atan(x))
              + std::complex<double>(0.0, 1.0) * std::sin(m * std::atan(x)));
  };
  auto yi = burst_y(xi);
  auto burst_dy = [mm = static_cast<double>(m)](auto&& x) {
    return (1.0 / std::sqrt(1.0 + x * x) / mm
            * ((x + std::complex<double>(0.0, 1.0) * mm)
                   * std::cos(mm * std::atan(x))
               + (-mm + std::complex<double>(0.0, 1.0) * x)
                     * std::sin(mm * std::atan(x))));
  };
  auto dyi = burst_dy(xi);
  auto eps = 1e-12;
  auto epsh = 1e-13;
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

TEST_F(Riccati, evolve_nondense_reverse_hardstop_airy) {
  using namespace riccati;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 16,
                                           32, 32, 32);
  auto xi = 1e6;
  auto xf = 1e0;
  auto eps = 1e-12;
  auto epsh = 1e-13;
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
                              << " y_est = " << std::get<6>(res)[i];
  }
  EXPECT_LE(y_err.maxCoeff(), 9e-6);
}

TEST_F(Riccati, evolve_nondense_fwd_hardstop_airy) {
  using namespace riccati;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 16,
                                           32, 20, 20);
  auto xi = 1e0;
  auto xf = 1e6;
  auto eps = 1e-12;
  auto epsh = 1e-13;
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

TEST_F(Riccati, evolve_nondense_fwd_hardstop_bremer) {
  using namespace riccati;
  constexpr double l = 10.0;
  auto omega_fun = [l](auto&& x) {
    using namespace ::riccati;
    return eval(matrix(l * sqrt(1.0 - square(array(x)) * cos(3.0 * array(x)))));
  };
  auto gamma_fun = [](auto&& x) { return riccati::zero_like(x); };
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 8,
                                           32, 32, 32);
  auto xi = -1.0;
  auto xf = 1.0;
  auto eps = 1e-12;
  auto epsh = 1e-13;
  std::complex<double> yi = 0.0;
  std::complex<double> dyi = l;
  Eigen::Matrix<double, 0, 0> x_eval;
  auto res
      = riccati::evolve(info, xi, xf, yi, dyi, eps, epsh, 0.1, x_eval, true);
  auto x_steps = Eigen::Map<Eigen::VectorXd>(std::get<0>(res).data(),
    std::get<0>(res).size()); 

  auto ytrue = 0.2913132934408612;
  auto y_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
      std::get<1>(res).data(), std::get<1>(res).size());
  auto y_err = std::abs((ytrue - y_est(y_est.size() - 1)) / ytrue);
  EXPECT_LE(y_err, 9e-12);
}

TEST_F(Riccati, vectorizer_evolve_nondense_fwd_hardstop_bremer) {
  using namespace riccati;
  constexpr double l = 10.0;
  auto omega_scalar = [l](auto&& x) {
    using namespace ::riccati;
    return l * std::sqrt(1.0 - x * x * std::cos(3.0 * x));
  };
  auto gamma_scalar = [](auto&& x) { return 0.0; };
  auto omega_fun = riccati::vectorize(omega_scalar);
  auto gamma_fun = riccati::vectorize(gamma_scalar);
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 8,
                                           32, 32, 32);
  auto xi = -1.0;
  auto xf = 1.0;
  auto eps = 1e-12;
  auto epsh = 1e-13;
  std::complex<double> yi = 0.0;
  std::complex<double> dyi = l;
  Eigen::Matrix<double, 0, 0> x_eval;
  auto res
      = riccati::evolve(info, xi, xf, yi, dyi, eps, epsh, 1.0, x_eval, true);
  auto x_steps = Eigen::Map<Eigen::VectorXd>(std::get<0>(res).data(),
    std::get<0>(res).size()); 
  auto ytrue = 0.2913132934408612;
  auto y_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
      std::get<1>(res).data(), std::get<1>(res).size());
  auto y_err = std::abs((ytrue - y_est(y_est.size() - 1)) / ytrue);
  EXPECT_LE(y_err, 9e-12);
}
