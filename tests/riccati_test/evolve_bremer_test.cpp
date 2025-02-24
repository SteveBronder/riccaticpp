
#include <riccati/evolve.hpp>
#include <riccati/solver.hpp>
#include <riccati/vectorizer.hpp>
#include <riccati_test/utils.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <string>


TEST_F(Riccati, evolve_bremer_nondense_output) {
  using namespace riccati;

  Eigen::Matrix<double, -1, -1> bremer_table{
      {{1e1, 0.2913132934408612e0, 7e-14},
       {1e2, 0.5294889561602804e0, 5e-13},
       {1e3, -0.6028749132401260e0, 3e-12},
       {1e4, -0.4813631690625038e0, 5e-11},
       {1e5, 0.6558931145821987e0, 3e-10},
       {1e6, -0.4829009413372087e0, 5e-9},
       {1e7, -0.6634949630196019e0, 4e-8}}};
  constexpr std::array lambda_arr
      = {10, 100, 1000, 10000, 100000, 1000000, 10000000};
  constexpr double xi = -1.0;
  constexpr double xf = 1.0;
  constexpr std::array epss = {1e-12, 1e-8};
  constexpr std::array epshs = {1e-13, 1e-9};
  constexpr std::array ns = {36, 28};
  for (std::size_t j = 0; j < lambda_arr.size(); ++j) {
    double lambda_scalar = lambda_arr[j];
    for (size_t i_eps = 0; i_eps < epss.size(); ++i_eps) {
      for (size_t i_ns = 0; i_ns < ns.size(); ++i_ns) {
        double eps = epss[i_eps];
        double epsh = epshs[i_eps];
        int n = ns[i_ns];
        // Find the corresponding reference value for ytrue and err
        double ytrue = bremer_table(j, 1);
        //            double errref = bremer_table(index, 2);
        // Define omega and gamma functions
        auto omega_fun = [lambda_scalar](auto&& x) {
          return eval(
              matrix(lambda_scalar
                     * sqrt(1.0 - square(array(x)) * cos(3.0 * array(x)))));
        };
        auto gamma_fun = [](auto&& x) { return zero_like(x); };
        // Initialize solver
        auto info = riccati::make_solver<double>(
            omega_fun, gamma_fun, allocator, 8, std::max(32, n), n, n);
        // Choose initial step size
        auto init_step = choose_nonosc_stepsize(info, xi, 1.0, epsh);
        // Perform the evolution
        Eigen::Matrix<double, 0, 0> x_eval;
        auto res = evolve(info, xi, xf, std::complex<double>(0.0),
                          std::complex<double>(lambda_scalar), eps, epsh,
                          init_step, x_eval, true, riccati::LogLevel::INFO);
        // Get the final value of y
        auto y_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
                         std::get<1>(res).data(), std::get<1>(res).size())
                         .eval();
        // Calculate error
        double yerr = std::abs((ytrue - y_est[y_est.size() - 1]) / ytrue);
        // See Fig 5 from https://arxiv.org/pdf/2212.06924
        double err_val
            = eps == 1e-12 ? eps * lambda_scalar : eps * lambda_scalar * 1e-3;
        err_val = std::max(err_val, 1e-9);
        EXPECT_LE(yerr, err_val)
            << "\nLambda: " << lambda_scalar << "\nn: " << n
            << "\nepsh: " << epsh << "\neps: " << eps << "\nytrue: " << ytrue
            << "\ny_est: " << y_est[y_est.size() - 1].real()
            << "\nyerr: " << yerr << "\nerr_val: " << err_val;
      }
    }
  }
}

TEST_F(Riccati, evolve_bremer_nondense_fwd_hardstop) {
  using namespace riccati;
  constexpr double l = 10.0;
  auto omega_fun = [l](auto&& x) {
    using namespace ::riccati;
    return eval(matrix(l * sqrt(1.0 - square(array(x)) * cos(3.0 * array(x)))));
  };
  auto gamma_fun = [](auto&& x) { return riccati::zero_like(x); };
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator,
    8, 32, 32, 32, riccati::DefaultLogger<std::stringstream>{
    std::make_unique<std::stringstream>()
  });
  constexpr auto xi = -1.0;
  constexpr auto xf = 1.0;
  constexpr auto eps = 1e-12;
  constexpr auto epsh = 1e-13;
  std::complex<double> yi = 0.0;
  std::complex<double> dyi = l;
  Eigen::Matrix<double, 0, 0> x_eval;
  auto res
      = riccati::evolve(info, xi, xf, yi, dyi, eps, epsh, 0.1, x_eval, true,
      LogLevel::INFO);
  auto ytrue = 0.2913132934408612;
  auto y_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
      std::get<1>(res).data(), std::get<1>(res).size());
  auto y_err = std::abs((ytrue - y_est(y_est.size() - 1)) / ytrue);
  EXPECT_LE(y_err, 9e-11);
  //std::cout << "LOGS: \n" << info.logger().output_->str();
}

TEST_F(Riccati, evolve_bremer_vectorizer_nondense_fwd_hardstop) {
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
  constexpr auto xi = -1.0;
  constexpr auto xf = 1.0;
  constexpr auto eps = 1e-12;
  constexpr auto epsh = 1e-13;
  std::complex<double> yi = 0.0;
  std::complex<double> dyi = l;
  Eigen::Matrix<double, 0, 0> x_eval;
  auto res
      = riccati::evolve(info, xi, xf, yi, dyi, eps, epsh, 1.0, x_eval, true);
  constexpr auto ytrue = 0.2913132934408612;
  auto y_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(
      std::get<1>(res).data(), std::get<1>(res).size());
  auto y_err = std::abs((ytrue - y_est(y_est.size() - 1)) / ytrue);
  EXPECT_LE(y_err, 9e-11);
}

