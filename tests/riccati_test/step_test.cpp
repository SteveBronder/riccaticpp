#include <riccati/solver.hpp>
#include <riccati/step.hpp>
#include <riccati_test/utils.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <string>
/*
TEST_F(Riccati, step_osc_test) {
  using namespace riccati::test;
  using riccati::array;
  using riccati::eval;
  using riccati::matrix;
  using riccati::zero_like;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 16,
                                           32, 32, 32);
  auto x0 = 10.0;
  auto h = 20.0;
  auto eps = 1e-12;
  auto xscaled = (x0 + h / 2.0 + h / 2.0 * info.xn().array()).matrix().eval();
  auto omega_n = info.omega_fun_(xscaled).eval();
  auto gamma_n = info.gamma_fun_(xscaled).eval();
  auto y0 = airy_ai(-x0);
  auto dy0 = -airy_ai_prime(-x0);
  auto res
      = riccati::osc_step<false>(info, omega_n, gamma_n, x0, h, y0, dy0, eps);
  auto y_ana = airy_ai(-(x0 + h));
  auto dy_ana = -airy_ai_prime(-(x0 + h));
  auto y_err = std::abs((std::get<1>(res) - y_ana) / y_ana);
  auto dy_err = std::abs((std::get<2>(res) - dy_ana) / dy_ana);
  EXPECT_NEAR(y_err, 0, 1e-10);
  EXPECT_NEAR(dy_err, 0, 1e-10);
}

TEST_F(Riccati, step_nonosc_test) {
  using namespace riccati::test;
  using riccati::array;
  using riccati::eval;
  using riccati::matrix;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 16,
                                           32, 32, 32);
  auto xi = 1e0;
  auto h = 0.5;
  auto yi = airy_bi(-xi);
  auto dyi = -airy_bi_prime(-xi);
  auto eps = 1e-12;
  auto res = riccati::nonosc_step(info, xi, h, yi, dyi, eps);
  auto y_ana = airy_bi(-xi - h);
  auto dy_ana = -airy_bi_prime(-xi - h);
  auto y_err = std::abs((std::get<1>(res) - y_ana) / y_ana);
  auto dy_err = std::abs((std::get<2>(res) - dy_ana) / dy_ana);
  EXPECT_NEAR(y_err, 0, 1e-10);
  EXPECT_NEAR(dy_err, 0, 1e-10);
}
*/
TEST_F(Riccati, step_osc_schrodinger_test) {
  using namespace riccati::test;
  using riccati::array;
  using riccati::eval;
  using riccati::matrix;
  using riccati::zero_like;
  constexpr double l = 1.0;
  constexpr double m = 0.5;
  auto potential = [l](auto&& x_arr) {
    auto x_square = riccati::square(x_arr);
    return eval(x_square + l * riccati::square(x_square));
  };
  // These are not used
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto omega_fun = [](auto&& x) {
    return zero_like(x);
  };
  auto info = riccati::make_solver<double>(omega_fun, gamma_fun, allocator, 16,
                                          35, 35, 35);
  auto x0 = 13.1699;
  auto h = -0.694582;
  auto eps = 1e-05;
  auto xscaled = (x0 + h / 2.0 + h / 2.0 * info.xn().array()).matrix().eval();
  auto omega_n = Eigen::Matrix<std::complex<double>, -1, 1>{{
    std::complex{0.0,49.25102512814059707579872338101267814636},
std::complex{0.0,49.36094740723803653281720471568405628204},
 std::complex{0.0,49.6885946711504971062822733074426651001},
std::complex{0.0,50.22774483206748641350714024156332015991},
std::complex{0.0,50.96846131988201733520327252335846424103},
std::complex{0.0,51.89767306058865159457127447240054607391},
std::complex{0.0,52.99986518404975299745274242013692855835},
std::complex{0.0,54.25779670866025128361798124387860298157},
std::complex{0.0,55.65317497766444176932054688222706317902},
std::complex{0.0,57.16723912548682307033232063986361026764},
 std::complex{0.0,58.7812288558823112794016196858137845993},
std::complex{0.0,60.47673496158682837631204165518283843994},
std::complex{0.0,62.23594186595506982939696172252297401428},
std::complex{0.0,64.04177997081407625046267639845609664917},
std::complex{0.0,65.87800805317056074272841215133666992188},
std::complex{0.0,67.72924507967285023823933443054556846619},
std::complex{0.0,69.58096814202566804397065425291657447815},
std::complex{0.0,71.41948988649892271496355533599853515625},
std::complex{0.0,73.23192551631235858167201513424515724182},
std::complex{0.0,75.00615654811652177613723324611783027649},
std::complex{0.0,76.73079613662699216547480318695306777954},
std::complex{0.0,78.39515894634361359294416615739464759827},
std::complex{0.0,79.98923718348632405650278087705373764038},
std::complex{0.0,81.50368341982853337412961991503834724426},
std::complex{0.0,82.92980015848654318233457161113619804382},
std::complex{0.0,84.25953563639826882081251824274659156799},
std::complex{0.0,85.48548506984559480770258232951164245605},
std::complex{0.0,86.60089638241113618732924805954098701477},
std::complex{0.0,87.59967937542411675622133770957589149475},
std::complex{0.0,88.47641728485830014960811240598559379578},
std::complex{0.0,89.22637969843522398605273338034749031067},
std::complex{0.0,89.8455358701571782376049668528139591217},
std::complex{0.0,90.33056755803649195968318963423371315002},
std::complex{0.0,90.67888061825449597108672605827450752258},
std::complex{0.0,90.88861471084020138277992373332381248474},
std::complex{0.0,90.95865060472604568531096447259187698364}
}
  };
  auto gamma_n = Eigen::VectorXd::Zero(36).eval();
  auto y0 = std::complex{-2.17713e+44,0.0};
  auto dy0 = std::complex{1.99224e+46,0.0};
  auto res
      = riccati::osc_step<false>(info, omega_n, gamma_n, x0, h, y0, dy0, eps);
}
