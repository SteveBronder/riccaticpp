
#include <riccati/chebyshev.hpp>
#include <riccati/solver.hpp>
#include <riccati_test/utils.hpp>
#include <riccati_test/chebyshev_output.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <string>

TEST_F(Riccati, cheby_chebyshev_err) {
  double a = 3.0;
  auto f
      = [a](auto&& x) { return (a * x.array() + 1.0).sin().matrix().eval(); };
  auto df = [a](auto&& x) {
    return (a * (a * x.array() + 1.0).cos()).matrix().eval();
  };
  for (auto i : std::vector{16, 32, 35}) {
    auto [D, x] = riccati::chebyshev<double>(i);
    auto maxerr = (D * f(x) - df(x)).array().abs().maxCoeff();
    EXPECT_LT(maxerr, 1e-8);
  }
}

TEST_F(Riccati, cheby_quad_wts2) {
  auto f = [](auto&& x) { return std::sin(3.0 * x + 1.0); };
  auto df = [](auto&& x) {
    return (3.0 * (3.0 * x.array() + 1.0).cos()).matrix().eval();
  };
  for (auto val : std::vector<int>{8, 16, 17, 22, 32, 35, 64}) {
    auto chebyshev_pair = riccati::chebyshev<double>(val);
    auto dfs = df(chebyshev_pair.second);
    auto weights = riccati::quad_weights<double>(val);
    auto f1 = f(1) - f(-1);
    auto max_error = weights.dot(dfs) - f1;
    if (val == 8) {
      EXPECT_NEAR(max_error, 0, 1.5e-6) << "n: " << val;
    } else {
      EXPECT_NEAR(max_error, 0, 1e-8) << "n: " << val;
    }
  }
}

TEST_F(Riccati, cheby_quad_wts) {
  Eigen::VectorXd truth{
      {0.0008163265306122449, 0.007855617388824607, 0.016093648780921472,
       0.02384692501092013,   0.031558228574027666, 0.03893311572526589,
       0.046046156121968086,  0.052753386928556995, 0.059060993255814126,
       0.06487444199575146,   0.07017971496057955,  0.07490903400617871,
       0.07904368405938869,   0.08253539095828089,  0.08536753716441627,
       0.08750870504899079,   0.0889478045795044,   0.0896692889099981,
       0.08966928890999809,   0.0889478045795044,   0.0875087050489908,
       0.08536753716441628,   0.08253539095828089,  0.0790436840593887,
       0.0749090340061787,    0.07017971496057952,  0.06487444199575146,
       0.05906099325581414,   0.052753386928557,    0.04604615612196809,
       0.03893311572526589,   0.03155822857402768,  0.023846925010920138,
       0.01609364878092149,   0.007855617388824612, 0.0008163265306122449}};
  auto weights = riccati::quad_weights<double>(35);
  for (Eigen::Index i = 0; i < weights.size(); ++i) {
    EXPECT_NEAR(weights(i), truth(i), 1e-8)
        << " i: " << i << "\n weights(i): " << weights(i)
        << "\n truth(i):   " << truth(i)
        << "\n diff:       " << weights(i) - truth(i);
  }
}

TEST_F(Riccati, cheby_chebyshev_coeffs_to_cheby_nodes_truth) {
  Eigen::Map<Eigen::Matrix<double, 16, 16>> truth(
      riccati::test::output::chebyshev_coeffs_to_cheby_nodes_truth.data());
  Eigen::Matrix<double, 16, 16> inp = Eigen::Matrix<double, 16, 16>::Identity();
  Eigen::MatrixXd result = riccati::coeffs_to_cheby_nodes(inp);
  EXPECT_EQ(result.rows(), truth.rows());
  EXPECT_EQ(result.cols(), truth.cols());
  for (Eigen::Index i = 0; i < truth.size(); ++i) {
    EXPECT_FLOAT_EQ(truth(i), result(i));
  }
}

TEST_F(Riccati, cheby_chebyshev_cheby_nodes_to_coeffs_truth) {
  Eigen::Map<Eigen::Matrix<double, 16, 16>> truth(
      riccati::test::output::chebyshev_cheby_nodes_to_coeffs_truth.data());
  Eigen::Matrix<double, 16, 16> inp
      = Eigen::Matrix<double, 16, 16>::Identity(16, 16);
  Eigen::MatrixXd result = riccati::cheby_nodes_to_coeffs(inp);
  EXPECT_EQ(result.rows(), truth.rows());
  EXPECT_EQ(result.cols(), truth.cols());
  for (Eigen::Index j = 0; j < truth.cols(); ++j) {
    for (Eigen::Index i = 0; i < truth.rows(); ++i) {
      EXPECT_FLOAT_EQ(truth(i, j), result(i, j));
    }
  }
}

TEST_F(Riccati, cheby_coeffs_and_cheby_nodes) {
  Eigen::Matrix<double, 16, 16> inp
      = Eigen::Matrix<double, 16, 16>::Identity(16, 16);
  auto result = riccati::coeffs_and_cheby_nodes(inp);
  Eigen::Map<Eigen::Matrix<double, 16, 16>> cheby_nodes_truth(
      riccati::test::output::chebyshev_coeffs_to_cheby_nodes_truth.data());
  Eigen::Map<Eigen::Matrix<double, 16, 16>> coeffs_truth(
      riccati::test::output::chebyshev_cheby_nodes_to_coeffs_truth.data());
  auto&& cheby_node_res = result.first;
  auto&& coeff_res = result.second;
  EXPECT_EQ(cheby_node_res.rows(), cheby_nodes_truth.rows());
  EXPECT_EQ(cheby_node_res.cols(), cheby_nodes_truth.cols());
  EXPECT_EQ(coeff_res.rows(), coeffs_truth.rows());
  EXPECT_EQ(coeff_res.cols(), coeffs_truth.cols());
  for (Eigen::Index i = 0; i < cheby_node_res.rows(); ++i) {
    for (Eigen::Index j = 0; j < cheby_node_res.cols(); ++j) {
      EXPECT_FLOAT_EQ(cheby_nodes_truth(i, j), cheby_node_res(i, j))
          << "cheby_node(" << i << ", " << j << ")";
    }
  }
  for (Eigen::Index i = 0; i < coeff_res.rows(); ++i) {
    for (Eigen::Index j = 0; j < coeff_res.cols(); ++j) {
      EXPECT_FLOAT_EQ(coeffs_truth(i, j), coeff_res(i, j))
          << "coeffs(" << i << ", " << j << ")";
    }
  }
}

TEST_F(Riccati, cheby_chebyshev_integration_truth) {
  constexpr Eigen::Index n = 16;
  Eigen::Map<Eigen::Matrix<double, 16, 16>> truth(
      riccati::test::output::chebyshev_integration_truth.data());
  auto Im = riccati::integration_matrix<double>(n);
  for (Eigen::Index j = 0; j < truth.cols(); ++j) {
    for (Eigen::Index i = 0; i < truth.rows(); ++i) {
      EXPECT_NEAR(truth(i, j), Im(i, j), 1e-8)
          << "for index: (" << i << ", " << j << ")";
    }
  }
}

TEST_F(Riccati, cheby_quad_weights_test) {
  constexpr Eigen::Index n = 32;
  Eigen::Map<Eigen::Matrix<double, 33, 1>> truth(
      riccati::test::output::quad_weights_truth.data());
  auto&& weights = riccati::quad_weights<double>(n);
  EXPECT_EQ(weights.size(), truth.size());
  for (Eigen::Index i = 0; i < weights.size(); ++i) {
    EXPECT_NEAR(weights(i), truth(i), 1e-8);
  }
}

TEST_F(Riccati, cheby_chebyshev_chebyshev_truth) {
  constexpr Eigen::Index n = 32;
  auto chebyshev_pair = riccati::chebyshev<double>(n);
  Eigen::Map<Eigen::Matrix<double, 33, 1>> x_truth(
      riccati::test::output::chebyshev_chebyshev_truth.data());
  auto&& x = chebyshev_pair.second;
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x(i), x_truth(i));
  }
  Eigen::Map<Eigen::Matrix<double, 33, 33>> D_truth(
      riccati::test::output::D_real.data());
  D_truth.transposeInPlace();
  auto&& D = chebyshev_pair.first;
  for (Eigen::Index j = 0; j < D_truth.cols(); ++j) {
    for (Eigen::Index i = 0; i < D_truth.rows(); ++i) {
      EXPECT_NEAR(D_truth(i, j), D(i, j), 1e-10)
          << "for index: (" << i << ", " << j << ")";
    }
  }
}

TEST_F(Riccati, cheby_chebyshev_integration) {
  constexpr Eigen::Index n = 32;
  constexpr double a = 3.0;
  auto f = [a](auto x) {
    return riccati::sin(a * x.array() + 1.0).matrix().eval();
  };
  auto df = [a](auto x) {
    return (a * riccati::cos(a * x.array() + 1.0)).matrix().eval();
  };
  auto chebyshev_pair = riccati::chebyshev<double>(n);
  auto dfs = df(chebyshev_pair.second);
  auto fs = f(chebyshev_pair.second);
  fs.array() -= fs.coeff(fs.size() - 1);
  auto Im = riccati::integration_matrix<double>(n + 1);
  auto fs_est = Im * dfs;
  auto maxerr = ((fs_est - fs).array() / fs.array()).abs().maxCoeff();
  EXPECT_NEAR(maxerr, 0.0, 1e-13);
}

TEST_F(Riccati, cheby_interpolate_test) {
  riccati::vector_t<double> x_scaled(33);
  x_scaled << 1.4880213, 1.4868464, 1.4833327, 1.4775143, 1.4694471, 1.4592089,
      1.4468981, 1.4326335, 1.4165523, 1.3988094, 1.3795757, 1.3590365,
      1.3373895, 1.3148432, 1.2916148, 1.2679279, 1.2440107, 1.2200934,
      1.1964066, 1.1731781, 1.1506318, 1.1289848, 1.1084456, 1.0892119,
      1.0714691, 1.0553879, 1.0411232, 1.0288125, 1.0185742, 1.010507,
      1.0046886, 1.001175, 1;
  riccati::vector_t<double> x_dense(5);
  x_dense << 1., 1.0990991, 1.1981982, 1.2972973, 1.3963964;
  auto ans = riccati::interpolate(x_scaled, x_dense, allocator);
  const auto r = x_scaled.size();
  const auto q = x_dense.size();
  auto V = riccati::matrix_t<double>::Ones(r, r).eval();
  auto R = riccati::matrix_t<double>::Ones(q, r).eval();
  for (std::size_t i = 1; i < static_cast<std::size_t>(r); ++i) {
    V.col(i).array() = V.col(i - 1).array() * x_scaled.array();
    R.col(i).array() = R.col(i - 1).array() * x_dense.array();
  }
  Eigen::MatrixXd LL = (V.transpose() * ans.transpose()).transpose();
  Eigen::MatrixXd err = ((R - LL).array().abs() / R.array()).eval();
  for (Eigen::Index j = 0; j < err.cols(); ++j) {
    for (Eigen::Index i = 0; i < err.rows(); ++i) {
      EXPECT_NEAR(err(i, j), 0.0, 1e-9)
          << "for index: (" << i << ", " << j << ")";
    }
  }
}

TEST_F(Riccati, cheby_spectral_chebyshev_test) {
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
  constexpr auto xi = 1.0;
  const auto h = 0.4880213350286135;
  const auto y0 = std::complex<double>(0.5355608832923522, 0.10399738949694468);
  const auto dy0
      = std::complex<double>(0.010160567116645175, -0.5923756264227923);
  constexpr auto niter = 0;
  auto ret = riccati::spectral_chebyshev(info, xi, h, y0, dy0, niter);
  auto&& spec_y1 = riccati::test::output::spectral_cheby_y1;
  auto&& spec_dy1 = riccati::test::output::spectral_cheby_dy1;
  for (Eigen::Index i = 0; i < spec_y1.size(); ++i) {
    EXPECT_NEAR(std::get<0>(ret)(i).real(), spec_y1(i).real(), 1e-12);
    EXPECT_NEAR(std::get<0>(ret)(i).imag(), spec_y1(i).imag(), 1e-12);
  }
  for (Eigen::Index i = 0; i < spec_y1.size(); ++i) {
    EXPECT_NEAR(std::get<1>(ret)(i).real(), spec_dy1(i).real(), 1e-12);
    EXPECT_NEAR(std::get<1>(ret)(i).imag(), spec_dy1(i).imag(), 1e-12);
  }
}
