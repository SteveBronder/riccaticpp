#include <riccati/evolve.hpp>
#include <riccati/solver.hpp>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <vector>

int main(int argc, char* argv[]) {
  riccati::arena_alloc* arena{new riccati::arena_alloc{}};
  riccati::arena_allocator<double, riccati::arena_alloc> allocator{arena};
  using namespace riccati;
  auto xi = std::atof(argv[1]);
  auto xf = std::atof(argv[2]);
  std::complex<double> yi = std::atof(argv[3]);
  double l = std::atof(argv[4]);
  std::complex<double> dyi = l;
  double eps = std::atof(argv[5]);
  double epsh = std::atof(argv[6]);
  const int N = std::atoi(argv[7]);
  const int iter_amt = std::atoi(argv[8]);
  auto omega_fun
      = [l](auto&& x) {
      using namespace::riccati;
      return eval(matrix(l * sqrt(1.0 -
        square(array(x)) * cos(3.0 * array(x)))));
      };
  auto gamma_fun = [](auto&& x) { return riccati::zero_like(x); };
  auto info = riccati::make_solver<false, double>(omega_fun, gamma_fun,
    8, 32, N, N);
  Eigen::Index Neval = 1e3;
  riccati::vector_t<double> x_eval
      = riccati::vector_t<double>::LinSpaced(Neval, -0.5, 0.5);
  std::vector<double> timings;
  for (int i = 0; i < iter_amt; ++i) {
    auto start = std::chrono::system_clock::now();
    auto res = riccati::evolve(info, xi, xf, yi, dyi, eps, epsh, 0.1,
    x_eval, allocator, true);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    timings.push_back(elapsed_seconds.count());
  }
  double mean_time = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
  constexpr auto max_precision{std::numeric_limits<long double>::digits10 + 1};
  std::cout << std::setprecision(max_precision) << mean_time << " ";
  auto res = riccati::evolve(info, xi, xf, yi, dyi, eps, epsh, 0.1,
    x_eval, allocator, true);
  auto y_est = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(std::get<1>(res).data(), std::get<1>(res).size());
  auto y_back = y_est[y_est.size() - 1];
  std::cout << std::setprecision(max_precision) << y_back.real() << " " << y_back.imag() << std::endl;
}
