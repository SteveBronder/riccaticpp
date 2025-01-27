# %%
import numpy as np
import pyriccaticpp as ric
import scipy.special as sp
from scipy.integrate import solve_ivp
import os
from pathlib import Path
import time
import polars as pl
from enum import Enum
import timeit
import itertools
import inspect
import signal
from typing import Any, Callable, Dict, List, Tuple
from collections.abc import Iterable
import scipy.optimize as sci_opt
import matplotlib
from matplotlib import pyplot as plt


class Algo(Enum):
    """Enumeration of available algorithms for solving differential equations."""

    PYRICCATICPP = 1
    DOP853 = 2
    RK45 = 3
    RK23 = 4
    Radau = 5
    BDF = 6
    LSODA = 7

    def __str__(self) -> str:
        """Returns the name of the algorithm."""
        return str(self.name)


# WARNING: Not thread safe
class GlobalTimer:
    def __init__(self):
        self.execs = {}
        self.start_time = 0

    def start(self, name):
        if name not in self.execs:
            self.execs[name] = {"time": 0, "count": 0}
        self.start_time = time.time()

    def stop_nocall(self, name):
        end_time = time.time() - self.start_time
        self.execs[name]["time"] += end_time

    def stop(self, name):
        end_time = time.time() - self.start_time
        self.execs[name]["time"] += end_time
        self.execs[name]["count"] += 1


global_timer = GlobalTimer()


def to_string(algo: Algo, algo_iter: Dict[str, Any], problem):
    match algo:
        case Algo.PYRICCATICPP:
            return f"{algo} [eps={algo_iter[0]};epsh={algo_iter[1]};n={algo_iter[2]}][l={problem.l};m={problem.m};lb={problem.left_boundary};rb={problem.right_boundary}]"
        case _:
            return f"{algo} [rtol={algo_iter[0]}; atol={algo_iter[1]}][l={problem.l};m={problem.m};lb={problem.left_boundary};rb={problem.right_boundary}]"


class RiccatiSolver:
    def __init__(self, name, init_args, solver_args):
        self.type = Algo.PYRICCATICPP
        self.name = name
        global_timer.start(name)
        self.init_args = init_args
        self.init = ric.Init(*init_args.values())
        global_timer.stop_nocall(name)
        self.solver_args = solver_args

    def construct_args(self, problem, range, _) -> Dict[str, Any]:
        """
        Constructs the arguments required for solving the problem using pyriccaticpp.

        Args:
          problem (BaseProblem): The problem instance.
          eps (float): The epsilon parameter for the solver.
          epsh (float): The epsilon_h parameter for the solver.
          n (int): The parameter n for the solver.

        Returns:
          Dict[str, Any]: A dictionary containing the arguments for the solver.
        """
        global_timer.start(self.name)
        init_step = ric.choose_nonosc_stepsize(
            self.init, range[0], 1e-1, self.solver_args["epsh"]
        )
        global_timer.stop_nocall(self.name)
        return {
            "info": self.init,
            "xi": range[0],
            "xf": range[1],
            "yi": problem.yi_init(),
            "dyi": problem.dyi_init(),
            "eps": self.solver_args["eps"],
            "epsilon_h": self.solver_args["epsh"],
            "init_stepsize": init_step,
            "hard_stop": True,
        }

    def solve(self, args):
        global_timer.start(self.name)
        _, left_wavefunction, left_derivative, *unused = ric.evolve(**args)
        global_timer.stop(self.name)
        return left_wavefunction, left_derivative

    def __str__(self):
        return self.name


class SolveIVP:
    def __init__(self, name, method: Algo, rtol: float, atol: float):
        self.type = method
        self.rtol = rtol
        self.atol = atol
        self.name = name

    def construct_args(self, problem, range, energy: float) -> Dict[str, Any]:
        return {
            "fun": problem.f_gen(energy),
            "t_span": range,
            "y0": [problem.yi_init(), problem.dyi_init()],
            "method": str(self.type),
            "rtol": self.rtol,
            "atol": self.atol,
        }

    def solve(self, args):
        global_timer.start(self.name)
        res = solve_ivp(**args)
        global_timer.stop(self.name)
        return res.y[:, 0], res.y[:, 1]

    def __str__(self):
        return self.name


class SchrodingerProblem:
    def __init__(self, l: float, m: float, left_boundary: float, right_boundary: float):
        self.l = l
        self.m = m
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary

    def potential(self, x):
        """
        Potential function V(x) = x^2 + l*x^4
        """
        return x**2 + self.l * x**4

    def analytic_energy(n):
        """
        Optional analytic guess for the nth energy level
        """
        return np.sqrt(2.0) * (n - 0.5)

    def w_gen(self, energy):
        return lambda x: np.sqrt(2 * self.m * (complex(energy) - self.potential(x)))

    def g_gen(self):
        return lambda x: np.zeros_like(x)

    def f_gen(self, energy):
        def f(x, y):
            psi, dpsi = y
            return [dpsi, -2 * self.m * (complex(energy) - self.potential(x)) * psi]

        return f

    def yi_init(self):
        return complex(0)

    def dyi_init(self):
        return complex(1e-3)

    def solve(self, range: Tuple[float, float], energy: float, solver):
        args = solver.construct_args(self, range, energy)
        return solver.solve(args)


def construct_solver(
    algo: Algo,
    algo_iter: Dict[str, Any],
    problem: SchrodingerProblem,
    current_energy: float,
):
    name = to_string(algo, algo_iter, problem)
    if algo == Algo.PYRICCATICPP:
        return RiccatiSolver(
            name,
            {
                "omega_fun": problem.w_gen(current_energy),
                "gamma_fun": problem.g_gen(),
                "nini": 8,
                "nmax": max(32, algo_iter[2]),
                "n": algo_iter[2],
                "p": algo_iter[2],
            },
            {"eps": algo_iter[0], "epsh": algo_iter[1]},
        )
    else:
        return SolveIVP(name, algo, *algo_iter)


# %%
def energy_mismatch_functor(
    algo: Algo, algo_iter: Dict[str, Any], problem: SchrodingerProblem
):
    """
    Functor used to generate the energy mismatch function for a given algorithm and its parameters.
    """

    def f(current_energy):
        """
        Function to minimize with respect to 'current_energy' in order
        to find the correct energy eigenvalue for the bound state.

        Returns a mismatch measure between the left and right solution derivatives.
        """
        solver = construct_solver(algo, algo_iter, schrodinger, current_energy)

        left_boundary = -(current_energy**0.25) - 2.0
        right_boundary = -left_boundary
        midpoint = 0.5
        left_range, right_range = (left_boundary, midpoint), (midpoint, right_boundary)

        # Evolve (integrate) from left to midpoint
        left_wavefunction, left_derivative = problem.solve(
            left_range, current_energy, solver
        )
        right_wavefunction, right_derivative = problem.solve(
            right_range, current_energy, solver
        )
        # Final values at the midpoint
        psi_left = left_wavefunction[-1]
        psi_right = right_wavefunction[-1]
        dpsi_left = left_derivative[-1]
        dpsi_right = right_derivative[-1]

        left_log_derivative = np.log(abs(dpsi_left)) - np.log(abs(psi_left))
        right_log_derivative = np.log(abs(dpsi_right)) - np.log(abs(psi_right))
        mismatch = abs(np.exp(left_log_derivative) - np.exp(right_log_derivative))
        return mismatch

    return f


def flatten_tuple_impl(ret, x):
    if hasattr(type(x), "__iter__") and hasattr(type(x[0]), "__iter__"):
        for item in x:
            ret = flatten_tuple_impl(ret, item)
    elif hasattr(type(x), "__iter__"):
        for item in x:
            ret.append(item)
    else:
        ret.append(x)
    return ret


def flatten_tuple(x):
    ret = []
    return flatten_tuple_impl(ret, x)


# %%
epss = [1e-12, 1e-6]
epshs = [0.1 * x for x in epss]
cheby_order = [35, 20]
atol = [1e-13, 1e-7]
quantum_number = [50, 100, 1_000, 10_000]
energy_reference = [417.056, 1_035.544, 21_932.783, 471_103.666]
bounds = [(416.5, 417.5), (1_035, 1_037), (21_930, 21_940), (471_100, 471_110)]
algo_solutions = {}
algo_optim = {}
algorithm_dict = {
    Algo.PYRICCATICPP: {"args": [[epss, epshs], [cheby_order]]},
    Algo.DOP853: {"args": [[epss, atol]]},
    Algo.BDF: {"args": [[epss, atol]]},
    Algo.RK45: {"args": [[epss, atol]]},
}
algo_pl_lst: List[pl.DataFrame] = []
dir_path = os.getcwd()
if dir_path.endswith("benchmarks"):
    base_output_path = "./output/"
else:
    base_output_path = "./benchmarks/output/"
all_algo_pl_lst: List[pl.DataFrame] = []
first_write = True
with open(base_output_path + "schrodinger_times.csv", mode="a") as time_file:
    for algo, algo_params in algorithm_dict.items():
        algo_evals_pl_lst = []
        for benchmark_run in range(20):
          print("Algo: ", str(algo))
          if len(algo_params["args"]) > 1:
              algo_args_iter = itertools.product(
                  zip(*algo_params["args"][0]), *algo_params["args"][1]
              )
          else:
              algo_args_iter = zip(*algo_params["args"][0])
          print("Iter: ", benchmark_run)
          for algo_iter_ in algo_args_iter:
              algo_iter = flatten_tuple(algo_iter_)
              match algo:
                  case Algo.PYRICCATICPP:
                      args_str = (
                          f"n={algo_iter[2]};eps={algo_iter[0]};epsh={algo_iter[1]}"
                      )
                  case _:
                      args_str = f"rtol={algo_iter[0]};atol={algo_iter[1]}"
                      print("\tArgs: " + args_str)
              print("\tArgs: ", args_str)
              for energy_ref, bound in zip(energy_reference, bounds):
                  schrodinger = SchrodingerProblem(1.0, 0.5, *bound)
                  energy_mismatch = energy_mismatch_functor(algo, algo_iter, schrodinger)
                  res = sci_opt.minimize_scalar(energy_mismatch, bounds=bound, method="bounded", tol=algo_iter[0])
                  print("\t\tEigenenergy found: {}".format(res.x))
                  algo_key = to_string(algo, algo_iter, schrodinger)
                  algo_pl_tmp = pl.DataFrame(res)
                  algo_pl_tmp = algo_pl_tmp.rename({"x": "energy"})
                  algo_pl_tmp = algo_pl_tmp.with_columns(pl.lit(algo_key).alias("name"),
                                                         pl.lit(benchmark_run).alias("iter"),
                                                         pl.lit(energy_ref).alias("energy_reference"))
                  algo_pl_tmp = algo_pl_tmp.with_columns((pl.col("energy") - pl.col("energy_reference")).abs().alias("energy_error"))
                  algo_evals_pl_lst.append(algo_pl_tmp)
        algo_pl = pl.concat(algo_evals_pl_lst)
        print(algo_pl)
        algo_pl.write_csv(base_output_path + f"schrod_{str(algo)}.csv")
        all_algo_pl_lst.append(algo_pl)
        time_pl_lst = []
        for algo_key, time_st in global_timer.execs.items():
            time_pl_lst.append(
                pl.DataFrame(
                    {
                        "name": [algo_key],
                        "time": [time_st["time"]],
                        "count": [time_st["count"]],
                    }
                )
            )
        time_pl = pl.concat(time_pl_lst)
        print(time_pl)
        if first_write:
            time_pl.write_csv(time_file)
            first_write = False
        else:
            time_pl.write_csv(time_file, include_header=False)

all_algo_pl = pl.concat(all_algo_pl_lst)
all_algo_pl.write_csv(f"{base_output_path}schrod.csv")
# %%
# %%
time_pl_lst = []
for algo_key, time_st in global_timer.execs.items():
    print(algo_key)
    time_pl_lst.append(
        pl.DataFrame(
            {"name": [algo_key], "time": [time_st["time"]], "count": [time_st["count"]]}
        )
    )
time_pl = pl.concat(time_pl_lst)
time_pl.write_csv(base_output_path + "schrodinger_times.csv")


# %%
if False:
    ns = [50, 100]
    energies = solution_lst[:2]

    x_plot = np.linspace(-6, 6, 500)
    plt.figure(figsize=(10, 5))
    plt.plot(x_plot, V(x_plot), color="black", label="V(x)")

    default_init_step = 1e-12

    for j, (n, current_energy) in enumerate(zip(ns, energies)):
        # Boundaries of integration
        left_boundary = -((current_energy) ** 0.25) - 1.0
        right_boundary = -left_boundary
        midpoint = 0.0
        chebyshev_order = 32

        # Initialize Riccati solver
        riccati_info = ric.Init(
            w_gen(current_energy),
            g,
            8,
            max(32, chebyshev_order),
            chebyshev_order,
            chebyshev_order,
        )
        # Tolerances
        eps = 1e-12
        eps_h = eps * 1e-1
        # First integration range
        first_range = (left_boundary, right_boundary / 2.0)
        init_step = ric.choose_nonosc_stepsize(riccati_info, *first_range, eps_h)
        if init_step == 0:
            init_step = default_init_step
        print("iteration:", j)
        print("quantum_number:", n)
        print("left_boundary:", left_boundary)
        print("right_boundary:", right_boundary)
        print("midpoint:", midpoint)
        print("current_energy:", current_energy)
        print("init_step:", init_step)
        # Solve from left_boundary up to right_boundary/2
        full_range = (left_boundary, right_boundary)
        x_values = np.linspace(*full_range, 50_000)
        first_slice = x_values[x_values <= (right_boundary / 2.0)]
        left_solution = ric.evolve(
            riccati_info,
            *first_range,
            complex(0),
            complex(1e-8),
            eps,
            eps_h,
            init_step,
            first_slice,
            True,
        )
        left_times = left_solution[0]
        left_wavefunction = left_solution[6]
        left_step_types = left_solution[5]
        # Print debug info
        for i_val in range(len(left_solution)):
            print("i:", i_val, "\t", left_solution[i_val])
        # Find first Riccati index
        first_riccati_index = len(left_step_types) - 1
        for idx, step_type in enumerate(left_step_types):
            if step_type == 1 and 0 not in left_step_types[idx:]:
                first_riccati_index = idx
                break
        print("first_riccati_index:", first_riccati_index)
        print("range:", (left_times[first_riccati_index], midpoint))
        # Solve from right_boundary back to right_boundary/2 (or full range, whichever you need)
        init_step = ric.choose_nonosc_stepsize(riccati_info, *full_range, eps_h)
        if init_step == 0:
            init_step = default_init_step
        if full_range[0] > full_range[1]:
            init_step = -init_step
        print("init_step:", init_step)
        second_slice = x_values[x_values >= (right_boundary / 2.0)]
        right_solution = ric.evolve(
            riccati_info,
            *full_range,
            complex(0),
            complex(1e-8),
            eps,
            eps_h,
            init_step,
            second_slice,
            True,
        )
        right_wavefunction = right_solution[6]
        # Combine left and right solutions for plotting
        combined_wavefunction = np.concatenate((left_wavefunction, right_wavefunction))
        max_val = np.max(np.real(combined_wavefunction))
        scaled_wavefunction = (
            combined_wavefunction / max_val * 4.0 * np.sqrt(current_energy)
        )
        plt.plot(
            x_values,
            scaled_wavefunction + current_energy,
            color=f"C{j}",
            label=f"$\\Psi_n(x)$, n={n}, $E_n$={current_energy:.4f}",
        )

    plt.xlabel("x")
    plt.legend(loc="lower left")
    plt.show()

    # %%
