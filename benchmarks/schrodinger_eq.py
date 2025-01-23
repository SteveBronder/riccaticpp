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
from scipy.optimize import minimize_scalar
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
        self.times = {}
        self.calls = {}
        self.start = 0
    def start(self, name):
        self.start = time.time()
        if name not in self.times:
            self.times[name] = 0
            self.calls[name] = 0
    def stop_nocall(self, name):
        self.times[name] += time.time() - self.start
    def stop(self, name):
        self.times[name] += time.time() - self.start
        self.calls[name] += 1

global_timer = GlobalTimer()

class RiccatiSolver:
    def __init__(self, init_args, solver_args):
        self.type = Algo.PYRICCATICPP
        global_timer.start(str(self.type))
        self.init_args = init_args
        self.init = ric.Init(*init_args)
        global_timer.stop_nocall(str(self.type))
        self.solver_args = solver_args

    def construct_args(self, problem, range) -> Dict[str, Any]:
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
        global_timer.start(str(self.type))
        init_step = ric.choose_nonosc_stepsize(self.init, range[0], 1e-1, self.solver_args["epsh"])
        global_timer.stop_nocall(str(self.type))
        return {
                "info" : self.init,
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
        global_timer.start(str(self.type))
        _, left_wavefunction, left_derivative, *unused = ric.evolve(**args)
        global_timer.stop(str(self.type))
        return left_wavefunction, left_derivative
    def __str__(self):
        return str(self.type) + f" [eps={self.solver_args["eps"]};epsh={self.solver_args["epsh"]};n={self.init_args["n"]}]"

class SolveIVP:
    def __init__(self, method : Algo, rtol : float, atol : float):
        self.type = method
        self.rtol = rtol
        self.atol = atol
    def construct_args(self, problem, range) -> Dict[str, Any]:
        return {
            "fun": problem.f_gen(),
            "t_span": range,
            "y0": [problem.yi_init(), problem.dyi_init()],
            "method": str(self.type),
            "rtol": self.rtol,
            "atol": self.atol,
        }
    def solve(self, args):
        global_timer.start(str(self.type))
        res = solve_ivp(**args)
        global_timer.stop(str(self.type))
        return res.y[:,0], res.y[:,1]
    def __str__(self):
        return str(self.type) + f" [rtol={self.rtol}; atol={self.atol}]"

class SchrodingerProblem:
    def __init__(self, l: float, m: float):
        self.l = l
        self.m = m

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

    def solve_riccati(self, range, solver):
        args = solver.construct_args(self, range)
        res = ric.evolve(**args["solver_args"])
        return res

    def solve_ivp(self, range, solver):
        return solve_ivp(
            **solver
        )
    def solve(self, range, solver):
        args = solver.construct_args(self, range)
        return solver.solve(args)



# %%
schrodinger = SchrodingerProblem(1.0, 0.5)
def energy_mismatch(algo, algo_iter, problem):
    """
    Function to minimize with respect to 'current_energy' in order
    to find the correct energy eigenvalue for the bound state.

    Returns a mismatch measure between the left and right solution derivatives.
    """
    def f(current_energy):
      solver = RiccatiSolver([schrodinger.w_gen(current_energy),
                            schrodinger.g_gen(), 8,
                            max(32, chebyshev_order), chebyshev_order, chebyshev_order],
                            {"eps": 1e-12, "epsh": 1e-13})
      left_boundary = -(current_energy ** 0.25) - 2.0
      right_boundary = -left_boundary
      midpoint = 0.5
      left_range, right_range = (left_boundary, midpoint), (midpoint, right_boundary)

      # Evolve (integrate) from left to midpoint
      left_wavefunction, left_derivative = problem.solve(left_range, solver)
      right_wavefunction, right_derivative = problem.solve(right_range, solver)
      # Final values at the midpoint
      psi_left = left_wavefunction[-1]
      psi_right = right_wavefunction[-1]
      dpsi_left = left_derivative[-1]
      dpsi_right = right_derivative[-1]

      left_log_derivative  = np.log(abs(dpsi_left))  - np.log(abs(psi_left))
      right_log_derivative = np.log(abs(dpsi_right)) - np.log(abs(psi_right))
      mismatch = abs(np.exp(left_log_derivative) - np.exp(right_log_derivative))
      return mismatch

def flatten_tuple_impl(ret, x):
    if hasattr(type(x), '__iter__') and hasattr(type(x[0]), '__iter__'):
      for item in x:
        ret = flatten_tuple_impl(ret, item)
    elif hasattr(type(x), '__iter__'):
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
bounds = [ (416.5,417.5),(1035,1037),(21930,21940),(471100,471110)]
solution_lst = []
optim_res = []
algorithm_dict = {
    Algo.BDF: {"args": [[epss, atol]]},
    Algo.RK45: {"args": [[epss, atol]]},
    Algo.DOP853: {"args": [[epss, atol]]},
    Algo.PYRICCATICPP: {"args": [[epss, epshs], [cheby_order]]},
}
for algo, algo_params in algorithm_dict.items():
    if len(algo_params["args"]) > 1:
      algo_args_iter = itertools.product(zip(*algo_params["args"][0]), *algo_params["args"][1])
    else:
      algo_args_iter = zip(*algo_params["args"][0])
    algo_timing_pl_lst = []
    for algo_iter_ in algo_args_iter:
      algo_iter = flatten_tuple(algo_iter_)
      energy_mismatch_algo = energy_mismatch(algo, algo_iter)
      for bound in bounds:
          print("Running for bounds: {}".format(bound))
          res = minimize_scalar(energy_mismatch,bounds=bound,method='bounded')
          print("Eigenenergy found: {}".format(res.x))
          solution_lst.append(res.x)
          optim_res.append(res)
solution_lst
# %%
ns = [50, 100]
energies = solution_lst[:2]

x_plot = np.linspace(-6, 6, 500)
plt.figure(figsize=(10, 5))
plt.plot(x_plot, V(x_plot), color='black', label='V(x)')

default_init_step = 1e-12

for j, (n, current_energy) in enumerate(zip(ns, energies)):
    # Boundaries of integration
    left_boundary = -((current_energy)**0.25) - 1.0
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
        chebyshev_order
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
        True
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
        True
    )
    right_wavefunction = right_solution[6]
    # Combine left and right solutions for plotting
    combined_wavefunction = np.concatenate((left_wavefunction, right_wavefunction))
    max_val = np.max(np.real(combined_wavefunction))
    scaled_wavefunction = combined_wavefunction / max_val * 4.0 * np.sqrt(current_energy)
    plt.plot(
        x_values,
        scaled_wavefunction + current_energy,
        color=f'C{j}',
        label=f'$\\Psi_n(x)$, n={n}, $E_n$={current_energy:.4f}'
    )

plt.xlabel('x')
plt.legend(loc='lower left')
plt.show()

# %%
