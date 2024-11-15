
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

# %%
class Algo(Enum):
  PYRICCATICPP = 1
  DOP853 = 2
  RK45 = 3
  RK23 = 4
  Radau = 5
  BDF = 6
  LSODA = 7
  def __str__(self):
    return str(self.name)

str(Algo.PYRICCATICPP)

class Problem(Enum):
  BREMER237 = 1
  BURST = 2
  AIRY = 3
  SCHRODINGER = 4
  FLAME_PROP = 5
  def __str__(self):
    return str(self.name)

def get(dict, *args):
  if len(args) == 1:
    return dict[args[0]]
  else:
    return get(dict[args[0]], *args[1:])

#%%
class BaseProblem:
  def __init__(self, start :int, end:int, y1:float, relative_error:float):
    self.range = [start, end]
    self.y1 = y1
    self.relative_error = relative_error

## Bremer

# Self-reported accuracy of Bremer's phase function method on
# Eq. (237) in Bremer 2018.
# "On the numerical solution of second order ordinary
#  differential equations in the high-frequency regime".
dir_path = os.getcwd()
if dir_path.endswith("benchmarks"):
  reftable = "./data/eq237.csv"
else:
  reftable = "./benchmarks/data/eq237.csv"
try:
  refarray = pl.read_csv(reftable, separator=",")
except FileNotFoundError:
  print("Current Directory is ", dir_path)
  print("./data/eq237.csv was not found. This script should be run from the top level of the repository.")
refarray = refarray.with_columns(pl.lit(-1).alias("start"))
refarray = refarray.with_columns(pl.lit(1).alias("end"))
#%%
def get_args(func, kwargs):
  return {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(func).args}


class Bremer(BaseProblem):
  def __init__(self, start :int, end:int, y1:float, relative_error:float, lamb:float):
    super().__init__(start, end, y1, relative_error)
    self.lamb = lamb
  def w_gen(self):
    return lambda x: self.lamb * np.sqrt(1 - x**2 * np.cos(3 * x))

  def g_gen(self):
    return lambda x: np.zeros_like(x)

  # For the reference solution
  # TODO: Replace with general Eq with F() and G()
  def f_gen(self):
    def f(t, y):
      yp = np.zeros_like(y)
      yp[0] = y[1]
      yp[1] = -(self.lamb**2) * (1 - t**2 * np.cos(3 * t)) * y[0]
      return yp
    return f

  def yi_init(self):
    return complex(0.0)

  def dyi_init(self):
    return complex(self.lamb)
  def print_params(self):
    return "start={},end={},lamb={}".format(self.range[0], self.range[1], self.lamb)

problem_dictionary = {Problem.BREMER237: {"class": Bremer, "data": refarray}}

## Airy

class Airy(BaseProblem):
  def __init__(self, start:int, end:int, y1:float, relative_error:float):
    super().__init__(start, end, y1, relative_error)

  def w_gen():
    return lambda x: np.sqrt(x)

  def g_gen(self):
    return lambda x: np.zeros_like(x)

  def f_gen(self):
    return lambda t, y: [t*y[1],y[0]]

  def yi_init(self):
     complex(sp.airy(-self.range[0])[0] + 1j * sp.airy(-self.range[0])[2])

  def dyi_init(self):
    # NOTE: Hard coded init position as 1 here :(
    return complex(-sp.airy(-self.range[0])[1] - 1j * sp.airy(-self.range[0])[3])
  def print_params(self):
    return "start={},end={}".format(self.range[0], self.range[1])


airy_data = pl.DataFrame({"start": [-1.0], "end": [1.0], "y1": [0.0], "relative_error": [1e-12]})
problem_dictionary[Problem.AIRY] = {"class" : Airy, "data": airy_data}

## Burst

def gen_problem(problem, data:pl.DataFrame):
   for x in data.iter_rows(named=True):
      yield problem(**x)

##

# %%
def construct_riccati_args(problem, eps : float, epsh :float, n :int):
  return { "print_args":{"n": n, "p": n, "epsh": epsh},
           "init_args" : (problem.w_gen(),
                         problem.g_gen(),
                         8, max(32, n), n, n),
          "solver_args": {"xi": problem.range[0],
                          "xf": problem.range[1],
                          "yi": problem.yi_init(),
                          "dyi": problem.dyi_init(), "eps": eps,
                          "epsilon_h": epsh, "hard_stop": True}}

def construct_solve_ivp_args(problem, method : str, rtol : float, atol :float):
  return {"fun": problem.f_gen(),
          "t_span": problem.range,
          "y0": [problem.yi_init(), problem.dyi_init()],
          "method": str(method),
          "rtol": rtol,
          "atol": atol}

def construct_algo_args(algo:Algo, problem, args:dict):
  match algo:
    case Algo.PYRICCATICPP:
      # Here it is eps, epsh, n as first 3 of args
      return construct_riccati_args(problem, *args)
    case _:
      return construct_solve_ivp_args(problem, str(algo), *args)

## Start Benchmark
epss, epshs, ns = [1e-12, 1e-6], [1e-13, 1e-9], [35, 20]
atol = [1e-11, 1e-6]
rtol = [1e-14, 1e-12]
algorithm_dict = {Algo.DOP853: [epss, atol],
                  Algo.PYRICCATICPP: [epss, epshs, ns],

                  Algo.RK45: [epss, atol],
                  Algo.Radau: [epss, atol],
                  Algo.BDF: [epss, atol],
                  Algo.LSODA: [epss, atol]}

def benchmark(problem_info, algo_args, N=1000):
  match algo_args["method"]:
    case Algo.PYRICCATICPP:
      info = ric.Init(*algo_args["method_args"]["init_args"])
      solver_args = algo_args["method_args"]["solver_args"]
      algo_args["method_args"]["init_step_args"] = (info,
                                                      solver_args["xi"],
                                                      solver_args["xf"],
                                                      solver_args["epsilon_h"])
      init_step = ric.choose_nonosc_stepsize(*algo_args["method_args"]["init_step_args"])
      solver_args["init_stepsize"] = init_step
      runtime = timeit.timeit(lambda: ric.evolve(info=info, **solver_args), number=N) / N
      _, ys, _, _, _, _, _, _, _ = ric.evolve(info=info, **solver_args)
      ys = np.array(ys)
      # Compute statistics
      yerr = np.abs((problem_info.y1 - ys[-1]) / problem_info.y1)
      print_args = algo_args["method_args"]["print_args"]
      timing_df = pl.DataFrame({"eq_name":algo_args["function_name"],
                                "method": str(algo_args["method"]),
                                "eps": algo_args["eps"],
                                "relerr": yerr,
                                "walltime": runtime,
                                "errlessref": bool(yerr < problem_info.relative_error),
                                "problem_params": problem_info.print_params(),
                                "params": "n={};p={};epsh={}".format(print_args["n"],
                                                                    print_args["n"],
                                                                    print_args["epsh"])})
      return timing_df
    # All Python IVPs use the same scheme
    case _:
      runtime = timeit.timeit(lambda: solve_ivp(**algo_args["method_args"]), number=N) / N
      sol = solve_ivp(**algo_args["method_args"])
      yerr = np.abs((sol.y[0, -1] - problem_info.y1) / problem_info.y1)
      timing_df = pl.DataFrame({"eq_name":algo_args["function_name"],
                                "method": str(algo_args["method"]),
                                "eps": algo_args["eps"],
                                "relerr": yerr,
                                "walltime": runtime,
                                "errlessref": bool(yerr < problem_info.relative_error),
                                "problem_params": problem_info.print_params(),
                                "params": "rtol={};atol={}".format(algo_args["method_args"]["rtol"],
                                                                   algo_args["method_args"]["atol"])})
      return timing_df

timing_dfs = []
for problem_key, problem_item in problem_dictionary.items():
    for algo, algo_params in algorithm_dict.items():
      for algo_iter in list(itertools.product(*algo_params)):
        for problem_info in gen_problem(problem_item["class"], problem_item["data"]):
          print("Running ", str(algo), " on ", str(problem_key))
          algo_args = {"method": algo,
                        "function_name" : str(problem_key),
                        # Eps must always be first arg of tuple
                        "eps": algo_iter[0],
                        "method_args" :
                          construct_algo_args(algo, problem_info, algo_iter)}
          timing_dfs.append(benchmark(problem_info, algo_args, N=1000))

algo_times = pl.concat(timing_dfs, rechunk=True, how="vertical_relaxed")
algo_times

# %%
