
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

class Algo(Enum):
  PYRICCATICPP = 1
  DOP853 = 2
  RK45 = 3
  RK23 = 4
  RADAU = 5
  BDF = 6
  LSODA = 7
  def __str__(self):
    match self.value:
        case Algo.PYRICCATICPP:
            return "pyriccaticpp"
        case Algo.DOP853:
            return "DOP853"
        case Algo.RK45:
            return "RK45"
        case Algo.RK23:
            return "RK23"
        case Algo.RADAU:
            return "Radau"
        case Algo.BDF:
            return "BDF"
        case Algo.LSODA:
            return "LSODA"
        case _:
            return "unknown"


class Problem(Enum):
  BREMER237 = 1
  BURST = 2
  AIRY = 3
  def __str__(self):
     match self.value:
        case Problem.BREMER237:
            return "bremer237"
        case Problem.BURST:
            return "burst"
        case Problem.AIRY:
            return "airy"
        case _:
            return "unknown"

def to_string(algo : Algo):
    match algo:
        case Algo.PYRICCATICPP:
            return "pyriccaticpp"
        case Algo.DOP853:
            return "DOP853"
        case Algo.RK45:
            return "RK45"
        case Algo.RK23:
            return "RK23"
        case Algo.RADAU:
            return "Radau"
        case Algo.BDF:
            return "BDF"
        case Algo.LSODA:
            return "LSODA"
        case _:
            return "unknown"

def bremer_w_gen(*kwargs):
  l = kwargs[0]
  return lambda x: l * np.sqrt(1 - x**2 * np.cos(3 * x))

def bremer_g(*kwargs):
   return lambda x: np.zeros_like(x)

# For the reference solution
def bremer_f_gen(*kwargs):
  l = kwargs[0]
  def f(t, y):
    yp = np.zeros_like(y)
    yp[0] = y[1]
    yp[1] = -(l**2) * (1 - t**2 * np.cos(3 * t)) * y[0]
    return yp
  return f

def benchmark(method_args, ytrue, errref, N=1000):
  if method_args["method"] is Algo.PYRICCATICPP:
    info = ric.Init(*method_args["method_args"]["init_args"])
    method_args["method_args"]["init_step_args"] = (info,
                                                    method_args["method_args"]["solver_args"]["xi"],
                                                    method_args["method_args"]["solver_args"]["xf"],
                                                    method_args["method_args"]["solver_args"]["epsilon_h"])
    init_step = ric.choose_nonosc_stepsize(*method_args["method_args"]["init_step_args"])
    method_args["method_args"]["solver_args"]["init_stepsize"] = init_step
    runtime = timeit.timeit(lambda: ric.evolve(info=info, **method_args["method_args"]["solver_args"]), number=N) / N
    _, ys, _, _, _, _, _, _, _ = ric.evolve(info=info, **method_args["method_args"]["solver_args"])
    ys = np.array(ys)
    # Compute statistics
    yerr = np.abs((ytrue - ys[-1]) / ytrue)
    timing_df = pl.DataFrame({"eq_name":method_args["function_name"],
                              "method": to_string(method_args["method"]),
                              "l": l,
                              "eps": method_args["eps"],
                              "relerr": yerr,
                              "walltime": runtime,
                              "errlessref": bool(yerr < errref),
                              "params": "n={};p={};epsh={}".format(n, n, epsh)})
    return timing_df

## Bremer

# Self-reported accuracy of Bremer's phase function method on
# Eq. (237) in Bremer 2018.
# "On the numerical solution of second order ordinary
#  differential equations in the high-frequency regime".
reftable = "./benchmarks/data/eq237.csv"
try:
  refarray = pl.read_csv(reftable, separator=",")
except FileNotFoundError:
  dir_path = os.getcwd()
  print("Current Directory is ", dir_path)
  print("./data/eq237.csv was not found. This script should be run from the top level of the repository.")

def airy_w_gen(*kwargs):
  return lambda x: np.sqrt(x)

def airy_g_gen(*kwargs):
   return lambda x: np.zeros_like(x)

def airy_f_gen(*kwargs):
  return lambda t, y: [t*y[1],y[0]]


problem_dictionary = {Problem.BREMER237:
                   {"range":[-1.0, 1.0],
                    "init": complex(0.0),
                    "w": bremer_w_gen,
                    "g": bremer_g,
                    "f": bremer_f_gen,
                    "data": refarray}}

## Airy

problem_dictionary[Problem.AIRY] = {"range":[1, 1e6],
                                 "init": complex(sp.airy(-1.0)[0] + 1j * sp.airy(-1.0)[2]),
                                 "w": airy_w_gen,
                                 "g": airy_g_gen,
                                 "f": airy_f_gen,
                                 "data": None}

## Burst


## Start Benchmark
epss, epshs, ns = [1e-12, 1e-6], [1e-13, 1e-9], [35, 20]
timing_dfs = []
for key, item in problem_dictionary.items():
  for (l, ytrue, errref) in item["data"].iter_rows():
    for eps in epss:
      for epsh, n in zip(epshs, ns):
        xi = item["range"][0]
        xf = item["range"][1]
        yi = item["init"]
        method_args = {"method": Algo.PYRICCATICPP,
                      "function_name" : to_string(key),
                      "eps": eps,
                      "method_args" : {
                          "init_args" : (item["w"](l), item["g"](l), 8, max(32, n), n, n),
                          "solver_args": {"xi": xi, "xf": xf, "yi": yi,
                                          "dyi": complex(l), "eps": eps,
                                          "epsilon_h": epsh, "hard_stop": True}}}
        timing_dfs.append(benchmark(method_args, ytrue, errref, N=1000))

algo_times = pl.concat(timing_dfs, rechunk=True, how="vertical_relaxed")
algo_times


if False:
  print("Runge--Kutta")
  # We're only running this once because it's slow
  atol = 1e-14
  method_args = {"method": Algo.DOP853,
                "function_name" : "bremer237",
                "eps": eps,
                "l": l,
                "method_args" : {
                    "solver_args": {"fun" : f,
                                    "t_span": [xi, xf],
                                    # This seems odd? Shouldn't it be yi?
                                    "y0": [0, l],
                                    "method": to_string(Algo.DOP853),
                                    "rtol": eps,
                                    "atol": atol}}}
  time0 = time.time_ns()
  sol = solve_ivp(**method_args["method_args"]["solver_args"])
  time1 = time.time_ns()
  runtime = (time1 - time0) * 1e-9
  err = np.abs((sol.y[0, -1] - ytrue) / ytrue)[0]
  timing_new_df = pl.DataFrame({"method": to_string(method_args["method"]),
                                "l": l, \
                                "eps": method_args["eps"],
                                "relerr": err, \
                                "walltime": runtime,
                                "errlessref": err,
                                "params":"atol={}".format(atol)})
  timing_df = pl.concat([timing_df, timing_new_df], rechunk=True, how="vertical_relaxed")


  def Bremer237(l, n, eps, epsh, outdir, algo):
      """
      Solves problem (237) from Bremer's "On the numerical solution of second
      order ordinary differential equations in the high-frequency regime" paper.
      """

      def w(x):
          return l * np.sqrt(1 - x**2 * np.cos(3 * x))

      def g(x):
          return np.zeros_like(x)

      # For the reference solution
      def f(t, y):
          yp = np.zeros_like(y)
          yp[0] = y[1]
          yp[1] = -(l**2) * (1 - t**2 * np.cos(3 * t)) * y[0]
          return yp

      xi = -1.0
      xf = 1.0
      eps = eps
      yi = 0.0
      dyi = l
      yi_vec = np.array([yi, dyi])

      # Utility function for rounding to n significant digits
      round_to_n = lambda n, x: (
          x if x == 0 else round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))
      )

      # Reference solution and its reported error
      reftable = "./data/eq237.txt"
      refarray = np.genfromtxt(reftable, delimiter=",")
      ls = refarray[:, 0]
      ytrue = refarray[abs(ls - l) < 1e-8, 1]
      errref = refarray[abs(ls - l) < 1e-8, 2]

      if algo is Algo.PYRICCATICPP:
          print("pyriccaticpp")
          N = 1000  # Number of repetitions for timing
          epsh = epsh
          n = n
          p = n
          start = time.time_ns()
          info = ric.Init(w, g, 8, max(32, n), n, p)
          init_step = ric.choose_nonosc_stepsize(info, xi, 1.0, epsilon_h=epsh)
          for i in range(N):
              _, ys, _, _, _, _, _, _ = ric.evolve(
                  info=info,
                  xi=xi,
                  xf=xf,
                  yi=complex(yi),
                  dyi=complex(dyi),
                  eps=eps,
                  epsilon_h=epsh,
                  init_stepsize=init_step,
                  hard_stop=True,
              )
          end = time.time_ns()
          ys = np.array(ys)
          # Compute statistics
          runtime = (end - start) * 1e-9 / N
          yerr = np.abs((ytrue - ys[-1]) / ytrue)
          # Write to txt file
          # Create dir
          outputf = outdir + "bremer237-pyriccaticpp.txt"
          outputpath = Path(outputf)
          outputpath.touch(exist_ok=True)
          lines = ""
          if os.stat(outputf).st_size != 0:
              with open(outputf, "r") as f:
                  lines = f.readlines()
          with open(outputf, "w") as f:
              if lines == "":
                  f.write("method, l, eps, relerr, tsolve, errlessref, params\n")
              for line in lines:
                  f.write(line)
              f.write(
                  "{}, {}, {}, {}, {}, {}, {}".format(
                      "pyriccaticpp",
                      l,
                      eps,
                      max(yerr),
                      round_to_n(3, runtime),
                      (yerr < errref)[0],
                      "(n = {}; p = {}; epsh = {})".format(n, p, epsh),
                  )
              )
              f.write("\n")
      elif algo is Algo.RICCATICPP:
          print("riccaticpp")
          n = 32
          p = 32
          atol = 1e-14
          iters = 1000
          bash_cmd = f"./brenner237 {xi}, {xf}, {yi}, {dyi}, {eps}, {epsh} {n} {iters}"
          print(bash_cmd)
          res = subprocess.run(bash_cmd, capture_output=True, shell=True)
          runtime, yi_real, yi_imag = [float(x) for x in res.stdout.split()]
          yerr = np.abs((ytrue - complex(yi_real, yi_imag)) / ytrue)
          outputf = outdir + "bremer237-riccaticpp.txt"
          outputpath = Path(outputf)
          outputpath.touch(exist_ok=True)
          lines = ""
          if os.stat(outputf).st_size != 0:
              with open(outputf, "r") as f:
                  lines = f.readlines()
          with open(outputf, "w") as f:
              if lines == "":
                  f.write("method, l, eps, relerr, tsolve, errlessref, params\n")
              for line in lines:
                  f.write(line)
              f.write(
                  "{}, {}, {}, {}, {}, {}, {}".format(
                      "riccaticpp",
                      l,
                      eps,
                      max(yerr),
                      round_to_n(3, runtime),
                      (yerr < errref),
                      "(n = {}; p = {}; epsh = {})".format(n, p, epsh),
                  )
              )
              f.write("\n")
      elif algo is Algo.RK:
          print("Runge--Kutta")
          # We're only running this once because it's slow
          atol = 1e-14
          method = "DOP853"
          f = lambda t, y: np.array([y[1], -(l**2) * (1 - t**2 * np.cos(3 * t)) * y[0]])
          time0 = time.time_ns()
          sol = solve_ivp(f, [-1, 1], [0, l], method=method, rtol=eps, atol=atol)
          time1 = time.time_ns()
          runtime = (time1 - time0) * 1e-9
          err = np.abs((sol.y[0, -1] - ytrue) / ytrue)[0]
          # Write to txt file
          outputf = outdir + "bremer237-rk.txt"
          outputpath = Path(outputf)
          outputpath.touch(exist_ok=True)
          lines = ""
          if os.stat(outputf).st_size != 0:
              with open(outputf, "r") as f:
                  lines = f.readlines()
          with open(outputf, "w") as f:
              if lines == "":
                  f.write("method, l, eps, relerr, tsolve, errlessref, params\n")
              for line in lines:
                  f.write(line)
              f.write(
                  "{}, {}, {}, {}, {}, {}, {}".format(
                      "rk",
                      l,
                      eps,
                      round_to_n(3, err),
                      round_to_n(3, runtime),
                      (err < errref)[0],
                      "(atol = {}; method = {})".format(atol, method),
                  )
              )
              f.write("\n")


  #Let the benchmark run
  for x in range(30):
      for algo in Algo:
          for m in np.logspace(1, 7, num=7):
              print("Testing solver on Bremer 2018 Eq. (237) with lambda = {}".format(m))
              for eps, epsh, n in zip(epss, epshs, ns):
                  if m < 1e7 and algo is not Algo.RK:
                      Bremer237(m, n, eps, epsh, outdir, algo)

  # %%
