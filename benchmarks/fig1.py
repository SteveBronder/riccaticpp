
# %%
import numpy as np


import riccati
import pyriccaticpp as ric
import scipy.special as sp
from scipy.integrate import solve_ivp
import matplotlib
from matplotlib import pyplot as plt
import math
import os
from pathlib import Path
import time
import pyoscode
from matplotlib.legend_handler import HandlerTuple
import subprocess
from matplotlib.ticker import LogLocator
import polars as pl
from enum import Enum


class Algo(Enum):
    RICCATICPP = 1
    PYRICCATICPP = 2
    RK = 3
    RDC = 4
    OSCODE = 5
    WKBMARCHING = 6
    KUMMER = 7

#def run_benchmark(algo, w, g, l, n, eps, epsh):
l = 100
eps = 1e-6
epsh = 1e-6
n = 32
N = 1000
xi = -1.0
xf = 1.0
eps = eps
yi = 0.0
dyi = l
yi_vec = np.array([yi, dyi])

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

timing_df = pl.DataFrame({"method": "", "l": 123, \
                          "eps": eps, "relerr": 123.456, \
                          "walltime": 123.456,
                          "errlessref": 123.456,
                          "params": "test_str"})

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

print("pyriccaticpp")
N = 1000  # Number of repetitions for timing
epsh = epsh
n = n
p = n
start = time.time_ns()
info = ric.Init(w, g, 8, max(32, n), n, p)
init_step = ric.choose_nonosc_stepsize(info, xi, 1.0, epsilon_h=epsh)
for i in range(N):
    _, ys, _, _, _, _, _, _, _ = ric.evolve(
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
timing_new_df = pl.DataFrame({"method": ["pyriccaticpp"], "l": l, \
                          "eps": eps, "relerr": max(yerr), \
                          "walltime": runtime,
                          "errlessref": (yerr < errref)[0],
                          "params": "(n = {}; p = {}; epsh = {})".format(n, p, epsh)})
timing_df = pl.concat([timing_df, timing_new_df], rechunk=True)



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

joss_fig(outdir)

# %%
