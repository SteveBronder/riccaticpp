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

l=1.0
m=0.5

def Ei(n):
    """
    Energy eigenvalues (if analytic solution available)
    """
    return np.sqrt(2.0)*(n-0.5)

def V(t):
    """
    Potential well
    """
    return t**2 + l*t**4

def w_gen(E):
    """
    Frequency term in the Schrodinger equation
    """
    return lambda t: np.sqrt(2*m*(complex(E)-V(t)))

g = lambda x: np.zeros_like(x)
def f(current_energy):
  """
  Function to minimize wrt E to give the energy eigenvalues
  """
  cheby_n = 32
  time_left = -((current_energy)**(0.25))-2.0
  time_right = -time_left
  time_m = 0.5
  eps = 1e-12
  epsh = eps * 1e-1
  args = {"init_args": (w_gen(current_energy), g, 8, max(32, cheby_n), cheby_n, cheby_n),
          "left_solver_args": {
              "xi": time_left,
              "xf": time_m,
              "yi": complex(0),
              "dyi": complex(1e-3),
              "eps": eps,
              "epsilon_h": epsh,
              "hard_stop": True,
          },
          "right_solver_args": {
              "xi": time_right,
              "xf": time_m,
              "yi": complex(0),
              "dyi": complex(1e-3),
              "eps": eps,
              "epsilon_h": epsh,
              "hard_stop": True,
          }}
  # Grid of w, g
  #t = np.linspace(tl.real,tr.real,30000)
  #ws = np.log(w(t,E))
  #g = np.zeros(t.shape)
  info = ric.Init(*args["init_args"])
  # Solve left side
  solver_args = args["left_solver_args"]
  args["init_step_args"] = (
      info,
      solver_args["xi"],
      solver_args["xf"],
      solver_args["epsilon_h"],
  )
  init_step = ric.choose_nonosc_stepsize(*args["init_step_args"])
  if init_step==0:
      init_step = 1e-5
  # Solve left side
  solver_args["init_stepsize"] = init_step
  _, left_ys, left_dys, _, _, _, _, _, _ = ric.evolve(info=info, **solver_args)
  solver_args = args["right_solver_args"]
  args["init_step_args"] = (
      info,
      solver_args["xi"],
      solver_args["xf"],
      solver_args["epsilon_h"],
  )
  init_step = ric.choose_nonosc_stepsize(*args["init_step_args"])
  if init_step==0:
      init_step = 1e-5
  solver_args["init_stepsize"] = -init_step
  _, right_ys, right_dys, _, _, _, _, _, _ = ric.evolve(info=info, **solver_args)
  psi_l = left_ys[-1]
  psi_r = right_ys[-1]
  dpsi_l = left_dys[-1]
  dpsi_r = right_dys[-1]
  try:
      return abs(dpsi_l/psi_l - dpsi_r/psi_r)
  except ZeroDivisionError:
      return 1000.0

# %%
bounds = [ (416.5,417.5),(1035,1037),(21930,21940),(471100,471110)]
solution_lst = []
optim_res = []
for bound in bounds:
    print("Running for bounds: {}".format(bound))
    res = minimize_scalar(f,bounds=bound,method='bounded')
    print("Eigenenergy found: {}".format(res.x))
    solution_lst.append(res.x)
    optim_res.append(res)
solution_lst
# %%
ns = [50,100]
Es = solution_lst[:2]
t_v = np.linspace(-6,6,500)
plt.figure(figsize=(10,5))
plt.plot(t_v,V(t_v),color='black',label='V(x)')
for j,n,current_energy in zip(range(len(ns)),ns,Es):
    # Boundaries of integration
    tl = -((current_energy)**(0.25))-1.0
    tr = -tl
    tm = 0.0
    cheby_n = 32
    info = ric.Init(w_gen(current_energy), g, 8, max(32, cheby_n), cheby_n, cheby_n)
    # Grid of w, g
    eps = 1e-12
    epsh = eps * 1e-1
    init_step = ric.choose_nonosc_stepsize(info, tl, tr/2.0, epsh)
    if init_step==0:
        init_step = 1e-12
    print("j: ", j)
    print("n: ", n)
    print("tl: ",tl)
    print("tr: ",tr)
    print("tm: ",tm)
    print("current_energy: ",current_energy)
    print("init_step: ",init_step)
    # Solve left side
    x_eval = np.linspace(tl, tr/2.0,30000)
    print("x_eval: ",x_eval)
    sol_l = ric.evolve(info, tl, tr/2.0, complex(0), complex(1e-3), eps, epsh, init_step, x_eval, True)
    ts_l = sol_l[0]
    y_l = sol_l[6]
    types_l = sol_l[5]
    print("ts_l: ", ts_l)
    if True:
        print("Types: ",types_l)
        [print("i: ", i, "\t", sol_l[i]) for i in range(len(sol_l))]
    firstwkb = len(types_l) - 1
    for i,typ in enumerate(types_l):
        if typ==1 and 0 not in types_l[i:]:
            firstwkb = i
            break
    print("firstwkb: ",firstwkb)
    print("range: ", (ts_l[firstwkb],tr))
    t_eval = np.linspace(ts_l[firstwkb],tr,2000)
    # Solve Right side
    init_step = ric.choose_nonosc_stepsize(info, tr/2.0, tr, epsh)
    if init_step==0:
        init_step = 1e-12
    sol_r = ric.evolve(info, tl, tr, complex(0), complex(1e-3), eps, epsh, init_step, t_eval, True)
    y_eval = sol_r[6]
    y_pre_ric = y_l[:firstwkb]
    ts_l = ts_l[:firstwkb]
    Ts_l = np.concatenate((np.array(ts_l),t_eval))
    Ys_l = np.concatenate((np.array(y_pre_ric), y_eval))
    maxx = max(np.real(Ys_l))
    Ys_l = Ys_l/maxx*4*np.sqrt(current_energy)
    plt.plot(Ts_l,Ys_l+current_energy,color='C{}'.format(j),label='$\Psi_n(x)$, n={}, $E_n$={:.4f}'.format(n,current_energy))
    plt.plot(-1*Ts_l,Ys_l+current_energy,color='C{}'.format(j))
plt.xlabel('x')
plt.legend(loc='lower left')
plt.show()

# %%
