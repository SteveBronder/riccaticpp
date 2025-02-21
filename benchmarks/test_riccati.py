import numpy as np
from scipy.optimize import minimize_scalar
import riccati
import pyriccaticpp as ric
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple

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

def w(t,E):
    """
    Frequency term in the Schrodinger equation
    """
    return np.sqrt(2*m*(complex(E)-V(t)))

DEBUG = False
def f(E):
    """
    Function to minimize wrt E to give the energy eigenvalues
    """

    # Boundaries of integration
    tl = -((E)**(0.25))-2.0
    tr = -tl
    tm = 0.5

    # Grid of w, g
    t = np.linspace(tl.real,tr.real,30000)
    ws = np.log(w(t,E))
    g = np.zeros(t.shape)
    wfunc = lambda t: w(t, E)
    gfunc = lambda t: np.zeros_like(t)
    info = riccati.solversetup(wfunc, gfunc, n = 32, p = 32)
    if DEBUG:
      import pdb; pdb.set_trace()
    ricc_ts_l, ricc_ys_l, ricc_dys_l, *rest_l = riccati.solve(info, tl, tm, 0, 1e-3, eps = 1e-5, epsh = 1e-6, hard_stop = True)
    ricc_ts_r, ricc_ys_r, ricc_dys_r, *rest_r = riccati.solve(info, tr, tm, 0, 1e-3, eps = 1e-5, epsh = 1e-6, hard_stop = True)
    if DEBUG:
      ricc_ts = np.array(ricc_ts_l)
      ricc_ys = np.array(ricc_ys_l)
      ricc_dys = np.array(ricc_dys_l)
      # Stack them as columns (each list becomes a column)
      combined_matrix = np.column_stack((ricc_ts, ricc_ys, ricc_dys))
      print("left: \n", combined_matrix)
      print(combined_matrix)
      ricc_ts = np.array(ricc_ts_r)
      ricc_ys = np.array(ricc_ys_r)
      ricc_dys = np.array(ricc_dys_r)
      # Stack them as columns (each list becomes a column)
      combined_matrix = np.column_stack((ricc_ts, ricc_ys, ricc_dys))
      print("right: \n", combined_matrix)
      import pdb; pdb.set_trace()
#    for (t, y, dy) in   zip(ricc_ts, ricc_ys, ricc_dys):
#        print(t, y, dy)
    #for step, sol, dsol in zip(sol_l["t"], sol_l["sol"], sol_l["dsol"]):
    #    print(step, sol, dsol, w(step, E))
    psi_l = ricc_ys_l[-1]
    psi_r = ricc_ys_r[-1]
    dpsi_l = ricc_dys_l[-1]
    dpsi_r = ricc_dys_r[-1]
    energy_diff = abs(dpsi_l/psi_l - dpsi_r/psi_r)
    print("Iter: ")
    print("Energy: ")
    print(f"{E:.50f}")
    print("\tpsi_l: ", psi_l)
    print("\tdpsi_l: ", dpsi_l)
    print("\tpsi_r: ", psi_r)
    print("\tdpsi_r: ", dpsi_r)
    print("\tenergy_diff: ", energy_diff)
    try:
        return energy_diff
    except ZeroDivisionError:
        return 1000.0
print("Test run")
f(21940.0)
f(21936.1803400516510009765625)
print("test done")

bounds = [ #(416.5,417.5),(1035.0,1037.0),
          (21_930.0,21_940.0)]#, (471_100.0,471_110.0)]
ress = []
for bound in bounds:
    res = minimize_scalar(f,bounds=bound,method='bounded')
    print("x: ", res.x, "f(x): ", f(res.x), "Pass: ", res.success, "Msg: ", res.message)


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



def f2(E):
    """
    Function to minimize wrt E to give the energy eigenvalues
    """

    # Boundaries of integration
    tl = -((E)**(0.25))-2.0
    tr = -tl
    tm = 0.5

    # Grid of w, g
    t = np.linspace(tl.real,tr.real,30000)
    ws = np.log(w(t,E))
    g = np.zeros(t.shape)
    wfunc = lambda t: w(t, E)
    gfunc = lambda t: np.zeros_like(t)
    info = ric.Init(wfunc, gfunc, 8, 32, 32, 32)
    _, ricc_ys_l, ricc_dys_l, *unused = ric.evolve(info=info, xi=tl, xf=tm, yi=complex(1e-3), dyi=complex(1e-3), eps = 1e-5, init_stepsize = 1e-6, epsilon_h = 1e-6, hard_stop = True)
    _, ricc_ys_r, ricc_dys_r, *unused = ric.evolve(info, tr, tm, complex(0.0), complex(1e-3), eps = 1e-5, init_stepsize = -1.0, epsilon_h = 1e-6, hard_stop = True)
#    for (t, y, dy) in  zip(ricc_ts, ricc_ys, ricc_dys):
#        print(t, y, dy)
    #for step, sol, dsol in zip(sol_l["t"], sol_l["sol"], sol_l["dsol"]):
    #    print(step, sol, dsol, w(step, E))
    psi_l = ricc_ys_l[-1]
    psi_r = ricc_ys_r[-1]
    dpsi_l = ricc_dys_l[-1]
    dpsi_r = ricc_dys_r[-1]
    energy_diff = abs(dpsi_l/psi_l - dpsi_r/psi_r)
    print("Iter: ")
    print("Energy: ")
    print(f"{E:.50f}")
    print("\tpsi_l: ", psi_l)
    print("\tdpsi_l: ", dpsi_l)
    print("\tpsi_r: ", psi_r)
    print("\tdpsi_r: ", dpsi_r)
    print("\tenergy_diff: ", energy_diff)
    try:
        return energy_diff
    except ZeroDivisionError:
        return 1000.0

ress = []
for bound in bounds:
    res = minimize_scalar(f2,bounds=bound,method='bounded')
    print("x: ", res.x, "f(x): ", f(res.x), "Pass: ", res.success, "Msg: ", res.message)
