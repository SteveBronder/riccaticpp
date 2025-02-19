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
    ricc_ts_l, ricc_ys_l, ricc_dys_l, *rest_l = riccati.solve(info, tl, tm, 0, 1e-3, eps = 1e-5, epsh = 1e-6, hard_stop = True)
    ricc_ts_r, ricc_ys_r, ricc_dys_r, *rest_r = riccati.solve(info, tr, tm, 0, 1e-3, eps = 1e-5, epsh = 1e-6, hard_stop = True)

#    for (t, y, dy) in  zip(ricc_ts, ricc_ys, ricc_dys):
#        print(t, y, dy)
    #for step, sol, dsol in zip(sol_l["t"], sol_l["sol"], sol_l["dsol"]):
    #    print(step, sol, dsol, w(step, E))
    psi_l = ricc_ys_l[-1]
    psi_r = ricc_ys_r[-1]
    dpsi_l = ricc_dys_l[-1]
    dpsi_r = ricc_dys_r[-1]
    print(psi_l, psi_r, dpsi_l, dpsi_r)
    try:
        return abs(dpsi_l/psi_l - dpsi_r/psi_r)
    except ZeroDivisionError:
        return 1000.0

bounds = [ (416.5,417.5)]#,(1035,1037)]#,(21930,21940), (471100,471110)]
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

class RiccatiSolver:
    def __init__(self, name, init_args, solver_args):
        self.type = Algo.PYRICCATICPP
        self.name = name
        self.init_args = init_args
        self.init = ric.Init(*init_args.values())
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
        init_step = ric.choose_nonosc_stepsize(
            self.init, range[0], 1e-1, self.solver_args["epsh"]
        )
        if range[0] > range[1]:
            init_step = -init_step
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
        _, left_wavefunction, left_derivative, *unused = ric.evolve(**args)
        return left_wavefunction, left_derivative

    def __str__(self):
        return self.name




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
    solver = RiccatiSolver("ric", {
                "omega_fun": wfunc,
                "gamma_fun": gfunc,
                "nini": 8,
                "nmax": max(32, 32),
                "n": 32,
                "p": 32,
            },
            {"eps": 1e-5, "epsh": 1e-6})
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
    print(psi_l, psi_r, dpsi_l, dpsi_r)
    print(ricc_ys_l)
    print(ricc_ys_r)
    print(ricc_dys_l)
    print(ricc_dys_r)
    try:
        return abs(dpsi_l/psi_l - dpsi_r/psi_r)
    except ZeroDivisionError:
        return 1000.0

bounds = [ (416.5,417.5)]#,(1035,1037)]#,(21930,21940), (471100,471110)]
ress = []
for bound in bounds:
    res = minimize_scalar(f2,bounds=bound,method='bounded')
    print("x: ", res.x, "f(x): ", f(res.x), "Pass: ", res.success, "Msg: ", res.message)
