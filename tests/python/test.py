import numpy as np
import pyriccaticpp as ric
import scipy.special as sp
import mpmath
import warnings
import pytest
import os
from scipy.optimize import minimize_scalar

# Test corresponding to the C++ "evolve_schrodinger_nondense_fwd_path_optimize"
def test_schrodinger_nondense_fwd_path_optimize():
    # Parameters (from C++ test)
    l = 1.0
    m = 0.5
    eps = 1e-5
    epsh = 1e-6
    yi = complex(0.0, 0.0)
    dyi = complex(1e-3, 0.0)

    energy_arr = np.array([
        21933.819660112502,
        21936.180339887498,
        21932.360679775,
        21932.79926820149,
        21932.92560277868,
        21932.717324526293,
        21932.79823117818,
        21932.77380316041,
        21932.752230241815,
        21932.783055066633,
        21932.784813008315,
        21932.782726417554,
        21932.783722645006,
        21932.78413912673,
        21932.78339399592,
        21932.783722645006,
    ])
    energy_target = np.array([
        360.61859818087714,
        3027.7780665966357,
        142.72696896676825,
        5.159737196477863,
        47.0246242296563,
        22.04977431633256,
        4.8158692225837285,
        3.287104498701524,
        10.448477574824096,
        0.21748306005861195,
        0.3656714725908614,
        0.3265078696388173,
        0.003973495107516101,
        0.1421312060404034,
        0.10504908512007205,
        0.003973495107516101,
    ])

    # Loop over each energy value and compare the energy difference computed via two evolutions.
    for current_energy, target_diff in zip(energy_arr, energy_target):
        # potential(x) = x^2 + l*x^4  (equivalent to using square(x) in C++)
        def potential(x):
            return x**2 + l*(x**4)

        def gamma_fun(x):
          return np.zeros_like(x)

        # omega_fun returns sqrt(2*m*(current_energy - potential(x)))
        def omega_fun(x):
            return (2.0 * m * (complex(current_energy) - potential(x)))**0.5

        # Build the solver with parameters: 16, 35, 35, 35
        info = ric.Init(omega_fun, gamma_fun, 16, 35, 35, 35)

        # Set boundaries and midpoint (as in the C++ test)
        left_boundary = - (current_energy**0.25) - 2.0
        right_boundary = -left_boundary
        midpoint = 0.5
        init_step = 0.1

        # Evolve from left_boundary to midpoint
        left_res = ric.evolve(info=info, xi=left_boundary, xf=midpoint, yi=yi, dyi=dyi, eps = eps, init_stepsize=init_step, epsilon_h = 1e-6, hard_stop = True)
        # Unpack: xs, ys, dys, ... (we only need the last y and derivative)
        xs_left, ys_left, dys_left, *_ = left_res
        psi_l = ys_left[-1]
        dpsi_l = dys_left[-1]

        # Evolve from right_boundary to midpoint (using negative step)
        right_res = ric.evolve(info=info, xi=right_boundary, xf=midpoint, yi=yi, dyi=dyi, eps=eps, init_stepsize=-init_step, epsilon_h=epsh, hard_stop=True)
        xs_right, ys_right, dys_right, *_ = right_res
        psi_r = ys_right[-1]
        dpsi_r = dys_right[-1]

        # Compute energy difference from the logarithmic derivative differences
        energy_diff = abs((dpsi_l / psi_l) - (dpsi_r / psi_r))
        # Check that the difference agrees with the target value within tolerance.
        assert abs(energy_diff - target_diff) <= 1e-4, (
            f"Energy diff {energy_diff} differs from target {target_diff} for energy {current_energy}"
        )

# Test corresponding to the C++ "evolve_schrodinger_nondense_fwd_full_optimize"
def test_schrodinger_nondense_fwd_full_optimize():
    l = 1.0
    m = 0.5
    eps = 1e-5
    epsh = 1e-6
    yi = complex(0.0, 0.0)
    dyi = complex(1e-3, 0.0)

    # This function computes the energy difference (mismatch in the logarithmic derivative)
    # for a given current_energy.
    def energy_difference(current_energy):
        def potential(x):
            return x**2 + l*(x**4)

        def gamma_fun(x):
          return np.zeros_like(x)

        # omega_fun returns sqrt(2*m*(current_energy - potential(x)))
        def omega_fun(x):
            return np.sqrt(2.0 * m * (complex(current_energy) - potential(x)))


        # Note: use slightly different solver parameters: 16, 35, 32, 32.
        info = ric.Init(omega_fun, gamma_fun, 16, 35, 32, 32)
        print("Current energy: ", current_energy)
        left_boundary = - (current_energy**0.25) - 2.0
        right_boundary = -left_boundary
        midpoint = 0.5
        # Choose a nonoscillatory stepsize between left_boundary and midpoint.
        init_step = ric.choose_osc_stepsize(info, left_boundary, midpoint - left_boundary, epsh)
        left_res = ric.evolve(info=info, xi=left_boundary, xf=midpoint, yi=yi, dyi=dyi, eps = eps, init_stepsize=init_step, epsilon_h = 1e-6, hard_stop = True)
        xs_left, ys_left, dys_left, *_ = left_res
        psi_l = ys_left[-1]
        dpsi_l = dys_left[-1]
        init_step = ric.choose_osc_stepsize(info, right_boundary, right_boundary - midpoint, epsh)
        right_res = ric.evolve(info=info, xi=right_boundary, xf=midpoint, yi=yi, dyi=dyi, eps=eps, init_stepsize=-init_step, epsilon_h=epsh, hard_stop=True)
        xs_right, ys_right, dys_right, *_ = right_res
        psi_r = ys_right[-1]
        dpsi_r = dys_right[-1]

        return abs((dpsi_l / psi_l) - (dpsi_r / psi_r))

    # Define search intervals and reference energies (from the C++ test)
    bounds = [(416.5, 417.5), (1035.0, 1037.0), (21930.0, 21939.0), (471100.0, 471110.0)]
    reference_energy = [417.056, 1035.544, 21932.783, 471103.777]

    # For each energy interval, use a Brent-type minimizer to find the energy that minimizes energy_difference.
    for (a, b), ref in zip(bounds, reference_energy):
        res = minimize_scalar(
            energy_difference,
            bounds=(a, b),
            method='bounded',
            options = {"maxiter": 1500, "xatol": epsh}
        )
        found_energy = res.x
        assert abs(found_energy - ref) <= 8e-3, (
            f"Optimized energy {found_energy} differs from reference {ref}"
        )

def test_bremer_nondense():
    cwd = os.getcwd()
    bremer_reftable = cwd + "/tests/python/data/eq237.txt"
    bremer_refarray = np.genfromtxt(bremer_reftable, delimiter=",")
    ls = bremer_refarray[:, 0]
    lambda_arr = np.logspace(1, 7, num=7)
    xi = -1.0
    xf = 1.0
    epss, epshs, ns = [1e-12, 1e-8], [1e-13, 1e-9], [35, 20]
    for lambda_scalar in lambda_arr:
        for n in ns:
            for eps, epsh in zip(epss, epshs):
                ytrue = bremer_refarray[abs(ls - lambda_scalar) < 1e-8, 1]
                errref = bremer_refarray[abs(ls - lambda_scalar) < 1e-8, 2]
                w = lambda x: lambda_scalar * np.sqrt(1.0 - x**2 * np.cos(3.0 * x))
                g = lambda x: np.zeros_like(x)
                yi = complex(0.0)
                dyi = complex(lambda_scalar)
                p = n
                info = ric.Init(w, g, 8, max(32, n), n, p)
                init_step = ric.choose_nonosc_stepsize(info, xi, 1.0, epsilon_h=epsh)
                xs, ys, dys, ss, ps, stypes, _, _, _ = ric.evolve(
                    info=info,
                    xi=xi,
                    xf=xf,
                    yi=yi,
                    dyi=dyi,
                    eps=eps,
                    epsilon_h=epsh,
                    init_stepsize=init_step,
                    hard_stop=True,
                )
                ys = np.array(ys)
                yerr = np.abs((ytrue - ys[-1]) / ytrue)
                # See Fig 5 from here https://arxiv.org/pdf/2212.06924
                if eps == 1e-12:
                    err_val = eps * lambda_scalar * 140
                else:
                    err_val = eps * lambda_scalar * 1e-3
                assert yerr < err_val


def test_denseoutput():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = ric.Init(w, g, 16, 32, 32, 32)
    xi = 1e0
    xf = 1e6
    eps = 1e-12
    epsh = 1e-13
    yi = complex(sp.airy(-xi)[0] + 1j * sp.airy(-xi)[2])
    dyi = complex(-sp.airy(-xi)[1] - 1j * sp.airy(-xi)[3])
    Neval = int(1e2)
    xeval = np.linspace(xi, xf, Neval)
    hi = 2.0 * xi
    hi = ric.choose_osc_stepsize(info, xi, hi, epsh)
    xs, ys, dys, ss, ps, stypes, yeval, dyeval,_ = ric.evolve(
        info, xi, xf, yi, dyi, eps, epsh, init_stepsize=hi, x_eval=xeval
    )
    ys_true = np.array([mpmath.airyai(-x) + 1j * mpmath.airybi(-x) for x in xs])
    dys_true = np.array(
        [
            -mpmath.airyai(-x, derivative=1) - 1j * mpmath.airybi(-x, derivative=1)
            for x in xs
        ]
    )
    ys_err = np.abs((ys_true - ys) / ys_true)
    dys_err = np.abs((dys_true - dys) / dys_true)
    assert max(ys_err) < 1e-6 and max(dys_err) < 1e-6
    ytrue = np.array([mpmath.airyai(-x) + 1j * mpmath.airybi(-x) for x in xeval])
    dytrue = np.array(
        [
            -mpmath.airyai(-x, derivative=1) - 1j * mpmath.airybi(-x, derivative=1)
            for x in xeval
        ]
    )
    yerr = np.abs((ytrue - yeval) / ytrue)
    dyerr = np.abs((dytrue - dyeval) / dytrue)
    maxerr = max(yerr)
    maxderr = max(dyerr)
    assert maxerr < 1e-6 and maxderr < 1e-6


def test_denseoutput_xbac():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = ric.Init(w, g, 16, 32, 32, 32)
    xi = 1e0
    xf = 1e6
    eps = 1e-12
    epsh = 1e-13
    yi = complex(sp.airy(-xi)[0] + 1j * sp.airy(-xi)[2])
    dyi = complex(-sp.airy(-xi)[1] - 1j * sp.airy(-xi)[3])
    Neval = int(1e2)
    xeval = np.linspace(xf, xi, Neval)
    xs, ys, dys, ss, ps, stypes, yeval, dyeval,_ = ric.evolve(
        info, xi, xf, yi, dyi, eps, epsh, init_stepsize=0.01, x_eval=xeval
    )
    ytrue = np.array([mpmath.airyai(-x) + 1j * mpmath.airybi(-x) for x in xeval])
    dytrue = np.array(
        [
            -mpmath.airyai(-x, derivative=1) - 1j * mpmath.airybi(-x, derivative=1)
            for x in xeval
        ]
    )
    yerr = np.abs((ytrue - yeval) / ytrue)
    dyerr = np.abs((dytrue - dyeval) / dytrue)
    maxerr = max(yerr)
    maxderr = max(dyerr)
    assert maxerr < 1e-6 and maxderr < 1e-6


def test_denseoutput_err():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = ric.Init(w, g, 16, 32, 32, 32)
    xi = 1e0
    xf = 1e6
    eps = 1e-12
    epsh = 1e-13
    yi = complex(sp.airy(-xi)[0] + 1j * sp.airy(-xi)[2])
    dyi = complex(-sp.airy(-xi)[1] - 1j * sp.airy(-xi)[3])
    Neval = int(1e2)
    xeval = np.linspace(xi - 10.0, xi, Neval)
    # Turn on warnings
    warnings.simplefilter("always")
    with pytest.raises(Exception):
        xs, ys, dys, ss, ps, stypes, yeval, dyeval,_ = ric.evolve(
            info, xi, xf, yi, dyi, eps, epsh, init_stepsize=0.01, x_eval=xeval
        )


def test_solve_airy():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = ric.Init(w, g, 16, 32, 32, 32)
    xi = 1e0
    xf = 1e6
    eps = 1e-12
    epsh = 1e-13
    yi = complex(sp.airy(-xi)[0] + 1j * sp.airy(-xi)[2])
    dyi = complex(-sp.airy(-xi)[1] - 1j * sp.airy(-xi)[3])
    xs, ys, dys, ss, ps, stypes, yeval, dyeval,_ = ric.evolve(
        info, xi, xf, yi, dyi, eps=eps, epsilon_h=epsh, init_stepsize=0.01
    )
    xs = np.array(xs)
    ys = np.array(ys)
    ytrue = np.array([mpmath.airyai(-x) + 1j * mpmath.airybi(-x) for x in xs])
    yerr = np.abs((ytrue - ys) / ytrue)
    maxerr = max(yerr)
    print(maxerr)
    assert maxerr < 1e-6

test_solve_airy()

def test_solve_airy_backwards():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = ric.Init(w, g, 16, 32, 32, 32)
    xi = 1e6
    xf = 1e0
    eps = 1e-12
    epsh = 1e-13
    yi = complex(sp.airy(-xi)[0] + 1j * sp.airy(-xi)[2])
    dyi = complex(-sp.airy(-xi)[1] - 1j * sp.airy(-xi)[3])
    xs, ys, dys, ss, ps, stypes, yeval, dyeval,_ = ric.evolve(
        info,
        xi,
        xf,
        yi,
        dyi,
        eps=eps,
        epsilon_h=epsh,
        init_stepsize=-0.01,
        hard_stop=True,
    )
    xs = np.array(xs)
    ys = np.array(ys)
    ytrue = np.array([mpmath.airyai(-x) + 1j * mpmath.airybi(-x) for x in xs])
    yerr = np.abs((ytrue - ys) / ytrue)
    maxerr = max(yerr)
    print(maxerr, stypes)
    assert maxerr < 1e-6


def test_denseoutput_backwards_xfor():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = ric.Init(w, g, 16, 32, 32, 32)
    xi = 1e6
    xf = 1e0
    eps = 1e-12
    epsh = 1e-13
    yi = complex(sp.airy(-xi)[0] + 1j * sp.airy(-xi)[2])
    dyi = complex(-sp.airy(-xi)[1] - 1j * sp.airy(-xi)[3])
    Neval = int(1e2)
    xeval = np.linspace(xi, xf, Neval)
    xs, ys, dys, ss, ps, stypes, yeval, dyeval,_ = ric.evolve(
        info,
        xi,
        xf,
        yi,
        dyi,
        eps=eps,
        epsilon_h=epsh,
        x_eval=xeval,
        init_stepsize=-0.01,
        hard_stop=True,
    )
    ytrue = np.array([mpmath.airyai(-x) + 1j * mpmath.airybi(-x) for x in xeval])
    yerr = np.abs((ytrue - yeval) / ytrue)
    maxerr = max(yerr)
    print(maxerr)
    assert maxerr < 1e-6


def test_denseoutput_backwards_xback():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = ric.Init(w, g, 16, 32, 32, 32)
    xi = 1e6
    xf = 1e0
    eps = 1e-12
    epsh = 1e-13
    yi = complex(sp.airy(-xi)[0] + 1j * sp.airy(-xi)[2])
    dyi = complex(-sp.airy(-xi)[1] - 1j * sp.airy(-xi)[3])
    Neval = int(1e2)
    xeval = np.flip(np.linspace(xf, xi, Neval))
    xs, ys, dys, ss, ps, stypes, yeval, dyeval,_ = ric.evolve(
        info,
        xi,
        xf,
        yi,
        dyi,
        eps=eps,
        epsilon_h=epsh,
        init_stepsize=-0.01,
        x_eval=xeval,
        hard_stop=True,
    )
    ytrue = np.array([mpmath.airyai(-x) + 1j * mpmath.airybi(-x) for x in xeval])
    dytrue = np.array(
        [
            -mpmath.airyai(-x, derivative=1) - 1j * mpmath.airybi(-x, derivative=1)
            for x in xeval
        ]
    )
    yerr = np.abs((ytrue - yeval) / ytrue)
    dyerr = np.abs((dytrue - dyeval) / dytrue)
    maxerr = max(yerr)
    maxderr = max(dyerr)
    assert maxerr < 1e-6 and maxderr < 1e-6


def test_solve_burst():
    m = float(1e6)  # Frequency parameter
    w = lambda x: np.sqrt(m**2 - 1) / (1 + x**2)
    g = lambda x: np.zeros_like(x)
    bursty = (
        lambda x: np.sqrt(1 + x**2)
        / m
        * (np.cos(m * np.arctan(x)) + 1j * np.sin(m * np.arctan(x)))
    )
    burstdy = (
        lambda x: 1
        / np.sqrt(1 + x**2)
        / m
        * (
            (x + 1j * m) * np.cos(m * np.arctan(x))
            + (-m + 1j * x) * np.sin(m * np.arctan(x))
        )
    )
    xi = -m
    xf = m
    yi = bursty(xi)
    dyi = burstdy(xi)
    eps = 1e-12
    epsh = 1e-13
    info = ric.Init(w, g, 16, 35, 32, 32)
    xs, ys, dys, ss, ps, types, yeval, dyeval,_ = ric.evolve(
        info, xi, xf, yi, dyi, eps=eps, epsilon_h=epsh
    )
    xs = np.array(xs)
    ys = np.array(ys)
    ytrue = bursty(xs)
    yerr = np.abs((ytrue - ys)) / np.abs(ytrue)
    maxerr = max(yerr)
    assert maxerr < 3e-8


def test_osc_evolve():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = ric.Init(w, g, 16, 32, 32, 32)
    xi = 1e2
    xf = 1e6
    eps = 1e-12
    epsh = 1e-13
    yi = complex(sp.airy(-xi)[0] + 1j * sp.airy(-xi)[2])
    dyi = complex(-sp.airy(-xi)[1] - 1j * sp.airy(-xi)[3])
    # Store things
    xs, ys, dys = [], [], []
    # Always necessary for setting info.y
    # Always necessary for setting info.wn, info.gn, and for getting an initial stepsize
    hi = 2 * xi
    hi = ric.choose_osc_stepsize(info, xi, hi, epsilon_h=epsh)
    # Not necessary here because info.x is already xi, but in general it might be:
    while xi < xf:
        status, x_next, h_next, osc_ret, yeval, dyeval, dense_start, dense_size = (
            ric.osc_evolve(
                info, xi, xf, yi, dyi, eps=eps, epsilon_h=epsh, init_stepsize=hi
            )
        )
        if status is False:
            break
        xs.append(x_next)
        ys.append(osc_ret[1])
        xi = x_next
        yi = osc_ret[1]
        dyi = osc_ret[2]
        hi = h_next
    xs = np.array(xs)
    ys = np.array(ys)
    ytrue = np.array([mpmath.airyai(-x) + 1j * mpmath.airybi(-x) for x in xs])
    yerr = np.abs((ytrue - ys) / ytrue)
    maxerr = max(yerr)
    print("Forward osc evolve max error:", maxerr)
    assert maxerr < 1e-4


def test_nonosc_evolve():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = ric.Init(w, g, 16, 32, 32, 32)
    xi = 1e0
    xf = 4e1
    eps = 1e-12
    epsh = 2e-1  # Note different definition of epsh for Chebyshev steps!
    yi = complex(sp.airy(-xi)[0] + 1j * sp.airy(-xi)[2])
    dyi = complex(-sp.airy(-xi)[1] - 1j * sp.airy(-xi)[3])
    # Store things
    xs, ys, dys = [], [], []
    # Always necessary for setting info.y
    # Necessary for getting an initial stepsize
    hi = 1 / w(xi)
    hi = ric.choose_nonosc_stepsize(info, xi, hi, epsilon_h=epsh)
    # Not necessary here because info.x is already xi, but in general it might be:
    while xi < xf:
        status, x_next, h_next, nonosc_ret, yeval, dyeval, dense_start, dense_size = (
            ric.nonosc_evolve(
                info, xi, xf, yi, dyi, eps=eps, epsilon_h=epsh, init_stepsize=hi
            )
        )
        if status is False:
            break
        xs.append(x_next)
        ys.append(nonosc_ret[1])
        xi = x_next
        yi = nonosc_ret[1]
        dyi = nonosc_ret[2]
        hi = h_next
    xs = np.array(xs)
    ys = np.array(ys)
    ytrue = np.array([mpmath.airyai(-x) + 1j * mpmath.airybi(-x) for x in xs])
    yerr = np.abs((ytrue - ys) / ytrue)
    maxerr = max(yerr)
    print("Forward nonosc evolve max error:", maxerr)
    assert maxerr < 1e-4


def test_osc_evolve_backwards():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = ric.Init(w, g, 16, 32, 32, 32)
    xi = 1e6
    xf = 1e2
    eps = 1e-12
    epsh = 1e-13
    yi = complex(sp.airy(-xi)[0] + 1j * sp.airy(-xi)[2])
    dyi = complex(-sp.airy(-xi)[1] - 1j * sp.airy(-xi)[3])
    # Store things
    xs, ys, dys = [], [], []
    # Always necessary for setting info.y
    # Always necessary for setting info.wn, info.gn, and for getting an initial stepsize
    hi = -xi / 10
    hi = ric.choose_osc_stepsize(info, xi, hi, epsilon_h=epsh)
    # Not necessary here because info.x is already xi, but in general it might be:
    while xi > xf:
        status, x_next, h_next, osc_ret, yeval, dyeval, dense_start, dense_size = (
            ric.osc_evolve(
                info, xi, xf, yi, dyi, eps=eps, epsilon_h=epsh, init_stepsize=hi
            )
        )
        if status is False:
            break
        xs.append(x_next)
        ys.append(osc_ret[1])
        xi = x_next
        yi = osc_ret[1]
        dyi = osc_ret[2]
        hi = h_next
    xs = np.array(xs)
    ys = np.array(ys)
    ytrue = np.array([mpmath.airyai(-x) + 1j * mpmath.airybi(-x) for x in xs])
    yerr = np.abs((ytrue - ys) / ytrue)
    maxerr = max(yerr)
    print("Backwards osc evolve max error:", maxerr)
    assert maxerr < 1.5e-7


def test_nonosc_evolve_backwards():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = ric.Init(w, g, 16, 32, 32, 32)
    xi = 4e1
    xf = 1e0
    eps = 1e-12
    epsh = 2e-1  # Note different definition of epsh for Chebyshev steps!
    yi = complex(sp.airy(-xi)[0] + 1j * sp.airy(-xi)[2])
    dyi = complex(-sp.airy(-xi)[1] - 1j * sp.airy(-xi)[3])
    # Store things
    xs, ys, dys = [], [], []
    # Necessary for getting an initial stepsize
    hi = -1 / w(xi)
    hi = ric.choose_nonosc_stepsize(info, xi, hi, epsilon_h=epsh)
    # Not necessary here because info.x is already xi, but in general it might be:
    while xi > xf:
        status, x_next, h_next, nonosc_ret, yeval, dyeval, dense_start, dense_size = (
            ric.nonosc_evolve(
                info, xi, xf, yi, dyi, eps=eps, epsilon_h=epsh, init_stepsize=hi
            )
        )
        if status != 1:
            break
        xs.append(x_next)
        ys.append(nonosc_ret[1])
        xi = x_next
        yi = nonosc_ret[1]
        dyi = nonosc_ret[2]
        hi = h_next

    xs = np.array(xs)
    ys = np.array(ys)
    ytrue = np.array([mpmath.airyai(-x) + 1j * mpmath.airybi(-x) for x in xs])
    yerr = np.abs((ytrue - ys) / ytrue)
    maxerr = max(yerr)
    print("Backward nonosc evolve max error:", maxerr)
    assert maxerr < 1e-10
