import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar, minimize_scalar

# ----------------------------------------
# 1) Define the Schrödinger equation
#    Example: 1D with potential V(x) = 0 (free or infinite well-like region).
#
#    We are solving: -ħ²/(2m) d²ψ/dx² + V(x)*ψ = E*ψ
#
#    For convenience, we set ħ = 1 and m = 1. Then the Schrödinger equation
#    becomes: d²ψ/dx² = (V(x) - E)*ψ
#
#    We'll recast this 2nd-order ODE as a system of two 1st-order ODEs:
#       y[0] = ψ(x),
#       y[1] = dψ/dx.
#
#    Then:
#       dψ/dx   = y[1]
#       d²ψ/dx² = (V(x) - E)*y[0]
# ----------------------------------------
def schrodinger_equation(x, y, E, potential_fn):
    """
    Returns dy/dx for the wavefunction ODE system:
        y[0] = ψ(x),
        y[1] = ψ'(x).
    """
    psi = y[0]
    dpsi = y[1]
    return [
        dpsi,
        (potential_fn(x) - E) * psi
    ]

# Simple potential: V(x) = 0
def potential_fn(x):
    return 0.0

# ----------------------------------------
# 2) Define a helper to solve from x=a to x=b given an energy E
#    and initial conditions. We'll integrate using solve_ivp.
# ----------------------------------------
def solve_wavefunction(x_span, y0, E, potential_fn, **solve_ivp_kwargs):
    """
    Solve the Schrödinger equation from x_span[0] to x_span[1]
    with initial conditions y0 and energy E.
    Returns (xs, ys) where:
        xs are the integration points
        ys[:, 0] = psi(x)
        ys[:, 1] = dpsi/dx
    """
    # We define a wrapper so that 'E' and 'potential_fn' can be passed
    # into schrodinger_equation but not into solve_ivp directly.
    def odes(x, y):
        return schrodinger_equation(x, y, E, potential_fn)

    sol = solve_ivp(
        odes,
        x_span,
        y0,
        **solve_ivp_kwargs
    )
    return sol.t, sol.y.T  # (x values, [psi, dpsi] at each x)

# ----------------------------------------
# 3) Define the mismatch function
#    We'll integrate from left to midpoint and from right to midpoint,
#    then measure mismatch of log derivatives at the midpoint.
#
#    The example below is similar to your 'energy_mismatch_functor'
#    idea, but more self-contained.
# ----------------------------------------
def wavefunction_mismatch(E):
    """
    Returns a mismatch measure for the wavefunction at x=midpoint.
    If the wavefunction from the left and from the right do not match
    in derivative/phase, mismatch is large. Our goal is to find E
    that makes mismatch ~ 0.
    """
    # For demonstration, let's define boundaries in a symmetrical way:
    left_boundary = - (E**0.25) - 2.0
    right_boundary = -left_boundary
    midpoint = 0.5

    # We'll define some small amplitude for ψ at the left boundary
    # and attempt to integrate inward.
    # For typical Schr. eq. boundary conditions in a bound state,
    # we often start with (ψ, ψ') = (small, some_value).
    #
    # The actual choice can vary a lot depending on your problem.
    # For example, for a bound state, far to the left, ψ ~ 0
    # if the potential is confining.
    # Here, we just pick some small amplitude to avoid dividing by 0.
    psi_left_init = 1e-3
    dpsi_left_init = 1e-3

    # Solve from left to midpoint
    x_left_span = (left_boundary, midpoint)
    xs_left, ys_left = solve_wavefunction(
        x_left_span,
        y0=[psi_left_init, dpsi_left_init],
        E=E,
        potential_fn=potential_fn,
        max_step=1,
        rtol=1e-8,
        atol=1e-8,
        method="DOP853"
    )
    psi_left = ys_left[-1, 0]   # ψ at midpoint
    dpsi_left = ys_left[-1, 1]  # ψ' at midpoint

    # From the right side, we do a similar approach
    # Typically, for bound states, we also want ψ -> 0 as x -> ∞ (or large).
    # We'll choose an initial wavefunction on the right boundary, then
    # integrate backward to the midpoint.
    psi_right_init = 1e-3
    dpsi_right_init = -1e-3  # negative derivative initially, for example

    x_right_span = (right_boundary, midpoint)
    xs_right, ys_right = solve_wavefunction(
        x_right_span,
        y0=[psi_right_init, dpsi_right_init],
        E=E,
        potential_fn=potential_fn,
        max_step=0.01,
        rtol=1e-8,
        atol=1e-8
    )
    psi_right = ys_right[-1, 0]
    dpsi_right = ys_right[-1, 1]

    # Compute mismatch: e.g. difference in log derivatives
    # log derivative = (dpsi/psi). We'll take the absolute difference
    try:
        mismatch = np.abs((dpsi_left / psi_left) - (dpsi_right / psi_right))
    except ZeroDivisionError:
        mismatch = 1e8  # penalize zero wavefunction at midpoint
    return mismatch

# ----------------------------------------
# 4) Use either root finding or direct minimization on the mismatch
#    function, for each bracket in the provided bounds.
# ----------------------------------------

bounds = [(416.5, 417.5), (1035, 1037), (21930, 21940), (471100, 471110)]

if __name__ == "__main__":

    print("\n=== Using direct minimization (minimize_scalar) ===")
    energies_minimized = []
    for (low, high) in bounds:
        # Use Brent's method to find the minimum of wavefunction_mismatch in [low, high].
        res = minimize_scalar(
            wavefunction_mismatch,
            bounds=(low, high),
            method='bounded'
        )
        if res.success:
            energies_minimized.append(res.x)
            print(f"Bracket {low} - {high}, minimized mismatch E = {res.x:.6f}, mismatch={res.fun:.3e}")
        else:
            print(f"Bracket {low} - {high}, minimization did not converge.")

    print("Minimized energies    = ", energies_minimized)
