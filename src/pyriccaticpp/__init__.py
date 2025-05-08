from __future__ import annotations

from .pyriccaticpp import (
    evolve,
    choose_osc_stepsize,
    choose_nonosc_stepsize,
    osc_evolve,
    nonosc_evolve,
    Init_OF64_GF64,
    Init_OC64_GF64,
    Init_OF64_GC64,
    Init_OC64_GC64)
__all__ = ["evolve", "choose_osc_stepsize", "choose_nonosc_stepsize", "osc_evolve", "nonosc_evolve"]


class Init:
    def __new__(cls, omega_fun, gamma_fun, nini, nmax, n, p):
        """
        Factory class that chooses the appropriate SolverInfo instantiation
        based on the return types of omega_fun and gamma_fun.
        """
        # Evaluate the callables on a sample input (0 in this example).
        sample_omega = omega_fun(0)
        sample_gamma = gamma_fun(0)

        # Based on the type, choose the correct underlying class.
        if isinstance(sample_omega, complex):
            if isinstance(sample_gamma, complex):
                instance = Init_OC64_GC64(omega_fun, gamma_fun, nini, nmax, n, p)
            else:
                instance = Init_OC64_GF64(omega_fun, gamma_fun, nini, nmax, n, p)
        else:
            if isinstance(sample_gamma, complex):
                instance = Init_OF64_GC64(omega_fun, gamma_fun, nini, nmax, n, p)
            else:
                instance = Init_OF64_GF64(omega_fun, gamma_fun, nini, nmax, n, p)
        return instance
