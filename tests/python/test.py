import pyriccaticpp as ric
import numpy as np
import scipy.special as sp
import mpmath
import warnings
import pytest

w = lambda x: np.sqrt(x)
g = lambda x: np.zeros_like(x)

info = ric.Init(w, g, 16, 32, 32, 32)

xi = 1e0
xf = 1e6
eps = 1e-12
epsh = 1e-13
yi = sp.airy(-xi)[0] + 1j*sp.airy(-xi)[2]
dyi = -sp.airy(-xi)[1] - 1j*sp.airy(-xi)[3]
xs, ys, dys, ss, ps, stypes, yeval = ric.evolve(info, xi, xf, yi, dyi,\
                                                    eps = eps, epsh = epsh)
ytrue = np.array([mpmath.airyai(-x) + 1j*mpmath.airybi(-x) for x in xeval])