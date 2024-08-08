import sys
import os
import time
sys.path.insert(0, os.path.abspath('/Users/sbronder/opensource/riccaticpp/build'))

import pyriccaticpp as ric
import numpy as np
import scipy.special as sp
import mpmath
import warnings
import pytest

l = 1e1
w = lambda x: l * np.sqrt(1.0 -
        np.square(x) * np.cos(3.0 * x))
g = lambda x: np.zeros_like(x)

info = ric.Init(w, g, 16, 32, 32, 32)
#import pdb; pdb.set_trace()
info.mem_info()
xi = -1.0
xf = 1.0
eps = 1e-12
epsh = 1e-13
yi = complex(0.2913132934408612e0)
dyi = complex(7e-14)
test1 = ric.choose_osc_stepsize(info, xi, eps, epsh)
print("test1 = ", test1)
#test1 = ric.choose_nonosc_stepsize(info, xi, eps, epsh)
print("test1 = ", test1)
N = 100
begin_t = time.time()
for i in range(N):
  res = ric.evolve(info = info, xi = xi, xf = xf, yi = yi, dyi = dyi, eps = eps, epsilon_h = epsh, init_stepsize = 0.1)
end_t = time.time()
print("Time = ", (end_t - begin_t)/N)
print("res = ", res)

print("DONE!!")
#ytrue = np.array([mpmath.airyai(-x) + 1j*mpmath.airybi(-x) for x in xeval])