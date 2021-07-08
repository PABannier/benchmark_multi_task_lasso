from bcd_celer import celer_dual_mtl
from numpy.linalg import norm

import numpy as np


X = np.random.randn(50, 100)
Y = np.random.randn(50, 7)


alpha_max = np.max(norm(X.T @ Y, ord=2, axis=1)) / len(Y)

alpha = alpha_max / 1.1
W, Theta = celer_dual_mtl(X, Y, alpha, 10, verbose=2)


print(np.max(norm(X.T @ Theta, ord=2, axis=1)))
