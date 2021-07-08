from bcd_celer import celer_dual_mtl
from numpy.linalg import norm

import numpy as np


X = np.random.randn(50, 100)
Y = np.random.randn(50, 7)


alpha_max = np.max(norm(X.T @ Y, ord=2, axis=1)) / len(Y)

alpha = alpha_max / 1.1
W, Theta, R_ = celer_dual_mtl(X, Y, alpha, 10, gap_freq=1, verbose=2)

R = Y - X @ W
np.testing.assert_allclose(R, R_)
# print(np.max(norm(X.T @ Theta, ord=2, axis=1)))
# print(np.max(norm(X.T @ R / len(Y) / alpha, ord=2, axis=1)))
