from os import stat

import numpy as np
from numpy.linalg import norm

from numba import njit
from benchopt import BaseSolver

from ..utils import get_alpha_max, get_lipschitz


class BCDNoNumba(BaseSolver):
    """Block coordinate descent WITHOUT Numba
    for Multi-Task LASSO
    """

    name = "BCD (with Numba)"
    parameters = {"alpha_max_frac": [0.8, 0.6, 0.3]}

    def set_objective(self, G, M):
        self.G, self.M = G, M

    def run(self, n_iter):
        _, n_sources = self.G.shape
        _, n_times = self.M.shape

        active_set = np.zeros(n_sources, dtype=bool)

        alpha_max = get_alpha_max(self.G, self.M)
        alpha = alpha_max * self.parameters["alpha_max_frac"]

        self.X = np.zeros((n_sources, n_times))
        self.R = self.M - self.G @ self.X

        lipschitz_constants = get_lipschitz(self.G)
        alpha_lc = alpha / lipschitz_constants

        self.G = np.asfortranarray(self.G)
        one_over_lc = 1 / lipschitz_constants

        for _ in range(n_iter):
            self._bcd(one_over_lc, 1, alpha_lc, active_set)

    @staticmethod
    def _block_soft_thresh(x, u):
        norm_x = norm(x)
        if norm_x < u:
            return np.zeros_like(x), False
        else:
            return (1 - u / norm_x) * x, True

    def _bcd(self, one_over_lc, n_orient, alpha_lc, active_set):
        n_positions = self.X.shape[1] // n_orient
        for j in range(n_positions):
            idx = slice(j * n_orient, (j + 1) * n_orient)
            G_j = self.G[:, idx].copy()
            X_j = self.X[idx].copy()
            self.X[idx], active_set[idx] = self._block_soft_thresh(
                X_j + G_j.T @ self.R * one_over_lc[j], alpha_lc[j]
            )
            if X_j[0, 0] != 0:
                self.R += G_j @ X_j
            if np.all(active_set[idx] == True):
                self.R -= G_j @ self.X[idx]
