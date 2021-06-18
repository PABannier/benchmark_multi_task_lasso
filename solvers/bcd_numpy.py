from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm


def groups_norm2(A, n_orient):
    """Compute squared L2 norms of groups inplace."""
    n_positions = A.shape[0] // n_orient
    return np.sum(np.power(A, 2, A).reshape(n_positions, -1), axis=1)


def norm_l2inf(A, n_orient=1, copy=True):
    """L2-inf norm."""
    if A.size == 0:
        return 0.0
    if copy:
        A = A.copy()
    return np.sqrt(np.max(groups_norm2(A, n_orient)))


def norm_l21(A, n_orient=1, copy=True):
    """L21 norm."""
    if A.size == 0:
        return 0.0
    if copy:
        A = A.copy()
    return np.sum(np.sqrt(groups_norm2(A, n_orient)))


def get_lipschitz(G):
    return np.sum(G * G, axis=0)


def get_alpha_max(G, M, n_orient=1):
    return norm_l2inf(G.T @ M, n_orient)


class Solver(BaseSolver):
    """Block coordinate descent WITH Numpy
    for Multi-Task LASSO
    """

    name = "bcd_numpy"

    def set_objective(self, G, M, lmbd):
        self.G, self.M = G, M
        self.G = np.asfortranarray(self.G)
        self.lmbd = lmbd

    def run(self, n_iter):
        _, n_sources = self.G.shape
        _, n_times = self.M.shape

        active_set = np.zeros(n_sources, dtype=bool)

        self.X = np.zeros((n_sources, n_times))
        self.R = self.M.copy()

        lipschitz_constants = get_lipschitz(self.G)
        alpha_lc = self.lmbd / lipschitz_constants
        one_over_lc = 1 / lipschitz_constants

        for _ in range(n_iter):
            self._bcd(one_over_lc, 1, alpha_lc, active_set)

    def get_result(self):
        return self.X

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
