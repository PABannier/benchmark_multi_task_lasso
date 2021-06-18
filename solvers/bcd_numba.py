from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm
    from numba import njit


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


def get_lipschitz(G, n_orient):
    if n_orient == 1:
        return np.sum(G * G, axis=0)
    else:
        n_positions = G.shape[1] // n_orient
        lc = np.empty(n_positions)
        for j in range(n_positions):
            G_tmp = G[:, (j * n_orient) : ((j + 1) * n_orient)]
            lc[j] = np.linalg.norm(np.dot(G_tmp.T, G_tmp), ord=2)
        return lc


def get_alpha_max(G, M, n_orient=1):
    return norm_l2inf(G.T @ M, n_orient)


@njit
def _block_soft_thresh(x, u):
    norm_x = norm(x)
    if norm_x < u:
        return np.zeros_like(x), False
    else:
        return (1 - u / norm_x) * x, True


def _bcd(X, G, R, one_over_lc, n_orient, alpha_lc, active_set):
    n_positions = X.shape[1] // n_orient
    for j in range(n_positions):
        idx = slice(j * n_orient, (j + 1) * n_orient)
        G_j = G[:, idx].copy()
        X_j = X[idx].copy()
        X[idx], active_set[idx] = _block_soft_thresh(
            X_j + G_j.T @ R * one_over_lc[j], alpha_lc[j]
        )
        if X_j[0, 0] != 0:
            R += G_j @ X_j
        if np.all(active_set[idx] == True):
            R -= G_j @ X[idx]


_bcd_numbda = njit(_bcd)


class Solver(BaseSolver):
    """Block coordinate descent WITH Numba
    for Multi-Task LASSO
    """

    name = "bcd_numba"
    stop_strategy = "callback"
    parameters = {"use_numba": (True, False)}

    def _prepare_bcd(self):
        _, n_sources = self.G.shape
        _, n_times = self.M.shape

        active_set = np.zeros(n_sources, dtype=bool)
        self.X = np.zeros((n_sources, n_times))
        self.R = self.M.copy()

        lipschitz_constants = get_lipschitz(self.G, self.n_orient)
        alpha_lc = self.lmbd / lipschitz_constants
        one_over_lc = 1 / lipschitz_constants

        if self.use_numba:
            bcd_ = _bcd_numbda
        else:
            bcd_ = _bcd

        return one_over_lc, alpha_lc, active_set, bcd_

    def set_objective(self, G, M, lmbd, n_orient):
        self.G, self.M = G, M
        self.G = np.asfortranarray(self.G)
        self.lmbd = lmbd
        self.n_orient = n_orient

        # Make sure we cache the numba compilation.
        one_over_lc, alpha_lc, active_set, bcd_ = self._prepare_bcd()

        import ipdb; ipdb.set_trace()

        bcd_(
            self.X, self.G, self.R, one_over_lc, n_orient, alpha_lc, active_set
        )

    def run(self, callback):
        one_over_lc, alpha_lc, active_set, bcd_ = self._prepare_bcd()

        while callback(self.X):
            bcd_(
                self.X,
                self.G,
                self.R,
                one_over_lc,
                self.n_orient,
                alpha_lc,
                active_set,
            )

    def get_result(self):
        return self.X
