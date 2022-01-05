from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm
    from numba import njit
    from mtl_utils.common import (groups_norm2, get_lipschitz, get_duality_gap,
                                  build_full_coefficient_matrix)

if import_ctx.failed_import:

    def njit(f):  # noqa: F811
        return f


@njit
def _block_soft_thresh(x, u):
    norm_x = norm(x)
    if norm_x < u:
        return np.zeros_like(x), False
    else:
        return (1 - u / norm_x) * x, True


def _bcd(W, X, R, lipschitz, n_orient, active_set, alpha):
    n_positions = X.shape[1] // n_orient
    for j in range(n_positions):
        idx = slice(j * n_orient, (j + 1) * n_orient)
        X_j = X[:, idx].copy()
        W_j = W[idx].copy()
        W[idx], active_set[idx] = _block_soft_thresh(
            W_j + X_j.T @ R / lipschitz[j], alpha / lipschitz[j]
        )
        # if X_j[0, 0] != 0:
        if True:
            R += X_j @ W_j
        if np.all(active_set[idx]):
            R -= X_j @ W[idx]


_bcd_numbda = njit(_bcd)


def bcd(X, Y, lipschitz, init, _alpha, n_orient, bcd_factory, max_iter=2000,
        tol=1e-5):
    _, n_times = Y.shape
    _, n_features = X.shape

    if init is None:
        coef = np.zeros((n_features, n_times))
        R = Y.copy()
    else:
        coef = init
        R = Y - X @ coef

    X = X if np.isfortran(X) else np.asfortranarray(X)

    highest_d_obj = -np.inf
    active_set = np.zeros(n_features, dtype=bool)

    for _ in range(max_iter):
        bcd_factory(coef, X, R, lipschitz, n_orient, active_set, _alpha)

        _, p_obj, d_obj = get_duality_gap(
            X, Y, coef[active_set], active_set, _alpha, n_orient
        )

        highest_d_obj = max(d_obj, highest_d_obj)
        gap = p_obj - highest_d_obj

        if gap < tol:
            break

    coef = coef[active_set]
    return coef, active_set


class Solver(BaseSolver):
    """Block coordinate descent WITH Numba
    for Multi-Task LASSO
    """

    name = "numba_bcd_as"
    stopping_strategy = "callback"
    parameters = {"use_numba": (True, False)}

    requirements = ["numba"]

    def _prepare_bcd(self):
        _, n_sources = self.X.shape
        _, n_times = self.Y.shape

        self.W = np.zeros((n_sources, n_times))
        self.R = self.Y.copy()

        lipschitz_constants = get_lipschitz(self.X, self.n_orient)

        if self.use_numba:
            bcd_ = _bcd_numbda
        else:
            bcd_ = _bcd

        active_set = np.zeros(n_sources, dtype=bool)
        idx_large_corr = np.argsort(
            groups_norm2(np.dot(self.X.T, self.Y), self.n_orient)
        )
        new_active_idx = idx_large_corr[-self.active_set_size:]
        if self.n_orient > 1:
            new_active_idx = (
                self.n_orient * new_active_idx[:, None]
                + np.arange(self.n_orient)[None, :]
            ).ravel()

        active_set[new_active_idx] = True

        return lipschitz_constants, active_set, bcd_

    def set_objective(self, X, Y, lmbd, n_orient):
        self.X, self.Y = X, Y
        self.X = np.asfortranarray(self.X)
        self.lmbd = lmbd
        self.n_orient = n_orient
        self.active_set_size = 10
        self.max_iter = 100_000
        self.tol = 1e-8

        # Make sure we cache the numba compilation.
        lipschitz, active_set, bcd_ = self._prepare_bcd()

        bcd_(self.W, self.X, self.R, lipschitz, n_orient, active_set,
             self.lmbd)

    def run(self, callback):
        lipschitz, active_set, bcd_ = self._prepare_bcd()

        coef_init = None
        iter_idx = 0

        n_times = self.Y.shape[1]

        while callback(self.W):
            lipschitz_as = lipschitz[active_set[:: self.n_orient]]

            coef, as_ = bcd(self.X[:, active_set], self.Y, lipschitz_as,
                            coef_init, self.lmbd, self.n_orient, bcd_,
                            max_iter=self.max_iter, tol=self.tol)

            active_set[active_set] = as_.copy()
            idx_old_active_set = np.where(active_set)[0]

            self.W = build_full_coefficient_matrix(active_set, n_times, coef)

            if iter_idx < (self.max_iter - 1):
                R = self.Y - self.X[:, active_set] @ coef
                idx_large_corr = np.argsort(
                    groups_norm2(np.dot(self.X.T, R), self.n_orient)
                )
                new_active_idx = idx_large_corr[-self.active_set_size:]

                if self.n_orient > 1:
                    new_active_idx = (
                        self.n_orient * new_active_idx[:, None]
                        + np.arange(self.n_orient)[None, :]
                    )
                    new_active_idx = new_active_idx.ravel()

                active_set[new_active_idx] = True
                idx_active_set = np.where(active_set)[0]
                as_size = np.sum(active_set)
                coef_init = np.zeros((as_size, n_times), dtype=coef.dtype)
                idx = np.searchsorted(idx_active_set, idx_old_active_set)
                coef_init[idx] = coef

            iter_idx += 1

    def get_result(self):
        return self.W
