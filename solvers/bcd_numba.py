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


def get_lipschitz(X, n_orient):
    if n_orient == 1:
        return np.sum(X * X, axis=0)
    else:
        n_positions = X.shape[1] // n_orient
        lc = np.empty(n_positions)
        for j in range(n_positions):
            X_tmp = X[:, (j * n_orient) : ((j + 1) * n_orient)]
            lc[j] = np.linalg.norm(np.dot(X_tmp.T, X_tmp), ord=2)
        return lc


def sum_squared(X):
    X_flat = X.ravel(order="F" if np.isfortran(X) else "C")
    return np.dot(X_flat, X_flat)


def get_alpha_max(X, Y, n_orient=1):
    return norm_l2inf(X.T @ Y, n_orient)


def get_duality_gap(X, Y, W, active_set, alpha, n_orient=1, primal_only=False):
    Y_hat = np.dot(X[:, active_set], W)
    R = Y - Y_hat
    penalty = norm_l21(W, n_orient, copy=True)
    nR2 = sum_squared(R)
    p_obj = 0.5 * nR2 + alpha * penalty

    if primal_only:
        return p_obj

    dual_norm = norm_l2inf(np.dot(X.T, R), n_orient, copy=False)
    scaling = alpha / dual_norm
    scaling = min(scaling, 1.0)
    d_obj = (scaling - 0.5 * (scaling ** 2)) * nR2 + scaling * np.sum(
        R * Y_hat
    )
    gap = p_obj - d_obj
    return gap, p_obj, d_obj


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
        if np.all(active_set[idx] == True):
            R -= X_j @ W[idx]


_bcd_numbda = njit(_bcd)


def bcd(
    X,
    Y,
    lipschitz,
    init,
    _alpha,
    n_orient,
    bcd_factory,
    max_iter=2000,
    tol=1e-5,
):
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

    name = "bcd_numba"
    stop_strategy = "callback"
    parameters = {"use_numba": (True, False)}

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
        new_active_idx = idx_large_corr[-self.active_set_size :]
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
        self.max_iter = 3000
        self.tol = 1e-8

        # Make sure we cache the numba compilation.
        lipschitz, active_set, bcd_ = self._prepare_bcd()

        bcd_(
            self.W, self.X, self.R, lipschitz, n_orient, active_set, self.lmbd
        )

    def run(self, callback):
        lipschitz, active_set, bcd_ = self._prepare_bcd()

        coef_init = None
        iter_idx = 0

        n_times = self.Y.shape[1]

        while callback(self.W):
            lipschitz_as = lipschitz[active_set[:: self.n_orient]]

            coef, as_ = bcd(
                self.X[:, active_set],
                self.Y,
                lipschitz_as,
                coef_init,
                self.lmbd,
                self.n_orient,
                bcd_,
                max_iter=self.max_iter,
                tol=self.tol,
            )

            active_set[active_set] = as_.copy()
            idx_old_active_set = np.where(active_set)[0]

            self.build_full_coefficient_matrix(active_set, n_times, coef)

            if iter_idx < (self.max_iter - 1):
                R = self.Y - self.X[:, active_set] @ coef
                idx_large_corr = np.argsort(
                    groups_norm2(np.dot(self.X.T, R), self.n_orient)
                )
                new_active_idx = idx_large_corr[-self.active_set_size :]

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

        # XR = self.G.T @ (self.M - self.G @ self.X)
        # assert norm_l2inf(XR, self.n_orient) <= self.lmbd + 1e-12, "KKT check"

    def build_full_coefficient_matrix(self, active_set, n_times, coef):
        """Building full coefficient matrix and filling active set with
        non-zero coefficients"""
        final_coef_ = np.zeros((len(active_set), n_times))
        if coef is not None:
            final_coef_[active_set] = coef
        self.W = final_coef_

    def get_result(self):
        return self.W
