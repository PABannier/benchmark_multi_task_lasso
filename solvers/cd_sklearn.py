from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import functools
    import numpy as np
    from numpy.linalg import norm
    from sklearn.linear_model import MultiTaskLasso


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


@functools.lru_cache(None)
def _get_dgemm():
    return _get_blas_funcs(np.float64, "gemm")


@functools.lru_cache(None)
def _get_blas_funcs(dtype, names):
    from scipy import linalg

    return linalg.get_blas_funcs(names, (np.empty(0, dtype),))


def cd_(
    X,
    Y,
    lipschitz_constant,
    alpha,
    init,
    maxit=10000,
    tol=1e-8,
    n_orient=1,
    dgap_freq=10,
):
    clf = MultiTaskLasso(
        alpha=alpha / len(Y),
        tol=tol / sum_squared(Y),
        normalize=False,
        fit_intercept=False,
        max_iter=maxit,
        warm_start=True,
    )
    if init is not None:
        clf.coef_ = init.T
    else:
        clf.coef_ = np.zeros((X.shape[1], Y.shape[1])).T
    clf.fit(X, Y)

    W = clf.coef_.T
    active_set = np.any(W, axis=1)
    W = W[active_set]
    return W, active_set


class Solver(BaseSolver):
    """Block coordinate descent with
    low-level BLAS function calls"""

    name = "cd_sklearn"
    stop_strategy = "callback"

    def set_objective(self, X, Y, lmbd, n_orient):
        self.X, self.Y = X, Y
        self.X = np.asfortranarray(self.X)
        self.lmbd = lmbd
        self.n_orient = n_orient
        self.active_set_size = 10
        self.tol = 1e-8
        self.max_iter = 3000

    def run(self, callback):
        n_features = self.X.shape[1]
        n_times = self.Y.shape[1]

        lipschitz_consts = get_lipschitz(self.X, self.n_orient)

        # Initializing active set
        active_set = np.zeros(n_features, dtype=bool)
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
        as_size = np.sum(active_set)

        coef_init = None
        self.W = np.zeros((n_features, n_times))
        iter_idx = 0

        while callback(self.W):
            lipschitz_consts_tmp = lipschitz_consts[
                active_set[:: self.n_orient]
            ]

            coef, as_ = cd_(
                self.X[:, active_set],
                self.Y,
                lipschitz_consts_tmp,
                self.lmbd,
                coef_init,
                maxit=self.max_iter,
                tol=self.tol,
                n_orient=self.n_orient,
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

    def build_full_coefficient_matrix(self, active_set, n_times, coef):
        """Building full coefficient matrix and filling active set with
        non-zero coefficients"""
        final_coef_ = np.zeros((len(active_set), n_times))
        if coef is not None:
            final_coef_[active_set] = coef
        self.W = final_coef_

    def get_result(self):
        return self.W
