from benchopt import BaseSolver
from benchopt import safe_import_context
from numpy.core.numeric import isfortran


with safe_import_context() as import_ctx:
    import functools
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


def bcd_(
    X,
    Y,
    lipschitz,
    init,
    _alpha,
    n_orient,
    accelerated,
    K=5,
    max_iter=2000,
    tol=1e-5,
):
    n_samples, n_times = Y.shape
    n_samples, n_features = X.shape
    n_positions = n_features // n_orient

    if init is None:
        coef = np.zeros((n_features, n_times))
        R = Y.copy()
    else:
        coef = init
        R = Y - X @ coef

    X = X if np.isfortran(X) else np.asfortranarray(X)

    if accelerated:
        last_K_coef = np.empty((K + 1, n_features, n_times))
        U = np.zeros((K, n_features * n_times))

    highest_d_obj = -np.inf
    active_set = np.zeros(n_features, dtype=bool)

    for iter_idx in range(max_iter):
        coef_j_new = np.zeros_like(coef[:n_orient, :], order="C")
        dgemm = _get_dgemm()

        for j in range(n_positions):
            idx = slice(j * n_orient, (j + 1) * n_orient)
            coef_j = coef[idx]
            X_j = X[:, idx]

            dgemm(
                alpha=1 / lipschitz[j],
                beta=0.0,
                a=R.T,
                b=X_j,
                c=coef_j_new.T,
                overwrite_c=True,
            )

            if coef_j[0, 0] != 0:
                dgemm(
                    alpha=1.0,
                    beta=1.0,
                    a=coef_j.T,
                    b=X_j.T,
                    c=R.T,
                    overwrite_c=True,
                )
                coef_j_new += coef_j

            block_norm = np.sqrt(sum_squared(coef_j_new))
            alpha_lc = _alpha / lipschitz[j]

            if block_norm <= alpha_lc:
                coef_j.fill(0.0)
                active_set[idx] = False
            else:
                shrink = max(1.0 - alpha_lc / block_norm, 0.0)
                coef_j_new *= shrink

                dgemm(
                    alpha=-1.0,
                    beta=1.0,
                    a=coef_j_new.T,
                    b=X_j.T,
                    c=R.T,
                    overwrite_c=True,
                )
                coef_j[:] = coef_j_new
                active_set[idx] = True

        _, p_obj, d_obj = get_duality_gap(
            X, Y, coef[active_set], active_set, _alpha, n_orient
        )
        highest_d_obj = max(d_obj, highest_d_obj)
        gap = p_obj - highest_d_obj

        if accelerated:
            last_K_coef[iter_idx % (K + 1)] = coef

            if iter_idx % (K + 1) == K:
                for k in range(K):
                    U[k] = last_K_coef[k + 1].ravel() - last_K_coef[k].ravel()

                C = U @ U.T

                try:
                    z = np.linalg.solve(C, np.ones(K))
                    c = z / z.sum()

                    coef_acc = np.sum(
                        last_K_coef[:-1] * c[:, None, None], axis=0
                    )
                    active_set_acc = norm(coef_acc, axis=1) != 0

                    p_obj_acc = get_duality_gap(
                        X,
                        Y,
                        coef_acc[active_set_acc],
                        active_set_acc,
                        _alpha,
                        n_orient,
                        primal_only=True,
                    )

                    if p_obj_acc < p_obj:
                        coef = coef_acc
                        active_set = active_set_acc
                        R = Y - X[:, active_set] @ coef[active_set]

                except np.linalg.LinAlgError:
                    print("LinAlg Error")

        if gap < tol:
            print(f"Fitting ended after iteration {iter_idx + 1}.")
            break

    coef = coef[active_set]
    return coef, active_set


class Solver(BaseSolver):
    """Block coordinate descent with
    low-level BLAS function calls"""

    name = "bcd_blas_low_level"
    stop_strategy = "callback"
    parameters = {"accelerated": (True, False)}

    def set_objective(self, X, Y, lmbd, n_orient):
        self.X, self.Y = X, Y
        self.X = np.asfortranarray(self.X)
        self.lmbd = lmbd
        self.n_orient = n_orient
        self.active_set_size = 10
        self.tol = 1e-5
        self.max_iter = 2000

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

            coef, as_ = bcd_(
                self.X[:, active_set],
                self.Y,
                lipschitz_consts_tmp,
                coef_init,
                self.lmbd,
                self.n_orient,
                accelerated=self.accelerated,
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
