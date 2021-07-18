import functools
from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm
    from mtl_utils.common import (groups_norm2, get_lipschitz,
                                  sum_squared, get_duality_gap)


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
    _, n_times = Y.shape
    _, n_features = X.shape
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

        if gap < tol:
            break

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

    coef = coef[active_set]
    return coef, active_set


class Solver(BaseSolver):
    """Block coordinate descent with low-level BLAS function calls"""

    name = "bcd_geom_as"
    stop_strategy = "callback"

    def set_objective(self, X, Y, lmbd, n_orient):
        self.X, self.Y = X, Y
        self.X = np.asfortranarray(self.X)
        self.lmbd = lmbd
        self.n_orient = n_orient
        self.intial_active_set_size = 10
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
        new_active_idx = idx_large_corr[-self.intial_active_set_size:]
        if self.n_orient > 1:
            new_active_idx = (
                self.n_orient * new_active_idx[:, None]
                + np.arange(self.n_orient)[None, :]
            ).ravel()

        active_set[new_active_idx] = True

        coef_init = None
        self.W = np.zeros((n_features, n_times))

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
                accelerated=True,
                max_iter=self.max_iter,
                tol=self.tol,
            )

            active_set[active_set] = as_
            self.build_full_coefficient_matrix(active_set, n_times, coef)

            # Geometric growth of active set
            idx_old_active_set = np.where(active_set)[0]
            old_as_size = np.sum(active_set)

            R = self.Y - self.X[:, active_set] @ coef
            idx_large_corr = np.argsort(
                groups_norm2(np.dot(self.X.T, R), self.n_orient)
            )
            new_active_idx = idx_large_corr[-old_as_size:]

            if self.n_orient > 1:
                new_active_idx = (
                    self.n_orient * new_active_idx[:, None]
                    + np.arange(self.n_orient)[None, :]
                )
                new_active_idx = new_active_idx.ravel()

            active_set[new_active_idx] = True
            idx_active_set = np.where(active_set)[0]
            as_size = np.sum(active_set)  # 2 * old_as_size
            coef_init = np.zeros((as_size, n_times), dtype=coef.dtype)
            idx = np.searchsorted(idx_active_set, idx_old_active_set)
            coef_init[idx] = coef

    def build_full_coefficient_matrix(self, active_set, n_times, coef):
        """Building full coefficient matrix and filling active set with
        non-zero coefficients"""
        final_coef_ = np.zeros((len(active_set), n_times))
        if coef is not None:
            final_coef_[active_set] = coef
        self.W = final_coef_

    def get_result(self):
        return self.W
