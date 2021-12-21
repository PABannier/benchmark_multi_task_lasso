from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.linear_model import MultiTaskLasso
    from mtl_utils.common import (groups_norm2, get_lipschitz,
                                  sum_squared)


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

    name = "ws_sk"
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
        new_active_idx = idx_large_corr[-self.active_set_size:]
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

    def build_full_coefficient_matrix(self, active_set, n_times, coef):
        """Building full coefficient matrix and filling active set with
        non-zero coefficients"""
        final_coef_ = np.zeros((len(active_set), n_times))
        if coef is not None:
            final_coef_[active_set] = coef
        self.W = final_coef_

    def get_result(self):
        return self.W
