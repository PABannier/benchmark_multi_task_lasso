from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from mtl_utils.common import (groups_norm2, get_lipschitz, bcd_pass,
                                  build_full_coefficient_matrix)


class Solver(BaseSolver):
    """Block coordinate descent with low-level BLAS function calls"""

    name = "bcd_as_aa_blas"
    stop_strategy = "callback"
    # parameters = {"accelerated": (True, False)}

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
        idx_large_corr = np.argsort(groups_norm2(np.dot(self.X.T, self.Y),
                                    self.n_orient))
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

            coef, as_ = bcd_pass(self.X[:, active_set], self.Y,
                                 lipschitz_consts_tmp, coef_init, self.lmbd,
                                 self.n_orient, accelerated=True,
                                 max_iter=self.max_iter, tol=self.tol)

            active_set[active_set] = as_.copy()
            idx_old_active_set = np.where(active_set)[0]

            self.W = build_full_coefficient_matrix(active_set, n_times, coef)

            if iter_idx < (self.max_iter - 1):
                R = self.Y - self.X[:, active_set] @ coef
                idx_large_corr = np.argsort(groups_norm2(np.dot(self.X.T, R),
                                            self.n_orient))
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
