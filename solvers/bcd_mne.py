from mtl_utils.common import norm_l2inf
from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from mne.inverse_sparse.mxne_inverse import mixed_norm_solver
    from mtl_utils.common import build_full_coefficient_matrix, sum_squared


class Solver(BaseSolver):
    name = "bcd_mne"
    stop_strategy = "iteration"

    def set_objective(self, X, Y, lmbd, n_orient):
        self.X, self.Y = X, Y
        self.n_orient = n_orient
        self.tol = 1e-12
        self.lmbd = lmbd

    def run(self, n_iter):
        if n_iter == 0:
            self.W = np.zeros((self.X.shape[1], self.Y.shape[1]))
        else:
            W_, as_, _ = mixed_norm_solver(self.Y, self.X, self.lmbd,
                                           maxit=n_iter, tol=self.tol,
                                           n_orient=self.n_orient,
                                           solver="bcd", verbose=0)
            self.W = build_full_coefficient_matrix(as_, self.Y.shape[1], W_)

    def get_result(self):
        return self.W
