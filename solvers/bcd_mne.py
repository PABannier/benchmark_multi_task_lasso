from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from mne.inverse_sparse.mxne_inverse import mixed_norm_solver
    from mtl_utils.common import build_full_coefficient_matrix


class Solver(BaseSolver):
    name = "bcd_mne"
    stop_strategy = "iteration"

    def set_objective(self, X, Y, lmbd, n_orient):
        self.X, self.Y, self.lmbd = X, Y, lmbd
        self.maxit = 100_000
        self.tol = 1e-12

    def run(self, n_iter):
        W, as_, _ = mixed_norm_solver(self.Y, self.X, self.lmbd,
                                      maxit=self.maxit, tol=self.tol,
                                      solver="bcd", verbose=0)
        self.W = build_full_coefficient_matrix(as_, self.Y.shape[1], W)

    def get_result(self):
        return self.W
