from mtl_utils.common import norm_l2inf
from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from mne.inverse_sparse.mxne_inverse import mixed_norm_solver
    from mtl_utils.common import build_full_coefficient_matrix, sum_squared


class Solver(BaseSolver):
    name = "bcd_mne"
    stop_strategy = "iteration"

    def set_objective(self, X, Y, lmbd, n_orient):
        self.X, self.Y = X.copy(), Y
        self.n_orient = n_orient
        self.tol = 1e-12 * sum_squared(self.Y)

        # Rescale alpha to be in [0, 100)
        self.lmbd_max = norm_l2inf(X.T @ Y, n_orient, copy=False) * 0.01
        self.reg = lmbd / (self.lmbd_max * 100)
        self.X /= self.lmbd_max

    def run(self, n_iter):
        max_iter = n_iter + 1
        W_, as_, _ = mixed_norm_solver(self.Y, self.X, self.reg * 100,
                                       maxit=max_iter, tol=self.tol,
                                       n_orient=self.n_orient, solver="bcd",
                                       verbose=0)
        self.W = build_full_coefficient_matrix(as_, self.Y.shape[1], W_)

    def get_result(self):
        return self.W
