import warnings
from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from celer import MultiTaskLasso
    from sklearn.exceptions import ConvergenceWarning
    from mtl_utils.common import sum_squared


class Solver(BaseSolver):
    name = "celer_cython"
    stop_strategy = 'iteration'

    def skip(self, X, Y, lmbd, n_orient):
        if n_orient != 1:
            return True, "Celer does not support n_orient != 1"
        return False, None

    def set_objective(self, X, Y, lmbd, n_orient):
        self.X, self.Y, self.lmbd = X, Y, lmbd
        self.maxit = 100_000
        self.tol = 1e-8
        self.clf = MultiTaskLasso(alpha=lmbd / len(Y),
                                  tol=self.tol / sum_squared(Y),
                                  fit_intercept=False,
                                  verbose=0, prune=True)

    def run(self, n_iter):
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        if n_iter == 0:
            self.clf.coef_ = np.zeros((self.Y.shape[1], self.X.shape[1]))
        else:
            self.clf.max_iter = n_iter
            self.clf.fit(self.X, self.Y)

    def get_result(self):
        return self.clf.coef_.T