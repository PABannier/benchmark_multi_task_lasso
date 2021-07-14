from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from celer import MultiTaskLasso
    from mtl_utils.common import (groups_norm2, get_lipschitz,
                                  sum_squared)


def cd_(X, Y, alpha, init, maxit=10000, tol=1e-8, n_orient=1, dgap_freq=10):
    clf = MultiTaskLasso(alpha=alpha / len(Y), tol=tol / sum_squared(Y),
                         normalize=False, fit_intercept=False, max_iter=maxit)
    W = clf.coef_.T
    return W


class Solver(BaseSolver):
    name = "bcd_celer_cython"
    stop_strategy = 'iteration'

    def set_objective(self, X, Y, lmbd, n_orient):
        self.X, self.Y, self.lmbd = X, Y, lmbd
        self.maxit = 100_000
        self.tol = 1e-12
        n_samples = self.X.shape[0]
        self.clf = MultiTaskLasso(alpha=lmbd / len(Y), tol=self.tol/sum_squared(Y),
                                  normalize=False, fit_intercept=False, 
                                  max_iter=self.maxit)
    
    def run(self, n_iter):
        self.clf.max_iter = n_iter + 1
        self.clf.fit(self.X, self.Y)
    
    def get_result(self):
        return self.clf.coef_.T

