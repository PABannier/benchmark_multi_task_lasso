from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import warnings
    import numpy as np
    from numpy.linalg import norm
    from celer import MultiTaskLasso as MTL_Celer
    from sklearn.linear_model import MultiTaskLasso
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    """ScikitLearn solver"""

    name = "sklearn"

    requirements = ['sklearn']
    parameters = {
        'oracle': [True, False]
    }

    def set_objective(self, X, Y, lmbd, n_orient):
        self.X, self.Y = X, Y
        self.lmbd = lmbd
        self.tol = 1e-8

        if self.oracle:
            clf_celer = MTL_Celer(
                alpha=self.lmbd / self.X.shape[0],
                tol=self.tol / (Y ** 2).sum(), fit_intercept=False
            ).fit(self.X, self.Y)
            self.true_support = norm(clf_celer.coef_.T, axis=1) != 0

        self.clf = MultiTaskLasso(
            alpha=self.lmbd / self.X.shape[0], tol=self.tol / (Y ** 2).sum(),
            fit_intercept=False, warm_start=False)

    def run(self, n_iter):
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        self.clf.max_iter = n_iter

        if self.oracle:
            self.clf.fit(self.X[:, self.true_support], self.Y)
        else:
            self.clf.fit(self.X, self.Y)

    def get_result(self):
        if self.oracle:
            W = np.zeros((self.X.shape[1], self.Y.shape[1]))
            W[self.true_support] = self.clf.coef_.T
            return W
        return self.clf.coef_.T
