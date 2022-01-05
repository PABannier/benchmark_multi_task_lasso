from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import warnings
    from numpy.linalg import norm
    from celer import MultiTaskLasso
    from sklearn.linear_model import MultiTaskLasso as MTL_Celer
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    """ScikitLearn solver"""

    name = "sklearn"

    requirements = ['sklearn']
    parameters = { 'oracle': [True, False] }

    def set_objective(self, X, Y, lmbd, n_orient):
        self.X, self.Y = X, Y
        self.lmbd = lmbd
        self.tol = 1e-8

        if self.oracle:
            self.clf_celer = MTL_Celer(
                alpha=self.lmbd / self.X.shape[0], 
                tol=self.tol / (Y ** 2).sum(),
                fit_intercept=False)
            self.clf_celer.fit(self.X, self.Y)

        self.clf = MultiTaskLasso(
            alpha=self.lmbd / self.X.shape[0], tol=self.tol / (Y ** 2).sum(),
            fit_intercept=False)

    def run(self, n_iter):
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        self.clf.max_iter = n_iter

        if self.oracle:
            self.true_support = norm(self.clf_celer.coef_.T, axis=1) != 0
            self.clf.fit(self.X[:, self.true_support], self.Y)
        else:
            self.clf.fit(self.X, self.Y)

    def get_result(self):
        if self.oracle:
            return self.clf.coef_.T, self.true_support
        return self.clf.coef_.T, None
