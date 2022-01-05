from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import warnings
    from sklearn.linear_model import MultiTaskLasso
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    """ScikitLearn solver"""

    name = "sklearn"

    install_cmd = 'conda'
    requirements = ['scikit-learn']

    def set_objective(self, X, Y, lmbd, n_orient):
        self.X, self.Y = X, Y
        self.lmbd = lmbd
        self.tol = 1e-8

        self.clf = MultiTaskLasso(
            alpha=self.lmbd / self.X.shape[0], tol=self.tol / (Y ** 2).sum(),
            fit_intercept=False)

    def run(self, n_iter):
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.Y)

    def get_result(self):
        return self.clf.coef_.T
