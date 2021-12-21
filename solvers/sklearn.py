from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.linear_model import MultiTaskLasso


class Solver(BaseSolver):
    """ScikitLearn solver"""

    name = "sklearn"

    def set_objective(self, X, Y, lmbd, n_orient):
        self.X, self.Y = X, Y
        self.lmbd = lmbd

        self.clf = MultiTaskLasso(alpha=self.lmbd / self.X.shape[0])

    def run(self, n_iter):
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.Y)

    def get_result(self):
        return self.clf.coef_.T
