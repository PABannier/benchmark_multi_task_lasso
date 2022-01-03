from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from flashcd.estimators import MultiTaskLasso
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from mtl_utils.common import sum_squared


class Solver(BaseSolver):
    """FlashCD solver"""

    name = "flashcd"
    parameters = { 'ws': [True, False], "use_acc": [True]}

    def set_objective(self, X, Y, lmbd, n_orient):
        self.X, self.Y = X, Y
        self.lmbd = lmbd
        self.tol = 1e-8

        if self.ws:
            p0 = 10
            prune = True
        else:
            p0 = X.shape[1]
            prune = False

        self.clf = MultiTaskLasso(
            alpha=self.lmbd / self.X.shape[0], tol=self.tol / sum_squared(Y),
            fit_intercept=False, p0=p0, prune=prune, use_acc=self.use_acc)

        # Caching Numba compilation
        self.run(1)

    def run(self, n_iter):
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.Y)

    def get_result(self):
        return self.clf.coef_.T
    
    @staticmethod
    def get_next(previous):
        "Linear growth for n_iter."
        return previous + 1
