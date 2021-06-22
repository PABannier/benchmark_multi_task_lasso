from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm


def norm_l21(A, n_orient=1, copy=True):
    """L21 norm."""
    if A.size == 0:
        return 0.0
    if copy:
        A = A.copy()
    return np.sum(np.sqrt(groups_norm2(A, n_orient)))


def groups_norm2(A, n_orient):
    """Compute squared L2 norms of groups inplace."""
    n_positions = A.shape[0] // n_orient
    return np.sum(np.power(A, 2, A).reshape(n_positions, -1), axis=1)


def norm_l2inf(A, n_orient=1, copy=True):
    """L2-inf norm."""
    if A.size == 0:
        return 0.0
    if copy:
        A = A.copy()
    return np.sqrt(np.max(groups_norm2(A, n_orient)))


def get_alpha_max(X, Y, n_orient=1):
    return norm_l2inf(X.T @ Y, n_orient, copy=False)


class Objective(BaseObjective):
    name = "Objective"
    parameters = {"reg": [0.1], "n_orient": [3]}

    def __init__(self, reg=0.1, n_orient=1):
        self.reg = reg
        self.n_orient = n_orient

    def set_data(self, X, Y):
        self.X, self.Y = X, Y
        self.alpha_max = get_alpha_max(self.X, self.Y, self.n_orient)
        self.lmbd = self.reg * self.alpha_max

    def compute(self, W):
        R = self.Y - self.X @ W
        obj = 0.5 * norm(R, ord="fro") ** 2 + self.lmbd * norm_l21(
            W, self.n_orient
        )
        return obj

    def to_dict(self):
        return dict(X=self.X, Y=self.Y, lmbd=self.lmbd, n_orient=self.n_orient)
