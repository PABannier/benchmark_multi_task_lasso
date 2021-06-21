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


def get_alpha_max(G, M, n_orient=1):
    return norm_l2inf(G.T @ M, n_orient, copy=False)


class Objective(BaseObjective):
    name = "Objective"
    parameters = {"reg": [1], "n_orient": [3]}  # lambda max: 0.138

    def __init__(self, reg=0.1, n_orient=1):
        self.reg = reg
        self.n_orient = n_orient

    def set_data(self, G, M):
        self.G, self.M = G, M
        self.alpha_max = get_alpha_max(self.G, self.M, self.n_orient)
        self.lmbd = self.reg * self.alpha_max

    def compute(self, X):
        R = self.M - self.G @ X
        obj = 0.5 * norm(R, ord="fro") ** 2 + self.lmbd * norm_l21(
            X, self.n_orient
        )
        print(np.count_nonzero(X.sum(axis=-1)))
        return obj

    def to_dict(self):
        return dict(G=self.G, M=self.M, lmbd=self.lmbd, n_orient=self.n_orient)
