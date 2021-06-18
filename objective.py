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
    return norm_l2inf(G.T @ M, n_orient)


class Objective(BaseObjective):
    name = "Objective"

    parameters = {"reg": [0.3, 0.5, 0.7]}

    def __init__(self, reg=0.1):
        self.reg = reg

    def set_data(self, G, M):
        self.G, self.M = G, M
        self.lmbd = self.reg * get_alpha_max(self.G, self.M)

    def compute(self, X):
        R = self.M - self.G @ X
        return 0.5 * norm(R, ord="fro") ** 2 + self.lmbd * norm_l21(X)

    def to_dict(self):
        return dict(G=self.G, M=self.M, lmbd=self.lmbd)
