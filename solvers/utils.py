import numpy as np


def groups_norm2(A, n_orient):
    """Compute squared L2 norms of groups inplace."""
    n_positions = A.shape[0] // n_orient
    return np.sum(np.power(A, 2, A).reshape(n_positions, -1), axis=1)


def norm_l2inf(A, n_orient, copy=True):
    """L2-inf norm."""
    if A.size == 0:
        return 0.0
    if copy:
        A = A.copy()
    return np.sqrt(np.max(groups_norm2(A, n_orient)))


def get_lipschitz(self, G):
    return np.sum(G * G, axis=0)


def get_alpha_max(G, M, n_orient=1):
    return norm_l2inf(G.T @ M, n_orient)
