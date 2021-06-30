import numpy as np


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


def norm_l21(A, n_orient=1, copy=True):
    """L21 norm."""
    if A.size == 0:
        return 0.0
    if copy:
        A = A.copy()
    return np.sum(np.sqrt(groups_norm2(A, n_orient)))


def get_lipschitz(X, n_orient):
    if n_orient == 1:
        return np.sum(X * X, axis=0)
    else:
        n_positions = X.shape[1] // n_orient
        lc = np.empty(n_positions)
        for j in range(n_positions):
            X_tmp = X[:, (j * n_orient): ((j + 1) * n_orient)]
            lc[j] = np.linalg.norm(np.dot(X_tmp.T, X_tmp), ord=2)
        return lc


def sum_squared(X):
    X_flat = X.ravel(order="F" if np.isfortran(X) else "C")
    return np.dot(X_flat, X_flat)


def get_alpha_max(X, Y, n_orient=1):
    return norm_l2inf(X.T @ Y, n_orient)


def get_duality_gap(X, Y, W, active_set, alpha, n_orient=1, primal_only=False):
    Y_hat = np.dot(X[:, active_set], W)
    R = Y - Y_hat
    penalty = norm_l21(W, n_orient, copy=True)
    nR2 = sum_squared(R)
    p_obj = 0.5 * nR2 + alpha * penalty

    if primal_only:
        return p_obj

    dual_norm = norm_l2inf(np.dot(X.T, R), n_orient, copy=False)
    scaling = alpha / dual_norm
    scaling = min(scaling, 1.0)
    d_obj = (scaling - 0.5 * (scaling ** 2)) * nR2 + scaling * np.sum(
        R * Y_hat
    )
    gap = p_obj - d_obj
    return gap, p_obj, d_obj
