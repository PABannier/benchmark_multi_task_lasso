from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm
    from mtl_utils.common import (groups_norm2, get_duality_gap,
                                  build_full_coefficient_matrix)


def prox_l21(Y, alpha, n_orient, shape=None):
    if len(Y) == 0:
        return np.zeros_like(Y), np.zeros((0,), dtype=bool)
    if shape is not None:
        shape_init = Y.shape
        Y = Y.reshape(*shape)
    n_positions = Y.shape[0] // n_orient

    rows_norm = np.sqrt(
        (Y * Y.conj()).real.reshape(n_positions, -1).sum(axis=1)
    )
    # Ensure shrink is >= 0 while avoiding any division by zero
    shrink = np.maximum(1.0 - alpha / np.maximum(rows_norm, alpha), 0.0)
    active_set = shrink > 0.0
    if n_orient > 1:
        active_set = np.tile(active_set[:, None], [1, n_orient]).ravel()
        shrink = np.tile(shrink[:, None], [1, n_orient]).ravel()
    Y = Y[active_set]
    if shape is None:
        Y *= shrink[active_set][:, np.newaxis]
    else:
        Y *= shrink[active_set][:, np.newaxis, np.newaxis]
        Y = Y.reshape(-1, *shape_init[1:])
    return Y, active_set


def pgd_(X, Y, lipschitz, init, _alpha, n_orient, dgap_freq=10, max_iter=200,
         tol=1e-8):
    n_samples, n_times = Y.shape
    _, n_features = X.shape

    if n_features < n_samples:
        gram = np.dot(X.T, X)
        GTM = np.dot(X.T, Y)
    else:
        gram = None

    if init is None:
        W = 0.0
        R = Y.copy()
        if gram is not None:
            R = np.dot(X.T, R)
    else:
        W = init
        if gram is None:
            R = Y - np.dot(X, W)
        else:
            R = GTM - np.dot(gram, W)

    t = 1.0
    B = np.zeros((n_features, n_times))  # FISTA aux variable
    highest_d_obj = -np.inf
    active_set = np.ones(n_features, dtype=bool)  # start with full AS

    for i in range(max_iter):
        W0, active_set_0 = W, active_set
        if gram is None:
            B += np.dot(X.T, R) / lipschitz  # ISTA step
        else:
            B += R / lipschitz  # ISTA step
        W, active_set = prox_l21(B, _alpha / lipschitz, n_orient)

        t0 = t
        t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t ** 2))
        B.fill(0.0)
        dt = (t0 - 1.0) / t
        B[active_set] = (1.0 + dt) * W
        B[active_set_0] -= dt * W0
        B_as = active_set_0 | active_set

        if gram is None:
            R = Y - np.dot(X[:, B_as], B[B_as])
        else:
            R = GTM - np.dot(gram[:, B_as], B[B_as])

        if (i + 1) % dgap_freq == 0:
            _, p_obj, d_obj = get_duality_gap(X, Y, W, active_set, _alpha,
                                              n_orient)
            highest_d_obj = max(d_obj, highest_d_obj)
            gap = p_obj - highest_d_obj
            if gap < tol:
                break

    return W, active_set


class Solver(BaseSolver):
    """Proximal gradient descent"""

    name = "pgd"
    stop_strategy = "callback"

    def set_objective(self, X, Y, lmbd, n_orient):
        self.X, self.Y = X, Y
        self.lmbd = lmbd
        self.n_orient = n_orient
        self.active_set_size = 10
        self.tol = 1e-8
        self.max_iter = 3000

    def run(self, callback):
        n_features = self.X.shape[1]
        n_times = self.Y.shape[1]

        # Initializing active set
        active_set = np.zeros(n_features, dtype=bool)
        idx_large_corr = np.argsort(groups_norm2(np.dot(self.X.T, self.Y),
                                    self.n_orient))
        new_active_idx = idx_large_corr[-self.active_set_size:]
        if self.n_orient > 1:
            new_active_idx = (
                self.n_orient * new_active_idx[:, None]
                + np.arange(self.n_orient)[None, :]
            ).ravel()

        active_set[new_active_idx] = True
        as_size = np.sum(active_set)

        coef_init = None
        self.W = np.zeros((n_features, n_times))

        while callback(self.W):
            X_as = self.X[:, active_set]
            lipschitz_consts_tmp = norm(X_as, ord=2) ** 2

            coef, as_ = pgd_(X_as, self.Y, lipschitz_consts_tmp, coef_init,
                             self.lmbd, self.n_orient, max_iter=self.max_iter,
                             tol=self.tol)

            active_set[active_set] = as_.copy()
            idx_old_active_set = np.where(active_set)[0]

            self.W = build_full_coefficient_matrix(active_set, n_times, coef)

            R = self.Y - self.X[:, active_set] @ coef
            idx_large_corr = np.argsort(groups_norm2(np.dot(self.X.T, R),
                                        self.n_orient))
            new_active_idx = idx_large_corr[-self.active_set_size:]

            if self.n_orient > 1:
                new_active_idx = (
                    self.n_orient * new_active_idx[:, None]
                    + np.arange(self.n_orient)[None, :]
                )
                new_active_idx = new_active_idx.ravel()

            active_set[new_active_idx] = True
            idx_active_set = np.where(active_set)[0]
            as_size = np.sum(active_set)
            coef_init = np.zeros((as_size, n_times), dtype=coef.dtype)
            idx = np.searchsorted(idx_active_set, idx_old_active_set)
            coef_init[idx] = coef

    def get_result(self):
        return self.W
