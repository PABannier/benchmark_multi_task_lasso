from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import functools
    import numpy as np
    from numpy.linalg import norm


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


def get_lipschitz(G, n_orient):
    if n_orient == 1:
        return np.sum(G * G, axis=0)
    else:
        n_positions = G.shape[1] // n_orient
        lc = np.empty(n_positions)
        for j in range(n_positions):
            G_tmp = G[:, (j * n_orient) : ((j + 1) * n_orient)]
            lc[j] = np.linalg.norm(np.dot(G_tmp.T, G_tmp), ord=2)
        return lc


def sum_squared(X):
    X_flat = X.ravel(order="F" if np.isfortran(X) else "C")
    return np.dot(X_flat, X_flat)


def get_alpha_max(G, M, n_orient=1):
    return norm_l2inf(G.T @ M, n_orient)


def primal(X, Y, coef, active_set, alpha, n_orient=1):
    """Primal objective function for multi-task
    LASSO
    """
    Y_hat = np.dot(X[:, active_set], coef)
    R = Y - Y_hat
    penalty = norm_l21(coef, n_orient, copy=True)
    nR2 = sum_squared(R)
    p_obj = 0.5 * nR2 + alpha * penalty
    return p_obj


@functools.lru_cache(None)
def _get_dgemm():
    return _get_blas_funcs(np.float64, "gemm")


@functools.lru_cache(None)
def _get_blas_funcs(dtype, names):
    from scipy import linalg

    return linalg.get_blas_funcs(names, (np.empty(0, dtype),))


def bcd(X, G, R, one_over_lc, n_orient, alpha_lc, active_set, list_G_j_c):
    X_j_new = np.zeros_like(X[0:n_orient, :], order="C")
    dgemm = _get_dgemm()

    for j, G_j_c in enumerate(list_G_j_c):
        idx = slice(j * n_orient, (j + 1) * n_orient)
        G_j = G[:, idx]
        X_j = X[idx]
        dgemm(
            alpha=one_over_lc[j],
            beta=0.0,
            a=R.T,
            b=G_j,
            c=X_j_new.T,
            overwrite_c=True,
        )
        # X_j_new = G_j.T @ R
        # Mathurin's trick to avoid checking all the entries
        was_non_zero = X_j[0, 0] != 0
        # was_non_zero = np.any(X_j)
        if was_non_zero:
            dgemm(
                alpha=1.0,
                beta=1.0,
                a=X_j.T,
                b=G_j_c.T,
                c=R.T,
                overwrite_c=True,
            )
            # R += np.dot(G_j, X_j)
            X_j_new += X_j
        block_norm = np.sqrt(sum_squared(X_j_new))
        if block_norm <= alpha_lc[j]:
            X_j.fill(0.0)
            active_set[idx] = False
        else:
            shrink = max(1.0 - alpha_lc[j] / block_norm, 0.0)
            X_j_new *= shrink
            dgemm(
                alpha=-1.0,
                beta=1.0,
                a=X_j_new.T,
                b=G_j_c.T,
                c=R.T,
                overwrite_c=True,
            )
            # R -= np.dot(G_j, X_j_new)
            X_j[:] = X_j_new
            active_set[idx] = True


class Solver(BaseSolver):
    """Block coordinate descent with
    low-level BLAS function calls"""

    name = "bcd_blas_low_level"
    stop_strategy = "callback"
    parameters = {"accelerated": (True, False)}

    def _prepare_bcd(self):
        _, n_sources = self.G.shape
        _, n_times = self.M.shape
        n_positions = n_sources // self.n_orient

        active_set = np.zeros(n_sources, dtype=bool)
        self.X = np.zeros((n_sources, n_times))
        self.R = self.M.copy()

        lipschitz_constants = get_lipschitz(self.G, self.n_orient)
        alpha_lc = self.lmbd / lipschitz_constants
        one_over_lc = 1 / lipschitz_constants

        list_G_j_c = []
        for j in range(n_positions):
            idx = slice(j * self.n_orient, (j + 1) * self.n_orient)
            list_G_j_c.append(np.ascontiguousarray(self.G[:, idx]))

        return one_over_lc, alpha_lc, active_set, list_G_j_c

    def set_objective(self, G, M, lmbd, n_orient):
        self.G, self.M = G, M
        self.G = np.asfortranarray(self.G)
        self.lmbd = lmbd
        self.n_orient = n_orient
        self.K = 5

    def run(self, callback):
        one_over_lc, alpha_lc, active_set, list_G_j_c = self._prepare_bcd()

        if self.accelerated:
            n_features, n_times = self.G.shape[1], self.M.shape[1]
            last_K_coef = np.empty((self.K + 1, n_features, n_times))
            U = np.zeros((self.K, n_features * n_times))

        iter_idx = 0

        while callback(self.X):
            bcd(
                self.X,
                self.G,
                self.R,
                one_over_lc,
                self.n_orient,
                alpha_lc,
                active_set,
                list_G_j_c,
            )

            p_obj = primal(
                self.G,
                self.M,
                self.X[active_set],
                active_set,
                self.lmbd,
                self.n_orient,
            )

            if self.accelerated:
                if iter_idx < self.K + 1:
                    last_K_coef[iter_idx] = self.X
                else:
                    for k in range(self.K):
                        last_K_coef[k] = last_K_coef[k + 1]
                    last_K_coef[self.K - 1] = self.X

                    for k in range(self.K):
                        U[k] = (
                            last_K_coef[k + 1].ravel() - last_K_coef[k].ravel()
                        )
                    C = np.dot(U, U.T)

                    try:
                        z = np.linalg.solve(C, np.ones(self.K))
                        c = z / z.sum()
                        X_acc = np.sum(
                            last_K_coef[:-1] * c[:, None, None], axis=0
                        )
                        active_set_acc = norm(X_acc, axis=1) != 0

                        p_obj_acc = primal(
                            self.G,
                            self.M,
                            X_acc[active_set_acc],
                            active_set_acc,
                            self.lmbd,
                            self.n_orient,
                        )

                        if p_obj_acc < p_obj:
                            print("IT WORKS")
                            self.X = X_acc
                            active_set = active_set_acc
                            self.R = (
                                self.M
                                - self.G[:, active_set] @ self.X[active_set]
                            )

                    except np.linalg.LinAlgError:
                        print("LinAlgError")

            iter_idx += 1

        # XR = self.G.T @ (self.M - self.G @ self.X)
        # assert norm_l2inf(XR, self.n_orient) <= self.lmbd + 1e-12, "KKT check"

    def get_result(self):
        return self.X
