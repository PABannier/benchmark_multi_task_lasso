from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    # from mtl_utils.common import get_alpha_max, get_lipschitz, groups_norm2
    from mtl_utils.common import get_lipschitz
    import numpy as np
    from numpy.linalg import norm
    from numba import njit
    from mtl_utils.common import sum_squared, _get_dgemm, norm_l21


if import_ctx.failed_import:

    def njit(f):  # noqa: F811
        return f


def dual_mtl(alpha, norm_Y2, Theta, Y):
    """
    Problem solved:
    max - 0.5 * (lambda ** 2) * ||(Y/lambda) - Theta|| ** 2 + 0.5 * ||Y|| ** 2
    """
    d_obj = - ((Y / alpha - Theta) ** 2).sum()
    d_obj *= 0.5 * alpha ** 2
    d_obj += norm_Y2 / 2.
    return d_obj


def primal_mtl(W, alpha, R, n_orient):
    """
    Problem solved:
    min 0.5 * ||Y - XB|| ** 2 + lambda * ||B||_{2, 1}
    """
    p_obj = sum_squared(R) / 2
    p_obj += norm_l21(W, n_orient, copy=False) * alpha
    return p_obj


@njit
def set_prios_mtl(X, W, norms_X_block, prios, screened, radius,
                  n_screened, norm_XT_Theta, n_orient):
    n_features = X.shape[1]
    n_positions = n_features // n_orient
    for j in range(n_positions):
        if screened[j] or norms_X_block[j] == 0:
            prios[j] = np.inf
            continue
        nrm = norm_XT_Theta[j]
        prios[j] = (1. - nrm) / norms_X_block[j]
        if prios[j] > radius:
            idx = slice(j * n_orient, (j + 1) * n_orient)
            if np.any(W[idx, :]):
                screened[j] = True
                n_screened += 1
    return n_screened


@njit
def dual_scaling_mtl(Theta, Xs_j, C, skip):
    n_positions = len(Xs_j)
    norm_XT_Theta = np.zeros(n_positions)
    nrm = 0
    for j in C:
        if skip[j]:
            continue
        Xj_theta_nrm = norm(np.dot(Xs_j[j], Theta))
        norm_XT_Theta[j] = Xj_theta_nrm
        if Xj_theta_nrm > nrm:
            nrm = Xj_theta_nrm
    return nrm, norm_XT_Theta


# def dual_scaling_mtl(Theta, Xs_j, C, skip):
#     n_positions, n_orient, _ = Xs_j.shape
#     norm_XT_Theta = np.zeros(n_positions)
#     # dgemm = _get_dgemm()
#     n_tasks = Theta.shape[1]
#     Xj_theta = np.zeros((n_orient, n_tasks))
#     nrm = 0
#     for j in C:
#         if skip[j]:
#             continue
#         # dgemm(alpha=1.0, beta=1.0, a=Theta.T, b=Xs_j[j], c=Xj_theta.T,
#         #      overwrite_c=True)
#         np.dot(Xs_j[j], Theta, out=Xj_theta)
#         Xj_theta_nrm = norm(Xj_theta, ord='fro')
#         norm_XT_Theta[j] = Xj_theta_nrm
#         if Xj_theta_nrm > nrm:
#             nrm = Xj_theta_nrm
#     return nrm, norm_XT_Theta


@njit
def create_ws_mtl(prune, W, prios, p0, t, screened, C, n_screened, ws_size,
                  n_orient):
    n_features = W.shape[0]
    n_positions = n_features // n_orient

    if t == 0:
        ws_size = p0
        for j in range(n_positions):
            idx = slice(j * n_orient, (j + 1) * n_orient)
            if np.any(W[idx, :]):
                prios[j] = -1
    else:
        nnz = 0
        if prune:
            for j in range(n_positions):
                idx = slice(j * n_orient, (j + 1) * n_orient)
                if np.any(W[idx, 0]):
                    prios[j] = -1
                    nnz += 1
            ws_size = 2 * nnz
        else:
            for k in range(ws_size):
                if not screened[C[k]]:
                    prios[C[k]] = -1
            ws_size = 2 * ws_size
    ws_size = min(n_positions - n_screened, ws_size)
    return ws_size


@njit
def create_accel_pt(epoch, gap_freq, alpha, R, out, last_K_R, U, UtU,
                    verbose):
    K = U.shape[0] + 1

    if epoch // gap_freq < K:
        last_K_R[(epoch // gap_freq), :] = R
    else:
        for k in range(K - 1):
            last_K_R[k, :] = last_K_R[k + 1, :]
        last_K_R[K - 1, :] = R
        for k in range(K - 1):
            U[k] = last_K_R[k + 1].ravel() - last_K_R[k].ravel()

        # double for loop but small: K**2/2
        for k in range(K - 1):
            for j in range(k, K - 1):
                UtU[k, j] = np.dot(U[k], U[j])
                UtU[j, k] = UtU[k, j]

        try:
            anderson = np.linalg.solve(UtU, np.ones(UtU.shape[0]))
        except Exception:
            if verbose:
                print("Singular matrix when computing accelerated point.")
        else:
            anderson /= np.sum(anderson)

            out[:] = 0
            for k in range(K - 1):
                out += anderson[k] * last_K_R[k, :]
            out /= alpha
            # out now holds the extrapolated dual point


def bcd_epoch(C, lipschitz_consts, X, R, alpha, W, n_orient):
    n_tasks = R.shape[1]
    W_j_new = np.zeros((n_orient, n_tasks))
    dgemm = _get_dgemm()
    alpha_lc = alpha / lipschitz_consts

    for j in C:
        if lipschitz_consts[j] == 0.:
            continue
        idx = slice(j * n_orient, (j + 1) * n_orient)
        W_j = W[idx]
        X_j = X[:, idx]

        # W_j_new = X_j.T @ R * inv_lc[j]
        dgemm(alpha=1/lipschitz_consts[j], beta=0.0, a=R.T, b=X_j, c=W_j_new.T,
              overwrite_c=True)

        if np.any(W_j[0, :]):
            # R += np.dot(X_j, W_j)
            dgemm(alpha=1.0, beta=1.0, a=W_j.T, b=X_j.T, c=R.T,
                  overwrite_c=True)
            W_j_new += W_j

        block_norm = norm(W_j_new)

        if block_norm <= alpha_lc[j]:
            W_j.fill(0.0)
        else:
            shrink = 1.0 - alpha_lc[j] / block_norm
            W_j_new *= shrink
            # R -= np.dot(X_j, W_j_new)
            dgemm(alpha=-1.0, beta=1.0, a=W_j_new.T, b=X_j.T, c=R.T,
                  overwrite_c=True)
            W_j[:] = W_j_new


def celer_dual_mtl(X, Y, alpha, n_iter, max_epochs=10_000, gap_freq=10,
                   tol=1e-12, p0=10, verbose=0, prune=True, n_orient=1):
    n_samples, n_features = X.shape
    n_tasks = Y.shape[1]
    n_obs = n_samples * n_tasks
    n_positions = n_features // n_orient

    W = np.zeros((n_features, n_tasks))
    R = Y.copy()
    verbose_in = max(verbose - 1, 0)
    n_screened = 0
    norm_Y2 = norm(Y, 'fro') ** 2

    tol *= norm_Y2
    if p0 > n_positions:
        p0 = n_positions

    prios = np.zeros(n_positions, dtype=X.dtype)
    screened = np.zeros(n_positions, dtype=np.int32)
    notin_WS = np.zeros(n_positions, dtype=np.int32)

    # acceleration variables
    K = 6
    last_K_R = np.empty((K, n_samples, n_tasks), dtype=X.dtype)
    U = np.empty((K - 1, n_obs), dtype=X.dtype)
    UtU = np.empty((K - 1, K - 1), dtype=X.dtype)

    # norms_X_block = norm(X, axis=0) => norm of each column
    # inv_lc = 1 / norms_X_block ** 2
    lipschitz_consts = get_lipschitz(X, n_orient)
    norms_X_block = np.sqrt(lipschitz_consts)  # TODO: CHECK! (TBD)

    gaps = np.zeros(n_iter, dtype=X.dtype)

    Theta = np.zeros((n_samples, n_tasks), dtype=X.dtype)
    Theta_in = np.zeros((n_samples, n_tasks), dtype=X.dtype)
    Thetacc = np.zeros((n_samples, n_tasks), dtype=X.dtype)

    Xs_j = np.empty((n_positions, n_orient, n_samples))
    for i in range(n_positions):
        idx = slice(i * n_orient, (i + 1) * n_orient)
        Xs_j[i] = X[:, idx].T

    # d_obj_from_inner = 0.
    all_positions = np.arange(n_positions)
    C = all_positions.copy()
    ws_size = p0

    for t in range(n_iter):
        Theta[:] = R / alpha
        scal, norm_XT_Theta = dual_scaling_mtl(Theta, Xs_j, all_positions,
                                               screened)

        if scal > 1.:
            Theta /= scal
            norm_XT_Theta /= scal
        d_obj = dual_mtl(alpha, norm_Y2, Theta, Y)

        # if t > 0:
        #     scal, norm_XT_Theta_in = dual_scaling_mtl(Theta_in, X,
        #                                               all_positions,
        #                                               screened,
        #                                               n_orient)
        #     if scal > 1.:
        #         Theta_in /= scal
        #         norm_XT_Theta_in /= scal
        # d_obj_from_inner = dual_mtl(alpha, norm_Y2, Theta_in, Y)
        # if d_obj_from_inner > d_obj:
        #     d_obj = d_obj_from_inner
        #     Theta[:] = Theta_in
        #     norm_XT_Theta[:] = norm_XT_Theta_in

        highest_d_obj = d_obj

        p_obj = primal_mtl(W, alpha, R, n_orient)
        gap = p_obj - highest_d_obj
        gaps[t] = gap

        if verbose:
            print("Iter {:d}: primal {:.10f}, gap {:.2e}".format(
                  t, p_obj, gap), end="")

        if gap <= tol:
            if verbose:
                print("\nEarly exit, gap: {:.2e} < {:.2e}".format(gap, tol))
            break

        radius = np.sqrt(2 * gap) / alpha

        n_screened = set_prios_mtl(X, W, norms_X_block, prios, screened,
                                   radius, n_screened, norm_XT_Theta, n_orient)
        ws_size = create_ws_mtl(prune, W, prios, p0, t, screened, C,
                                n_screened, ws_size, n_orient)
        # if ws_size == n_features then argpartition will break
        if ws_size == n_positions:
            C = all_positions
        else:
            C = np.argpartition(np.asarray(prios), ws_size)[
                :ws_size].astype(np.int32)

        notin_WS.fill(1)
        notin_WS[C] = 0

        tol_in = 0.3 * gap if prune else tol

        if verbose:
            print(", {:d} feats in subpb ({:d} left)".format(
                  len(C), n_positions - n_screened))

        highest_d_obj_in = 0
        for epoch in range(max_epochs):
            if epoch > 0 and epoch % gap_freq == 0:
                Theta_in[:] = R / alpha

                scal = dual_scaling_mtl(Theta_in, Xs_j, C, screened)[0]
                if scal > 1.:
                    Theta_in /= scal

                d_obj_in = dual_mtl(alpha, norm_Y2, Theta_in, Y)

                if True:
                    create_accel_pt(epoch, gap_freq, alpha, R, Thetacc,
                                    last_K_R, U, UtU, verbose_in)

                    if epoch // gap_freq >= K:
                        scal = dual_scaling_mtl(Thetacc, Xs_j, C, screened)[0]

                        if scal > 1.:
                            Thetacc /= scal

                        d_obj_accel = dual_mtl(alpha, norm_Y2, Thetacc, Y)
                        if d_obj_accel > d_obj_in:
                            d_obj_in = d_obj_accel
                            Theta_in[:] = Thetacc

                highest_d_obj_in = max(highest_d_obj_in, d_obj_in)

                p_obj_in = primal_mtl(W, alpha, R, n_orient)
                gap_in = p_obj_in - highest_d_obj_in

                if verbose_in:
                    print("Epoch {:d}, primal {:.10f}, gap: {:.2e}".format(
                        epoch, p_obj_in, gap_in))
                if gap_in < tol_in:
                    if verbose_in:
                        print("Exit epoch {:d}, gap: {:.2e} < {:.2e}".format(
                            epoch, gap_in, tol_in))
                    break

            bcd_epoch(C, lipschitz_consts, X, R, alpha, W, n_orient)
        else:
            print("!!! Inner solver did not converge at epoch "
                  "{:d}, gap: {:.2e} > {:.2e}".format(epoch, gap_in, tol_in))
    return W, Theta, R


class Solver(BaseSolver):
    name = "celer_python"
    stop_strategy = "iteration"

    def set_objective(self, X, Y, lmbd, n_orient):
        self.Y, self.lmbd = Y, lmbd
        self.X = np.asfortranarray(X)
        self.n_orient = n_orient

        # Make sure we cache the Numba compilation
        self.run(1)

    def run(self, n_iter):
        W = celer_dual_mtl(self.X, self.Y, self.lmbd, n_iter,
                           max_epochs=100_000, prune=True, verbose=0,
                           n_orient=self.n_orient, tol=1e-8)[0]
        self.W = W

    def get_next(n_iter):
        return n_iter + 1

    def get_result(self):
        return self.W
