import numpy as np
from numpy.linalg import norm
from numba import njit
from mtl_utils.common import (norm_l2inf, sum_squared, _get_dgemm,
                              get_lipschitz, get_duality_gap, groups_norm2)


def build_full_coefficient_matrix(active_set, n_times, coef):
    """Building full coefficient matrix and filling active set with
    non-zero coefficients"""
    final_coef_ = np.zeros((len(active_set), n_times))
    if coef is not None:
        final_coef_[active_set] = coef
    W = final_coef_
    return W


def bcd_(X, Y, lipschitz, init, _alpha, n_orient, accelerated, K=5,
         max_iter=2000, tol=1e-5):
    n_times = Y.shape[1]
    n_features = X.shape[1]
    n_positions = n_features // n_orient

    if init is None:
        coef = np.zeros((n_features, n_times))
        R = Y.copy()
    else:
        coef = init
        R = Y - X @ coef

    X = X if np.isfortran(X) else np.asfortranarray(X)

    if accelerated:
        last_K_coef = np.empty((K + 1, n_features, n_times))
        U = np.zeros((K, n_features * n_times))

    highest_d_obj = -np.inf
    active_set = np.zeros(n_features, dtype=bool)

    for iter_idx in range(max_iter):
        coef_j_new = np.zeros_like(coef[:n_orient, :], order="C")
        dgemm = _get_dgemm()

        for j in range(n_positions):
            idx = slice(j * n_orient, (j + 1) * n_orient)
            coef_j = coef[idx]
            X_j = X[:, idx]

            dgemm(alpha=1 / lipschitz[j], beta=0.0, a=R.T, b=X_j,
                  c=coef_j_new.T, overwrite_c=True)

            if coef_j[0, 0] != 0:
                dgemm(alpha=1.0, beta=1.0, a=coef_j.T, b=X_j.T, c=R.T,
                      overwrite_c=True)
                coef_j_new += coef_j

            block_norm = np.sqrt(sum_squared(coef_j_new))
            alpha_lc = _alpha / lipschitz[j]

            if block_norm <= alpha_lc:
                coef_j.fill(0.0)
                active_set[idx] = False
            else:
                shrink = max(1.0 - alpha_lc / block_norm, 0.0)
                coef_j_new *= shrink

                dgemm(alpha=-1.0, beta=1.0, a=coef_j_new.T, b=X_j.T, c=R.T,
                      overwrite_c=True)
                coef_j[:] = coef_j_new
                active_set[idx] = True

        _, p_obj, d_obj = get_duality_gap(X, Y, coef[active_set], active_set,
                                          _alpha, n_orient)
        highest_d_obj = max(d_obj, highest_d_obj)
        gap = p_obj - highest_d_obj

        if gap < tol:
            break

        if accelerated:
            last_K_coef[iter_idx % (K + 1)] = coef

            if iter_idx % (K + 1) == K:
                for k in range(K):
                    U[k] = last_K_coef[k + 1].ravel() - last_K_coef[k].ravel()

                C = U @ U.T

                try:
                    z = np.linalg.solve(C, np.ones(K))
                    c = z / z.sum()

                    coef_acc = np.sum(
                        last_K_coef[:-1] * c[:, None, None], axis=0
                    )
                    active_set_acc = norm(coef_acc, axis=1) != 0

                    p_obj_acc = get_duality_gap(X, Y, coef_acc[active_set_acc],
                                                active_set_acc, _alpha,
                                                n_orient, primal_only=True)

                    if p_obj_acc < p_obj:
                        coef = coef_acc
                        active_set = active_set_acc
                        R = Y - X[:, active_set] @ coef[active_set]

                except np.linalg.LinAlgError:
                    print("LinAlg Error")

    coef = coef[active_set]
    return coef, active_set


def solver_scipy(X, Y, lmbd, active_set_size=10, tol=1e-8, max_iter=3000,
                 accelerated=True):
    n_features = X.shape[1]
    n_times = Y.shape[1]

    lipschitz_consts = get_lipschitz(X, 1)

    # Initializing active set
    active_set = np.zeros(n_features, dtype=bool)
    idx_large_corr = np.argsort(groups_norm2(np.dot(X.T, Y), 1))
    new_active_idx = idx_large_corr[-active_set_size:]

    active_set[new_active_idx] = True
    as_size = np.sum(active_set)

    coef_init = None
    W = np.zeros((n_features, n_times))
    iter_idx = 0

    highest_d_obj = -np.inf

    for _ in range(max_iter):
        lipschitz_consts_tmp = lipschitz_consts[active_set[::1]]

        coef, as_ = bcd_(X[:, active_set], Y, lipschitz_consts_tmp, coef_init,
                         lmbd, 1, accelerated, max_iter=max_iter, tol=tol)

        active_set[active_set] = as_.copy()
        idx_old_active_set = np.where(active_set)[0]

        _, p_obj, d_obj = get_duality_gap(X, Y, coef, active_set, lmbd, 1)
        highest_d_obj = max(highest_d_obj, d_obj)
        gap = p_obj - highest_d_obj
        print("Iteration %d :: p_obj %f :: dgap %f ::"
              "n_active_start %d :: n_active_end %d" % (
              k + 1, p_obj, gap, as_size,np.sum(active_set)))
        if gap < tol:
            break

        W = build_full_coefficient_matrix(active_set, n_times, coef)

        if iter_idx < (max_iter - 1):
            R = Y -  X[:, active_set] @ coef
            idx_large_corr = np.argsort(groups_norm2(np.dot( X.T, R), 1))
            new_active_idx = idx_large_corr[active_set_size:]

            active_set[new_active_idx] = True
            idx_active_set = np.where(active_set)[0]
            as_size = np.sum(active_set)
            coef_init = np.zeros((as_size, n_times), dtype=coef.dtype)
            idx = np.searchsorted(idx_active_set, idx_old_active_set)
            coef_init[idx] = coef
        
        iter_idx += 1
    return W


def compute_lipschitz(X):
    return np.sum(X * X, axis=0)

@njit
def block_soft_thresh(x, u):
    norm_x = norm(x)
    if norm_x < u:
        return np.zeros_like(0)
    else:
        return (1 - u / norm_x) * x

def bcd_toy(X, Y, alpha, n_iter=2000, tol=1e-8, dgap_freq=10):
    n_samples, n_features = X.shape
    n_tasks = Y.shape[1]

    W = np.zeros((n_features, n_tasks), dtype=X.dtype)
    R = Y.copy()

    lipschitz = compute_lipschitz(X)
    alpha_lc = alpha / lipschitz
    step_size = n_samples / lipschitz

    dgemm = _get_dgemm()

    for i in range(n_iter):
        for j in range(n_features):
            idx = slice(j, j + 1)
            W_j = W[idx]
            X_j = X[:, idx]
            #BCD update
            grad_j = np.dot(X_j.T, X @ W - Y) / n_samples
            W_j = block_soft_thresh(W_j - step_size[j] * grad_j, alpha_lc[j])            

        # Get duality gap
        # breaker if tol

if __name__ == "__main__":

    RANDOM_STATE = 30
    N_SAMPLES = 100
    N_FEATURES = 500
    N_TASKS = 30
    NNZ = 10

    SNR = 3

    N_ITER = 2000

    rng = np.random.RandomState(RANDOM_STATE)
    X = rng.randn(N_SAMPLES, N_FEATURES)

    support = rng.choice(NNZ, size=N_FEATURES)
    W = np.zeros((N_FEATURES, N_TASKS))
    for k in support:
        W[k, :] = rng.normal(size=(N_TASKS))
    Y = X @ W

    noise = rng.randn(N_SAMPLES, N_TASKS)
    sigma = 1 / norm(noise) * norm(Y) / SNR

    Y += sigma * noise

    alpha_max = norm_l2inf(X.T @ Y) / len(Y)
    alpha = alpha_max * 0.1

    # BCD

