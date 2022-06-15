import numpy as np
from numpy.linalg import norm

from flashcd.estimators import MultiTaskLasso as FCD_MTL
from celer import MultiTaskLasso as C_MTL

from mtl_utils.common import get_alpha_max, sum_squared

random_state = 0
n_samples = 1000
n_features = 3000
n_tasks = 80
corr = 0.3
nnz = 2
snr = 2

# Generate toy data

rng = np.random.RandomState(random_state)
sigma = np.sqrt(1 - corr ** 2)
U = rng.randn(n_samples)

X = np.empty([n_samples, n_features], order="F")
X[:, 0] = U
for j in range(1, n_features):
    U *= corr
    U += sigma * rng.randn(n_samples)
    X[:, j] = U

support = rng.choice(n_features, nnz, replace=False)
W = np.zeros((n_features, n_tasks))

for k in support:
    W[k, :] = rng.normal(size=(n_tasks))

Y = X @ W

noise = rng.randn(n_samples, n_tasks)
sigma = 1 / norm(noise) * norm(Y) / snr

Y += sigma * noise

# Running both solver
alpha_max = get_alpha_max(X, Y)
alpha = alpha_max * 0.5

tol = 1e-8 / sum_squared(Y)

celer_clf = C_MTL(alpha / X.shape[0], tol=tol, fit_intercept=False, verbose=2)
celer_clf.fit(X, Y)

print("\n")
print("#" * 20)
print("\n")

fcd_clf = FCD_MTL(alpha / X.shape[0], tol=tol, fit_intercept=False, verbose=2)
fcd_clf.fit(X, Y)
