from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm


class Dataset(BaseDataset):
    name = "Simulated"

    parameters = {
        "n_samples, n_features, n_tasks": [
            (10, 300, 30),
        ]
    }

    def __init__(
        self,
        n_samples=10,
        n_features=50,
        n_tasks=30,
        nnz=2,
        snr=2,
        random_state=0,
    ):
        self.n_samples, self.n_features = n_samples, n_features
        self.n_tasks = n_tasks
        self.nnz, self.snr = nnz, snr
        self.random_state = random_state

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        X = rng.randn(self.n_samples, self.n_features)

        support = rng.choice(self.nnz, size=self.n_features)
        W = np.zeros((self.n_features, self.n_tasks))
        for k in support:
            W[k, :] = rng.normal(size=(self.n_tasks))
        Y = X @ W

        noise = rng.randn(self.n_samples, self.n_tasks)
        sigma = 1 / norm(noise) * norm(Y) / self.snr

        Y += sigma * noise

        data = dict(X=X, Y=Y)

        return X.size, data
