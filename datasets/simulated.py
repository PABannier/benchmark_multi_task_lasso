from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from benchopt.datasets import make_correlated_data
    import numpy as np
    from scipy.sparse import csc_matrix


class Dataset(BaseDataset):
    name = "Simulated"

    parameters = {
        "n_samples, n_features, n_tasks": [
            (10, 300, 30), # we win
            (10, 500, 30),
            (10, 500, 80),
            (50, 500, 80),  # we loose,
        ],
        "sparsity": ["sparse", "dense"]
    }

    def __init__(self, n_samples=10, n_features=50, n_tasks=30, rho=0.3,
                 sparsity="dense", snr=3, density=0.3, random_state=0):
        self.n_samples, self.n_features = n_samples, n_features
        self.n_tasks = n_tasks
        self.density, self.snr = density, snr
        self.random_state = random_state
        self.rho = rho
        self.sparsity = sparsity

    def get_data(self):
        X, Y, _ = make_correlated_data(
            n_samples=self.n_samples, n_features=self.n_features,
            n_tasks=self.n_tasks, rho=self.rho, snr=self.snr,
            density=self.density, random_state=self.random_state)
        
        if self.sparsity == "sparse":
            X = csc_matrix(X * np.random.binomial(1, 0.1, size=X.shape))

        data = dict(X=X, Y=Y)

        return (X.shape[1], Y.shape[1] if Y.ndim > 1 else 1), data
