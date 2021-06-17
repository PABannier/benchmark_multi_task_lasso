import numpy as np
from numpy.linalg import norm
from benchopt import BaseDataset


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples, n_features': [
            (1000, 500),
            (5000, 200)]
    }

    def __init__(self, n_samples=10, n_features=50, random_state=27):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state

    def get_data(self):

        rng = np.random.RandomState(self.random_state)
        X = rng.randn(self.n_samples, self.n_features)
        y = rng.randn(self.n_samples)

        data = dict(X=X, y=y)

        return self.n_features, data


class MultiTaskDataset(BaseDataset):
    name = "Multi task Simulated"

    parameters = {
        'n_samples, n_features, n_tasks': [
            (100, 500, 50)
        ]
    }

    def __init__(self, n_samples=10, n_features=50, n_tasks=30, support=2, 
                 snr=2, random_state=0):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_tasks = n_tasks
        self.support = support
        self.snr = snr
        self.random_state = random_state

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        G = rng.randn(self.n_samples, self.n_features)
        
        X = np.zeros((self.n_features, self.n_tasks))
        for k in self.support:
            X[k, :] = rng.normal(size=(self.n_tasks))
    
        M = G @ X

        noise = rng.randn(self.n_samples, self.n_tasks)
        sigma = 1 / norm(noise) * norm(M) / self.snr

        M += sigma * noise

        data = dict(G=G, M=M, X=X)

        return self.n_features, self.n_tasks, data
