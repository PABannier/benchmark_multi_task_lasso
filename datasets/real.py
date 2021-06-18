from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import os
    import numpy as np
    from numpy.linalg import norm


class Dataset(BaseDataset):
    name = "Real"

    def __init__(self):
        pass

    def get_data(self):
        cwd_path = os.getcwd()
        g_path = os.path.join(cwd_path, "benchmark_bcd", "data", "G.npy")
        m_path = os.path.join(cwd_path, "benchmark_bcd", "data", "M.npy")
        G = np.load(g_path)
        M = np.load(m_path)

        data = dict(G=G, M=M)
        size = G.shape[1] * M.shape[1]

        return size, data
