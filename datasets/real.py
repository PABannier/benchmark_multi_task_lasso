from pathlib import Path
from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):
    name = "Real"

    def __init__(self):
        pass

    def get_data(self):
        data_dir = Path(__file__).parent / "data"
        g_path = data_dir / "G.npy"
        m_path = data_dir / "M.npy"
        G = np.load(g_path)
        M = np.load(m_path)

        # M = M[:, M.max(axis=0).argmax()][:, np.newaxis]

        data = dict(G=G, M=M)
        size = G.shape[1] * M.shape[1]

        return size, data
