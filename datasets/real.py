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
        try:
            X = np.load(g_path)
            Y = np.load(m_path)
        except IOError:
            X = np.zeros((3, 4))
            Y = np.ones((3, 5))
            # TODO better data downloading

        data = dict(X=X, Y=Y)

        return (X.shape[1], Y.shape[1]), data
