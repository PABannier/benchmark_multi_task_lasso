from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from numpy.linalg import norm
    from mtl_utils.common import norm_l21, get_alpha_max


class Objective(BaseObjective):
    name = "Objective"
    parameters = {
        "reg": [1, 0.5, 0.1, 0.01],
        "n_orient": [1]
    }

    def __init__(self, reg=0.1, n_orient=1):
        self.reg = reg
        self.n_orient = n_orient

    def set_data(self, X, Y):
        self.X, self.Y = X, Y
        self.alpha_max = get_alpha_max(self.X, self.Y, self.n_orient)
        self.lmbd = self.reg * self.alpha_max

    def compute(self, W):
        R = self.Y - self.X @ W
        p_obj = 0.5 * norm(R, ord="fro") ** 2 + self.lmbd * norm_l21(
            W, self.n_orient
        )
        nnz = (norm(W.reshape(W.shape[0] // self.n_orient, -1), axis=1) != 0
               ).sum()
        return dict(value=p_obj, sparsity=nnz)

    def to_dict(self):
        return dict(X=self.X, Y=self.Y, lmbd=self.lmbd, n_orient=self.n_orient)
