from benchopt import BaseObjective
from .utils import norm_l21


class MTLObjective(BaseObjective):
    name = "Multi Task Lasso"

    parameters = {"reg": [0.3, 0.5, 0.7]}

    def __init__(self, reg=0.1):
        self.reg = reg

    def set_data(self, G, M):
        self.G, self.M = G, M

    def compute(self, X):
        diff = self.M - self.G @ X
        return 0.5 * diff @ diff + self.reg * norm_l21(X)

    def to_dict(self):
        return dict(G=self.G, M=self.M, reg=self.reg)
