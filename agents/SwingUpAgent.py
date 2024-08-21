import numpy as np


class SwingUpAgent:

    def __init__(self, ksu: float = 1.3, kcw: float = 3,
                 kvw: float = 3, kem: float = 7,
                 eta: float = 1.05, E0: float = 0.05,
                 xmax: float = 0.4, vmax: float = 5) -> None:
        self.ksu = ksu
        self.kcw = kcw
        self.kvw = kvw
        self.kem = kem
        self.eta = eta
        self.E0 = E0
        self.xmax = xmax
        self.vmax = vmax

    def sample(self, state: np.array) -> None:
        pass