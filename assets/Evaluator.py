import numpy as np


class Evaluator:

    def __init__(self, logpath: str, evalpath: str) -> None:
        self.logpath = logpath
        # load logged data
        self.evalpath = evalpath

    def evalLQR(self, Q: np.array, R: np.array) -> float:
        pass

    def evalEXP(self, Q: np.array, R: np.array) -> float:
        pass

    def evalLIN(self) -> float:
        pass

    def write(self) -> None:
        pass

    def eval(self) -> None:
        pass
