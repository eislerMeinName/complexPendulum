import numpy as np
from enum import IntEnum


class EvaluationDataType(IntEnum):
    """EvaluationDataType is an enum showing which part of the data should be evaluated."""
    COMPLETE = 1
    SWING_UP = 2
    STEP = 3


class SetupType(IntEnum):
    """SetupType is an enum showing which reward function is used for the evaluation setup."""
    LQR = 1
    EXP = 2
    LIN = 3


class EvalSetup:
    """Describes the evaluation setup."""
    def __init__(self, func: SetupType, name: str,
                 Q: np.array = None, R: np.array = None,
                 k: float = None) -> None:
        """Initialization.
        Input:
            func: SetupType
                The reward function.
            name: str
                The name.
            Q: np.array
                The Q matrix.
            R: np.array
                The R matrix.
            k: float
                The constant normalization factor.
        """

        self.func = func
        self.name = name
        self.Q = Q
        self.R = R
        self.k = k
