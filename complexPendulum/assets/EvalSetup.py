import numpy as np
from enum import IntEnum
from complexPendulum.assets import RewardType


class EvaluationDataType(IntEnum):
    """EvaluationDataType is an enum showing which part of the data should be evaluated."""
    COMPLETE = 1
    SWING_UP = 2
    STEP = 3


class EvalSetup:
    """Describes the evaluation setup."""
    def __init__(self, func: RewardType, name: str,
                 Q: np.array = None, R: np.array = None,
                 k: float = None) -> None:
        """Initialization.
        Input:
            func: RewardType
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
