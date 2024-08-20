from enum import IntEnum


class ActionType(IntEnum):
    """ActionType Enum showing the action of the environment."""
    DIRECT = 1
    GAIN = 2


class RewardType(IntEnum):
    """RewardType Enum showing the used Reward function of the environment."""
    LQ = 1
    EXP = 2
