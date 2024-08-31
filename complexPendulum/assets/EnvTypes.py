from enum import IntEnum


class ActionType(IntEnum):
    """ActionType Enum showing the action of the environment."""
    DIRECT = 1
    GAIN = 2


class RewardType(IntEnum):
    """RewardType Enum showing the used Reward function of the environment."""
    LQ = 1
    EXP = 2
    LIN = 3


class ActionTypeError(Exception):
    """Exception raised when the actiontype of env and agent does not match."""

    def __init__(self, args: list, value: list, message: str):
        """Initialization of Exception."""

        msg: str = ""
        for i, arg in enumerate(args):
            msg += ' ' + str(arg) + '(' + str(value[i]) + '), '

        msg = msg[0: -1]
        msg += '. ' + message
        super().__init__(msg)
