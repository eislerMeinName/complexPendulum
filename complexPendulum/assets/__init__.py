import numpy as np

from complexPendulum.assets.EvalSetup import EvalSetup, SetupType, EvaluationDataType
from complexPendulum.assets.EnvTypes import ActionType, RewardType, ActionTypeError
from complexPendulum.assets.Logger import Logger
from complexPendulum.assets.Evaluator import Evaluator

Setup1 = EvalSetup(SetupType.LQR, "LQR 1", np.eye(4), np.eye(1), 200)
Setup2 = EvalSetup(SetupType.LQR, "LQR 2",
                   np.eye(4), 0.5*np.eye(1), k=200)
Setup3 = EvalSetup(SetupType.LQR, "LQR 3",
                   np.array([[1, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]]),
                   np.eye(1), k=200)
Setup4 = EvalSetup(SetupType.EXP, "EXP 1", np.eye(4), np.eye(1))
Setup5 = EvalSetup(SetupType.EXP, "EXP 2", np.eye(4), 0.5*np.eye(1))
Setup6 = EvalSetup(SetupType.EXP, "EXP 3",
                   np.array([[1, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]]),
                   np.eye(1))
Setup7 = EvalSetup(SetupType.LIN, "LIN")
