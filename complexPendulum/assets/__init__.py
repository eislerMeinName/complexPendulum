import numpy as np

from complexPendulum.assets.EnvTypes import ActionType, RewardType, ActionTypeError
from complexPendulum.assets.EvalSetup import EvalSetup, EvaluationDataType
from complexPendulum.assets.Logger import Logger
from complexPendulum.assets.colors import bcolors
from complexPendulum.assets.Evaluator import Evaluator

Setup1 = EvalSetup(RewardType.LQ, "LQR 1", np.eye(4)/100, 0.1*np.eye(1)/100)
Setup2 = EvalSetup(RewardType.LQ, "LQR 2",
                   np.array([[1, 0, 0, 0],
                             [0, 0.1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 0.1]]),
                   0.1*np.eye(1))
Setup3 = EvalSetup(RewardType.LQ, "LQR 3",
                   np.array([[0.1, 0, 0, 0],
                             [0, 0.01, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 0.01]]),
                   0.01*np.eye(1))
Setup4 = EvalSetup(RewardType.EXP, "EXP 1", 0.01*np.eye(4), 0.001*np.eye(1))
Setup5 = EvalSetup(RewardType.EXP, "EXP 2",
                   np.array([[0.6, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0.6, 0],
                             [0, 0, 0, 0]]),
                   np.zeros(1))
Setup6 = EvalSetup(RewardType.LIN, "LIN",
                   np.array([[0.01, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0.1, 0],
                             [0, 0, 0, 0]]),
                   np.zeros(1))
