from typing import Tuple
import pandas as pd
from complexPendulum.envs.PendulumEnv import ComplexPendulum
from complexPendulum.assets import RewardType, ActionType
from complexPendulum.plots.s0 import calcS0
import numpy as np


class DirectUpPendulum(ComplexPendulum):
    """Gym environment that models Direct PWM control of the pendulum with matching preprocessing."""

    def __init__(self,
                 frequency: float = 100,
                 episode_len: float = 15,
                 path: str = 'params.xml',
                 Q: np.ndarray = np.eye(4),
                 R: np.ndarray = np.eye(1),
                 gui: bool = False,
                 rewardtype: RewardType = RewardType.LQ,
                 conditionReward: bool = False,
                 s0: np.array = None,
                 friction: bool = True,
                 log: bool = True,
                 render_mode: str = "rgb_array"
                 ) -> None:

        """
        Initialization.
        Input:
            frequency: float
                The control frequency.
            episode_len: float
                The length of the episode in seconds.
            path: str
                The path to the parameter file.
            Q: np.ndarray
                The Q array for the reward function.
            R: np.ndarray
                The R array for the reward function.
            gui: bool = False
                Render the gui.
            rewardtype: RewardType = RewardType.LQ
                The type of the reward function.
            conditionReward: bool = False
                Use conditioned reward function.
            s0: np.array = None
                The starting state.
            friction: bool = True
                Use friction during simulation.
            log: bool = True
                Use logger.
            render_mode: str = "rgb_array"
                The render mode.
        """

        self.vals = [(0.0811656494140625, 0.007813444394251738), 
                     (0.28743896484375026, 0.01748701857285227), 
                     (0.24282915872229727, 0.008945883462747176), 
                     (2.1068546188568154, 0.27649220059531304)]

        super().__init__(frequency=frequency, episode_len=episode_len,
                         path=path, Q=Q, R=R, gui=gui,
                         actiontype=ActionType.DIRECT, rewardtype=rewardtype,
                         s0=s0, friction=friction, log=log, conditionReward=conditionReward, render_mode=render_mode)

    def sampleS0(self) -> np.array:
        c = np.random.choice([0, 1])
        vals = self.vals
        sign = np.array([-1, -1, -1, 1]) if c == 0 else np.array([1, 1, 1, -1])
        s0 = np.array([np.random.normal(vals[0][0], vals[0][1]),
                       np.random.normal(vals[1][0], vals[1][1]),
                       np.random.normal(vals[2][0], vals[2][1]),
                       np.random.normal(vals[3][0], vals[3][1])])
        s0 = np.multiply(sign, s0)
        return s0

    def step(self, action: np.array) -> Tuple[np.array, float, bool, bool, dict]:
        """The step function simulates a single control step in the environment.
        Input:
            action: np.array
                The action chosen by the agent.

        Return:
            state: np.array
                The current state.
            reward: float
                The current reward.
            done: bool
                Indicating if the environment is finished.
            info: dict
                The true answer.
        """

        return super().step(0.5*action)

