from typing import Tuple

from complexPendulum.envs.PendulumEnv import ComplexPendulum
from complexPendulum.assets import RewardType, ActionType
import numpy as np


class DirectPendulum(ComplexPendulum):

    def __init__(self,
                 frequency: float = 100,
                 episode_len: float = 15,
                 path: str = 'params.xml',
                 Q: np.ndarray = np.eye(4),
                 R: np.ndarray = np.eye(1),
                 gui: bool = False,
                 rewardtype: RewardType = RewardType.LQ,
                 s0: np.array = None,
                 friction: bool = True,
                 log: bool = True,
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
            s0: np.array = None
                The starting state indicating random sampling or fixed starting state.
            friction: bool = True
                Use friction during simulation.
            log: bool = True
                Use logger.
        """

        super().__init__(frequency=frequency, episode_len=episode_len,
                         path=path, Q=Q, R=R, gui=gui,
                         actiontype=ActionType.DIRECT, rewardtype=rewardtype,
                         s0=s0, friction=friction, log=log)

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
