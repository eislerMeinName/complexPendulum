from typing import Tuple
import numpy as np
import control as ct

from complexPendulum.agents import LQAgent
from complexPendulum.envs import ComplexPendulum
from complexPendulum.assets import RewardType, ActionType


class BaselineGainPendulum(ComplexPendulum):
    """Gym environment that models baselined GainPendulum control."""

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
                The starting state indicating random sampling or fixed starting state.
            friction: bool = True
                Use friction during simulation.
            log: bool = True
                Use logger.
            render_moder: str = "rgb_array"
                The render mode.
        """

        super().__init__(frequency=frequency, episode_len=episode_len,
                         path=path, Q=Q*10, R=R*10, gui=gui,
                         actiontype=ActionType.GAIN, rewardtype=rewardtype,
                         s0=s0, friction=friction, log=log, conditionReward=conditionReward, render_mode=render_mode)

        self.K, _, _ = ct.lqr(self.getLinearSS(), self.Q, self.R)

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

        action = action / 2
        return super().step(self.K + action * self.K)