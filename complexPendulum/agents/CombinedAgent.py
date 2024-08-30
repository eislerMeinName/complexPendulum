import numpy as np
from complexPendulum.assets.EnvTypes import ActionType, ActionTypeError
from complexPendulum.agents.ProportionalAgent import ProportionalAgent
from complexPendulum.agents.SwingUpAgent import SwingUpAgent


class CombinedAgent:
    """Implements an Agent consisting of a swing-up and proportional controller."""

    def __init__(self, agent1: SwingUpAgent, agent2: ProportionalAgent) -> None:
        """
        Initialization:

        Input:
            agent1: SwingUpAgent
                The swing-up controller.
            agent2: ProportionalAgent
                The proportional controller.
        """

        if agent1.env.actiontype is ActionType.GAIN:
            raise ActionTypeError([agent1.env.actiontype], [ActionType.DIRECT], 'Environment has wrong actiontype.')

        self.a1 = agent1
        self.min1 = agent1.min
        self.max1 = agent1.max
        self.a2 = agent2

    def predict(self, state: np.array) -> np.array:
        """Samples the action (Direct PWM) based on the current state.
        Input:
            state: np.array
                The current state.

        Return:
            pwm: np.array
                The action applied as pwm.
        """

        if not self.min1 < state[2] < self.max1:
            return self.a1.predict(state)
        else:
            K = self.a2.predict(state)
            s = state.reshape(1, -1).copy()
            return np.array([-(K.reshape(1, -1)@s.T)[0, 0]])
