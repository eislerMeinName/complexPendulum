import numpy as np
from complexPendulum.assets.EnvTypes import ActionType, ActionTypeError
from complexPendulum.agents.ProportionalAgent import ProportionalAgent
from complexPendulum.agents.NeuralAgent import NeuralAgent
from complexPendulum.agents.SwingUpAgent import SwingUpAgent


class CombinedAgent:
    """Implements an Agent consisting of a swing-up and proportional controller."""

    def __init__(self, agent1: SwingUpAgent, agent2: ProportionalAgent | NeuralAgent) -> None:
        """
        Initialization:

        Input:
            agent1: SwingUpAgent
                The swing-up controller.
            agent2: ProportionalAgent | NeuralAgent
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
            force =  self.a1.predict(state)
            a_fric = force + np.sign(force) * self.a1.env.params[8]
            pwm = a_fric / self.a1.env.params[7]

            pwm = np.clip(pwm, -0.5, 0.5)
            return pwm
        else:
            a = self.a2.predict(state)
            if a.size == 4:
                s = state.reshape(1, -1).copy()
                a = -(a.reshape(1, -1)@s.T)[0, 0]
                a_fric = a + np.sign(a) * self.a1.env.params[8]
                pwm = a_fric / self.a1.env.params[7]
                pwm = np.clip(pwm, -0.5, 0.5)
                return np.array([pwm])
            else: return a
