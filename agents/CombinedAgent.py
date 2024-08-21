import numpy as np
from PendelEnv import ComplexPendulum
from assets.EnvTypes import ActionType, ActionTypeError
from ProportionalAgent import ProportionalAgent


class CombinedAgent:

    def __init__(self, agent1: tuple, agent2: ProportionalAgent, env: ComplexPendulum) -> None:
        if env.actiontype is ActionType.GAIN:
            raise ActionTypeError([env.actiontype], [ActionType.DIRECT], 'Environment has wrong actiontype.')

        self.a1, self.min1, self.max1 = agent1
        self.a2 = agent2

    def sample(self, state: np.array) -> np.array:
        if self.min1 < state[2] < self.max1:
            return self.a1.sample(state)
        else:
            K = self.a2.sample(state)
            s = state.reshape(1, -1).copy()
            return np.array([-(K.reshape(1, -1)@state.T)[0, 0]])
