import numpy as np
from complexPendulum.agents.ProportionalAgent import ProportionalAgent
from complexPendulum.envs.PendelEnv import ComplexPendulum
import control as ct


class LQAgent(ProportionalAgent):
    """Linear Quadratic Agent, that optimises LQ cost function of a given System."""

    def __init__(self, sys: ComplexPendulum) -> None:
        """
        Initialization.
        Input:
            sys: ComplexPendulum
                Your Gym environment.
        """

        K, _, _ = ct.lqr(sys.getLinearSS(), sys.Q, sys.R)
        K = np.array(K[0])
        super().__init__(K)
