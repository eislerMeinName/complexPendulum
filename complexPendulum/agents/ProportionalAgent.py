import numpy as np


class ProportionalAgent:
    """Proportional Agent with constant control gain."""

    def __init__(self, K: np.array) -> None:
        """
        Initialisation.
        Input:
            K: np.array
                The control gain matrix.
        """

        self.K = K

    def predict(self, state: np.array) -> np.array:
        """
        Sample function, that returns the control gain matrix.
        Input:
            state: np.array
                The current state.

        Return:
            K: np.array
                The control gain matrix.
        """

        return self.K

