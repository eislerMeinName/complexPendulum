import numpy as np
from complexPendulum.envs import ComplexPendulum


class SwingUpAgent:
    """Implements a SwingUpController."""

    def __init__(self, env: ComplexPendulum,
                 ksu: float = 1.3, kcw: float = 3,
                 kvw: float = 3, kem: float = 7,
                 eta: float = 1.05, E0: float = 0.05,
                 xmax: float = 0.4, vmax: float = 5,
                 minAng: float = -0.25, maxAng: float = 0.25) -> None:
        """
        Initialization.
        Input:
            env: ComplexPendulum
                The environment containing the important model parameters.
            ksu: float
                The swing-up gain.
            kcw: float
                The cart position well gain.
            kvw: float
                The cart velocity well gain.
            kem: float
                The energy maintenance gain.
            eta: float
                The energy maintenance parameter.
            E0: float
                The desired energy for the swing-up.
            xmax: float
                Distance of cart position well from rail center.
            vmax: float
                "Distance" of cart velocity well.
            minAng: float
                The min of the stabilization zone for switching to state feedback.
            maxAng: float
                The max of the stabilization zone for switching to state feedback.
        """

        self.env = env
        self.ksu = ksu
        self.kcw = kcw
        self.kvw = kvw
        self.kem = kem
        self.eta = eta
        self.E0 = E0
        self.xmax = xmax
        self.vmax = vmax
        self.min = minAng
        self.max = maxAng
        self.mp, self.l, self.J, self.m, _, _, self.g, _, _, _, _ = env.params

    def predict(self, state: np.array) -> np.array:
        """Samples the action (Direct PWM) based on the current state.
        Input:
            state: np.array
                The current state.

        Return:
            pwm: np.array
                The action applied as pwm.
        """

        energy: float = 0.5*self.J*state[3]**2 + self.mp*self.l*self.g*(np.cos(state[2])-1)
        poslim: float = np.clip(state[0], -self.xmax, self.xmax)
        vellim: float = np.clip(state[1], -self.vmax, self.vmax)

        swingup: float = self.ksu*np.sign(state[3]*np.cos(state[2]))
        cartposwell: float = self.kcw*np.sign(state[0])*np.log10(1.0000001 - np.abs(poslim/self.xmax))
        cartvelwell: float = self.kvw*np.sign(state[1])*np.log10(1.0000001 - np.abs(vellim/self.vmax))
        energymaintenance: float = self.kem*(np.exp(np.abs(energy - self.eta*self.E0)) - 1) * np.sign(energy - self.E0) * np.sign(state[3]*np.cos(state[2]))
        return np.array([-swingup+cartposwell+cartvelwell+energymaintenance])
