import numpy as np
from assets.EnvTypes import ActionType
import matplotlib.pyplot as plt
import pandas as pd


class Logger:
    """A logger class, that logs, writes and plots the data of the step response."""

    def __init__(self, actiontype: ActionType, s0: np.array) -> None:
        """
        Initialization.
        Input:
            actiontype: ActionType
                The type of action the agent uses.
            s0: np.array
                The starting state.
        """

        self.Time = [0.0]
        self.X = [s0[0]]
        self.Xdot = [s0[1]]
        self.Theta = [s0[2]]
        self.Thetadot = [s0[3]]
        self.pwm = []
        self.force = []
        self.a = []
        self.actiontype = actiontype
        self.STEP = 0
        self.ret = 0

    def log(self, t: float, state: np.array, pwm: float, force: float, a, rew: float) -> None:
        """Logs the current values at time step t.
        Input:
            t: float
                The timestep.
            state: np.array
                The current state.
            pwm: float
                The current pwm.
            force: float
                The current applied force u.
            a: float or np.array
                The current action.
            rew: float
                The current reward.
        """

        self.Time.append(t)
        self.X.append(state[0])
        self.Xdot.append(state[1])
        self.Theta.append(state[2])
        self.Thetadot.append(state[3])
        self.pwm.append(pwm)
        self.force.append(force)
        self.a.append(a)
        self.STEP += 1
        self.ret += rew

    def reset(self, s0: np.array) -> None:
        """Resets the current Logger.
        Input:
            s0: np.array
                The starting state.
        """

        self.Time = [0]
        self.X = [s0[0]]
        self.Xdot = [s0[1]]
        self.Theta = [s0[2]]
        self.Thetadot = [s0[3]]
        self.pwm = []
        self.force = []
        self.a = []
        self.ret = 0
        self.STEP = 0

    def write(self, path: str) -> None:
        """Writes the step response data to a csv file.
        Input:
            path: str
                The path to the csv file.
        """

        self.Time.insert(0, 0)
        self.pwm.insert(0, None)
        self.force.insert(0, None)
        d = {'Time': self.Time,
             'X': self.X,
             'Xdot': self.Xdot,
             'Theta': self.Theta,
             'Thetadot': self.Thetadot,
             'pwm': self.pwm,
             'force': self.force,
             }
        df = pd.DataFrame(data=d)
        df.to_csv(path, index=False)

    def show(self) -> None:
        """Plots the step response data."""

        nrows = 5 if self.actiontype is ActionType.GAIN else 4
        fig, axs = plt.subplots(nrows, 2)
        fig.tight_layout()
        title: str = 'Logged Control Data: ' + str(round(self.ret, 2))
        fig.suptitle(title, fontsize=15)

        dot = '\u0307'
        axs[0, 0].plot(self.Time, self.X, c='r')
        axs[0, 0].set_xlabel('t [s]')
        axs[0, 0].set_ylabel('X [m]')
        axs[0, 0].set_ylim(-1, 1)

        axs[1, 0].plot(self.Time, self.Xdot, c='r')
        axs[1, 0].set_xlabel('t [s]')
        axs[1, 0].set_ylabel('X' + dot + ' [m/s]')
        maxX = max(self.Xdot) if abs(max(self.Xdot)) >= abs(min(self.Xdot)) else min(self.Xdot)
        axs[1, 0].set_ylim(-1.1*abs(maxX), 1.1*abs(maxX))

        axs[2, 0].plot(self.Time, self.Theta, c='r')
        axs[2, 0].set_xlabel('t [s]')
        axs[2, 0].set_ylabel(r'$\theta$')
        axs[2, 0].set_ylim(-1.1*np.pi, 1.1*np.pi)

        axs[3, 0].plot(self.Time, self.Thetadot, c='r')
        axs[3, 0].set_xlabel('t [s]')
        axs[3, 0].set_ylabel(r'$\dot{\theta}$ [m/s]')
        maxTheta = max(self.Thetadot) if abs(max(self.Thetadot)) >= abs(min(self.Thetadot)) else min(self.Thetadot)
        axs[3, 0].set_ylim(-1.1 * abs(maxTheta), 1.1 * abs(maxTheta))

        _ = self.Time.pop(0)

        if self.actiontype is ActionType.GAIN:
            a1 = []
            a2 = []
            a3 = []
            a4 = []
            for act in self.a:
                a1.append(act[0])
                a2.append(act[1])
                a3.append(act[2])
                a4.append(act[3])
            axs[0, 1].plot(self.Time, a1, c='b')
            axs[0, 1].set_xlabel('t [s]')
            axs[0, 1].set_ylabel('a1')

            axs[1, 1].plot(self.Time, a2, c='b')
            axs[1, 1].set_xlabel('t [s]')
            axs[1, 1].set_ylabel('a2')

            axs[2, 1].plot(self.Time, a3, c='b')
            axs[2, 1].set_xlabel('t [s]')
            axs[2, 1].set_ylabel('a3')

            axs[3, 1].plot(self.Time, a4, c='b')
            axs[3, 1].set_xlabel('t [s]')
            axs[3, 1].set_ylabel('a4')

            axs[4, 1].plot(self.Time, self.pwm, c='g')
            axs[4, 1].set_xlabel('t [s]')
            axs[4, 1].set_ylabel('pwm')
            axs[4, 1].set_ylim(-0.7, 0.7)

            axs[4, 0].plot(self.Time, self.force, c='y')
            axs[4, 0].set_xlabel('t [s]')
            axs[4, 0].set_ylabel('force[N]')

        else:
            axs[0, 1].plot(self.Time, self.a, c='b')
            axs[0, 1].set_xlabel('t [s]')
            axs[0, 1].set_ylabel('a')

            axs[1, 1].plot(self.Time, self.pwm, c='g')
            axs[1, 1].set_xlabel('t [s]')
            axs[1, 1].set_ylabel('pwm')
            axs[1, 1].set_ylim(-0.7, 0.7)

            axs[2, 1].plot(self.Time, self.force, c='y')
            axs[2, 1].set_xlabel('t [s]')
            axs[2, 1].set_ylabel('force [N]')

            axs[3, 1].remove()

        plt.show()
