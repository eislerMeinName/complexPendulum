import pandas as pd
import numpy as np
from complexPendulum.assets.Evaluator import loadLog
import matplotlib.pyplot as plt


def loadMatlabLog(path: str) -> tuple:
    """Loading method that loads logged MatlabData.
    Input:
        path: str
            The path to the log file in csv format.

    Return:
        states: list[np.array]
            The logged states relevant for evaluation.
        pwm: list[float]
            The logged pwm signals.
    """

    frame = pd.read_csv(path)
    X = frame['X'].to_list()
    Xdot = frame['Xdot'].to_list()
    Theta = frame['Theta'].to_list()
    Thetadot = frame['Thetadot'].to_list()
    pwm = frame['PWM'].to_list()
    states = [np.array([[X[i], Xdot[i], Theta[i], Thetadot[i]]]) for i in range(0, len(X))]
    return states, pwm, None, frame['Time'].to_list()


if __name__ == "__main__":
    statesM, pwmM, _, TimeM = loadMatlabLog('sim.csv')
    statesP, pwmP, _, TimeP = loadLog('test.csv')
    TimeP.pop()
    XM = [s[0, 0] for s in statesM]
    XdotM = [s[0, 1] for s in statesM]
    ThetaM = [s[0, 2] for s in statesM]
    ThetadotM = [s[0, 3] for s in statesM]

    XP = [s[0, 0] for s in statesP]
    XdotP = [s[0, 1] for s in statesP]
    ThetaP = [s[0, 2] for s in statesP]
    ThetadotP = [s[0, 3] for s in statesP]

    fig, axs = plt.subplots(5, 1)
    fig.tight_layout()
    title: str = 'Comparison of Simulations'
    fig.suptitle(title, fontsize=15)

    dot = '\u0307'
    axs[0].plot(TimeM, XM, c='r')
    axs[0].plot(TimeP, XP, c='b')
    axs[0].set_xlabel('t [s]')
    axs[0].set_ylabel('X [m]')
    axs[0].set_ylim(-1, 1)

    axs[1].plot(TimeM, XdotM, c='r')
    axs[1].plot(TimeP, XdotP, c='b')
    axs[1].set_xlabel('t [s]')
    axs[1].set_ylabel('X' + dot + ' [m/s]')
    maxX = max(XdotM) if abs(max(XdotM)) >= abs(min(XdotM)) else min(XdotM)
    axs[1].set_ylim(-1.1 * abs(maxX), 1.1 * abs(maxX))

    axs[2].plot(TimeM, ThetaM, c='r')
    axs[2].plot(TimeP, ThetaP, c='b')
    axs[2].set_xlabel('t [s]')
    axs[2].set_ylabel(r'$\theta$')
    axs[2].set_ylim(-1.1 * np.pi, 1.1 * np.pi)

    axs[3].plot(TimeM, ThetadotM, c='r')
    axs[3].plot(TimeP, ThetadotP, c='b')
    axs[3].set_xlabel('t [s]')
    axs[3].set_ylabel(r'$\dot{\theta}$ [m/s]')
    maxTheta = max(ThetadotM) if abs(max(ThetadotM)) >= abs(min(ThetadotM)) else min(ThetadotM)
    axs[3].set_ylim(-1.1 * abs(maxTheta), 1.1 * abs(maxTheta))

    axs[4].plot(TimeM, pwmM, c='r')
    axs[4].plot(TimeP, pwmP, c='b')
    axs[4].set_xlabel('t [s]')
    axs[4].set_ylabel('pwm')
    axs[4].set_ylim(-0.7, 0.7)

    plt.show()

