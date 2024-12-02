import pandas as pd
import numpy as np
from sympy.physics.units import acceleration

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

    statesM006, _, _, TimeM006 = loadMatlabLog("../data/acceleration/Beschl006.csv")
    statesM008, _, _, TimeM008 = loadMatlabLog("../data/acceleration/Beschl008.csv")
    statesM01, _, _, TimeM01 = loadMatlabLog('../data/acceleration/Beschl.csv')
    statesM012, _, _, TimeM012 = loadMatlabLog("../data/acceleration/Beschl012.csv")

    statesP006, _, _, TimeP006 = loadLog("../data/acceleration/BeschlSim006.csv")
    statesP008, _, _, TimeP008 = loadLog("../data/acceleration/BeschlSim008.csv")
    statesP01, _, _, TimeP01 = loadLog('../data/acceleration/BeschlSim01.csv')
    statesP012, _, _, TimeP012 = loadLog('../data/acceleration/BeschlSim012.csv')


    XM006 = [s[0, 0] for s in statesM006]
    XdotM006 = [s[0, 1] for s in statesM006]

    XM008 = [s[0, 0] for s in statesM008]
    XdotM008 = [s[0, 1] for s in statesM008]

    XM01 = [s[0, 0] for s in statesM01]
    XdotM01 = [s[0, 1] for s in statesM01]

    XM012 = [s[0, 0] for s in statesM012]
    XdotM012 = [s[0, 1] for s in statesM012]

    XP006 = [s[0, 0] for s in statesP006]
    XdotP006 = [s[0, 1] for s in statesP006]

    XP008 = [s[0, 0] for s in statesP008]
    XdotP008 = [s[0, 1] for s in statesP008]

    XP01 = [s[0, 0] for s in statesP01]
    XdotP01 = [s[0, 1] for s in statesP01]

    XP012 = [s[0, 0] for s in statesP012]
    XdotP012 = [s[0, 1] for s in statesP012]

    fig, axs = plt.subplots(2, 1)
    fig.tight_layout()
    title: str = 'Comparison of Acceleration'
    fig.suptitle(title, fontsize=15)

    dot = '\u0307'
    #axs[0].plot(TimeM006, XM006, c='b', label='RealWorld (0.06)')
    #axs[0].plot(TimeP006, XP006, '--', c='b', label='Simulation (0.06)')
    axs[0].plot(TimeM008, XM008, c='g', label='RealWorld (0.08)')
    axs[0].plot(TimeP008, XP008, '--', c='g', label='Simulation (0.08)')

    axs[0].plot(TimeM01, XM01, c='r', label='RealWorld (0.1)')
    axs[0].plot(TimeP01, XP01, '--', c='r', label='Simulation (0.1)')

    axs[0].plot(TimeM012, XM012, c='black', label='RealWorld (0.12)')
    axs[0].plot(TimeP012, XP012, '--', c='black', label='Simulation (0.12)')
    #axs[0].axvline(x=index, color='g', label='axvline - full height')
    #axs[0].axvline(x=index2, color='g', label='axvline - full height')
    axs[0].set_xlabel('t [s]')
    axs[0].set_ylabel('X [m]')
    axs[0].set_xlim(0, 3)
    axs[0].set_ylim(0, 2)

    #axs[1].plot(TimeM006, XdotM006, c='b', label='RealWorld (0.06)')
    #axs[1].plot(TimeP006, XdotP006, '--', c='b', label='Simulation (0.06)')
    axs[1].plot(TimeM008, XdotM008, c='g', label='RealWorld (0.08)')
    axs[1].plot(TimeP008, XdotP008, '--', c='g', label='Simulation (0.08)')

    axs[1].plot(TimeM01, XdotM01, c='r', label='RealWorld (0.1)')
    axs[1].plot(TimeP01, XdotP01, '--', c='r', label='Simulation (0.1)')

    axs[1].plot(TimeM012, XdotM012, c='black', label='RealWorld (0.12)')
    axs[1].plot(TimeP012, XdotP012, '--', c='black', label='Simulation (0.12)')


    #axs[1].axvline(x=index, color='g', label='axvline - full height')
    axs[1].set_xlabel('t [s]')
    axs[1].set_ylabel('X' + dot + ' [m/s]')
    maxX = max(XdotM012) if abs(max(XdotM012)) >= abs(min(XdotM012)) else min(XdotM012)
    axs[1].set_ylim(-1.1 * abs(maxX), 1.1 * abs(maxX))
    axs[1].set_xlim(0, 3)

    axs[0].legend()
    axs[1].legend()
    plt.show()

