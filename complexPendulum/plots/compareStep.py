import pandas as pd
import numpy as np
#from complexPendulum.assets.Evaluator import loadLog
import matplotlib.pyplot as plt

def loadLog(path: str) -> tuple:
    """Loading method that loads logged Data.
    Input:
        path: str
            The path to the log file in csv format.

    Return:
        states: list[np.array]
            The logged states relevant for evaluation.
        pwm: list[float]
            The logged pwm signals.
        force: list[float]
            The logged applied forces.
    """

    frame = pd.read_csv(path)
    X = frame['X'].to_list()
    Xdot = frame['Xdot'].to_list()
    Theta = frame['Theta'].to_list()
    Thetadot = frame['Thetadot'].to_list()
    pwm = frame['pwm'].to_list()
    force = frame['force'].to_list()
    force.pop(-1)
    pwm.pop(-1)
    states = [np.array([[X[i], Xdot[i], Theta[i], Thetadot[i]]]) for i in range(0, len(X))]
    return states, pwm, force, frame['Time'].to_list()


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
    plt.rc('font', size=20)
    plt.rcParams["figure.figsize"] = (20,3)
    statesM, pwmM, _, TimeM = loadMatlabLog('../data/step/Kontrolle.csv')
    statesP, pwmP, _, TimeP = loadLog('../data/step/stepsim.csv')

    time = 0
    index = 0

    for i, s in enumerate(statesM):
        if abs(s[0, 2]) < 0.25:
            time = TimeM[i]
            index = i
            break

    statesM = statesM[i:]
    pwmM = pwmM[i:]
    TimeM = np.array(TimeM[i:]) - time

    #TimeP.pop(len(TimeP)-1)
    XM = [s[0, 0] for s in statesM]
    XdotM = [s[0, 1] for s in statesM]
    ThetaM = [s[0, 2] for s in statesM]
    ThetadotM = [s[0, 3] for s in statesM]

    XP = [s[0, 0] for s in statesP]
    XdotP = [s[0, 1] for s in statesP]
    ThetaP = [s[0, 2] for s in statesP]
    ThetadotP = [s[0, 3] for s in statesP]

    fig, axs = plt.subplots(4, 1)
    fig.tight_layout()
    #title: str = 'Comparison of Step Response'
    #fig.suptitle(title, fontsize=15)

    dot = '\u0307'
    axs[0].plot(TimeM, XM, c='r', label='RealWorld')
    axs[0].plot(TimeP, XP, '--', c='b', label='Simulation')
    #axs[0].axvline(x=index, color='g', label='axvline - full height')
    #axs[0].axvline(x=index2, color='g', label='axvline - full height')
    axs[0].set_xlabel('t [s]')
    axs[0].set_ylabel('X [m]')
    maxX = max(XM) if abs(max(XM)) >= abs(min(XM)) else min(XM)
    axs[0].set_ylim(-1.1 * abs(maxX), 1.1 * abs(maxX))
    axs[0].set_xlim(0, 10)

    axs[1].plot(TimeM, XdotM, c='r', label='RealWorld')
    axs[1].plot(TimeP, XdotP, '--', c='b', label='Simulation')
    #axs[1].axvline(x=index, color='g', label='axvline - full height')
    axs[1].set_xlabel('t [s]')
    axs[1].set_ylabel('X' + dot + ' [m/s]')
    maxX = max(XdotM) if abs(max(XdotM)) >= abs(min(XdotM)) else min(XdotM)
    axs[1].set_ylim(-1.1 * abs(maxX), 1.1 * abs(maxX))
    axs[1].set_xlim(0, 10)

    axs[2].plot(TimeM, ThetaM, c='r', label='RealWorld')
    axs[2].plot(TimeP, ThetaP, '--', c='b', label='Simulation')
    #axs[0].axvline(x=index, color='g', label='axvline - full height')
    axs[2].set_xlabel('t [s]')
    axs[2].set_ylabel(r'$\theta$')
    axs[2].set_ylim(1.2*min(ThetaM), 1.2*max(np.array(ThetaM)))
    axs[2].set_xlim(0, 10)

    axs[3].plot(TimeM, ThetadotM, c='r', label="RealWorld")
    axs[3].plot(TimeP, ThetadotP, '--', c='b', label="Simulation")
    #axs[3].axvline(x=index, color='g', label='axvline - full height')
    axs[3].set_xlabel('t [s]')
    axs[3].set_ylabel(r'$\dot{\theta}$ [m/s]')
    maxTheta = max(ThetadotM) if abs(max(ThetadotM)) >= abs(min(ThetadotM)) else min(ThetadotM)
    axs[3].set_ylim(-1.1 * abs(maxTheta), 1.1 * abs(maxTheta))
    axs[3].set_xlim(0, 10)

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()
    #plt.tight_layout()

    plt.show()
