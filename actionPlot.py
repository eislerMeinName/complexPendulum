from complexPendulum.agents.neuralAgents import * 
from complexPendulum.assets import * 
from complexPendulum.agents import LQAgent
from complexPendulum.agents.NeuralAgent import NeuralAgent
from complexPendulum.envs import ComplexPendulum

import matplotlib.pyplot as plt
import numpy as np

def actionplot(model1, model2, name1: str, name2: str) -> None:
    """
    Method that compares two policies within a reduced state space by plotting the corresponding actions.
    
    Inputs:
        model1: LQAgent
            The model to compare against.
        model2: NeuralAgent
            The learned model.
        name1: str
            Name of 1. model.
        name2: str
            Name of 2. model.
    """

    fig = plt.figure()

    ax = plt.axes(projection='3d')

    theta = np.arange(-0.25, 0.25, 0.05)
    x = np.arange(-0.4, 0.4, 0.05)

    xs, ts = np.meshgrid(x, theta)

    ts = ts.flatten()
    xs = xs.flatten()
    
    Z1 = [np.clip(-model1.predict(np.array([xs[i], 0, ts[i], 0])) @ np.array([xs[i], 0, ts[i], 0]), -0.5, 0.5) for i in range(0, len(ts))]
    
    Z2 = [model2.predict(np.array([xs[i], 0, ts[i], 0]))[0] for i in range(0, len(ts))]
    ax.plot_trisurf(xs, ts, Z1, edgecolor='none', color='tab:green', label=name1)
    ax.plot_trisurf(xs, ts, Z2, edgecolor='none', color='tab:blue', label=name2)

    ax.set_xlabel('x')
    ax.set_ylabel('θ')
    ax.set_zlabel('pwm')

    ax.view_init(elev=10., azim=-40)
    plt.legend()
    
    plt.xticks(np.arange(-0.4, 0.42, 0.4))
    plt.yticks(np.array([-0.25, 0, 0.25]))
    ax.set_zticks(np.array([-0.5, 0, 0.5]))
    plt.show()


if __name__ == "__main__":
    plt.rc('font', size=13)
    plt.rcParams["figure.figsize"] = (4,3)
    a1 = LQAgent(ComplexPendulum(Q=Setup3.Q, R=Setup3.R))
    a2 = NeuralAgent(DirectQR3, None)
    actionplot(a1, a2, 'LQ3', 'QR3')
