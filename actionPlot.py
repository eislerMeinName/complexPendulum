from complexPendulum.agents.neuralAgents import * 
from complexPendulum.assets import * 
from complexPendulum.agents import LQAgent
from complexPendulum.agents.NeuralAgent import NeuralAgent
from complexPendulum.envs import ComplexPendulum

import matplotlib.pyplot as plt
import numpy as np

def actionplot(model1, model2) -> None:
    """
    Method that compares two policies within a reduced state space by plotting the corresponding actions.
    
    Inputs:
        model1: LQAgent
            The model to compare against.
        model2: NeuralAgent
            The learned model.
    """

    fig = plt.figure()

    ax = plt.axes(projection='3d')

    theta = np.arange(-0.25, 0.25, 0.01)
    x = np.arange(-0.4, 0.4, 0.01)

    xs, ts = np.meshgrid(x, theta)

    ts = ts.flatten()
    xs = xs.flatten()
    
    Z1 = [np.clip(-model1.predict(np.array([xs[i], 0, ts[i], 0])) @ np.array([xs[i], 0, ts[i], 0]), -0.5, 0.5) for i in range(0, len(ts))]
    
    Z2 = [model2.predict(np.array([xs[i], 0, ts[i], 0]))[0] for i in range(0, len(ts))]
    ax.plot_trisurf(xs, ts, Z1, edgecolor='none', color='g')
    ax.plot_trisurf(xs, ts, Z2, edgecolor='none', color='b')

    ax.set_xlabel('x')
    ax.set_ylabel('θ')
    ax.set_zlabel('a')

    plt.show()


if __name__ == "__main__":
    plt.rc('font', size=16)
    a1 = LQAgent(ComplexPendulum(Q=Setup1.Q, R=Setup1.R))
    a2 = NeuralAgent(DirectQR1, None)
    actionplot(a1, a2)
