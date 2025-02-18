import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from tqdm import tqdm

from complexPendulum.agents import LQAgent, SwingUpAgent, CombinedAgent, ProportionalAgent
from complexPendulum.agents.neuralAgents import *
from complexPendulum.assets import * 
from complexPendulum.agents import LQAgent
from complexPendulum.agents.NeuralAgent import NeuralAgent
from complexPendulum.envs import ComplexPendulum
from complexPendulum.assets import Setup1, Setup2, Setup3, Setup4, Setup5


def simEpisode(model: NeuralAgent | LQAgent, actiontype: ActionType, s0: np.array) -> float:
    """Method that simulates a single episode and returns the comulative reward.

    Inputs:
        model: NeuralAgent | LQAgent
            The model that runs the episode.
        actiontype: ActionType
            The actiontype of the model.
        s0: np.array
            The starting state.

    Returns:
        comRew: float
            Comulative reward.

    """
    
    done: bool = False
    comRew: float = 0
    env = gym.make('complexPendulum-v0', gui=False, s0=s0, friction=True,
                    episode_len=10, actiontype=actiontype, log=False,
                    conditionReward=False, Q=Setup1.Q, R=Setup1.R)
    _, _ = env.reset()
    s = s0
    while not done:
        action = model.predict(s)
        s, rew, term, trun, _ = env.step(action)
        comRew += rew
        done = term or trun
   
    return comRew

def rewPlot3d(ax, model: NeuralAgent | LQAgent, actiontype: ActionType, name: str) -> None:
    
    theta = np.arange(-0.25, 0.25, 0.05)
    x = np.arange(-0.2, 0.2, 0.05)

    xs, ts = np.meshgrid(x, theta)

    ts = ts.flatten()
    xs = xs.flatten()

    R = []

    for i in tqdm(range(len(ts))):
        avgRew = []
        for j in range(len(xs)):
            state = np.array([xs[j], 0, ts[i], 0])
            avgRew.append(simEpisode(model, actiontype, state)) 

        R.append(np.mean(avgRew))
    
    c = 'g' if type(model) is LQAgent else 'b'
    ax.plot_trisurf(xs, ts, R, edgecolor='none', color=c, label=name)

if __name__ == "__main__":
    fig = plt.figure()

    ax = plt.axes(projection='3d')

    model = NeuralAgent(DirectQR1, None)
    
    model2 = LQAgent(ComplexPendulum(Q=Setup1.Q, R=Setup1.R))

    plt.rc('font', size=16)
   
    rewPlot3d(ax, model, ActionType.DIRECT, "DirectQR1")
    rewPlot3d(ax, model2, ActionType.GAIN, "LQ1")
    
    ax.set_xlabel('x')
    ax.set_ylabel('θ')
    ax.set_zlabel('a')
    plt.legend()
    plt.show()


