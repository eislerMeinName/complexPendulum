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
    env.state = s0
    env.last_state = s0
    s = s0
    while not done:
        action = model.predict(s)
        s, rew, term, trun, _ = env.step(action)
        comRew += rew
        done = term or trun
   
    return comRew

def rewPlot2d(model: NeuralAgent | LQAgent, actiontype: ActionType, name: str) -> None:
    """Method that plots the reward in 2D

    Inputs:
        model: NeuralAgent | LQAgent
            The model to be evaluated.
        actiontype: ActionType
            The type of the action.
        name: str
            The label of the model in the plot

    """
    
    theta = np.arange(-0.25, 0.25, 0.01)
    x = np.arange(-0.2, 0.2, 0.01)

    R = []

    for i in tqdm(range(len(theta))):
        avgRew = []
        for j in range(len(x)):
            state = np.array([x[j], 0, theta[i], 0])
            avgRew.append(simEpisode(model, actiontype, state)) 

        R.append(np.mean(avgRew))

    plt.plot(theta, R, label=name)

if __name__ == "__main__":
    model = NeuralAgent(DirectQR1, None)
    
    model2 = LQAgent(ComplexPendulum(Q=Setup1.Q, R=Setup1.R))

    plt.rc('font', size=16)
    fig = plt.figure()
   
    rewPlot2d(model, ActionType.DIRECT, "DirectQR1")
    rewPlot2d(model2, ActionType.GAIN, "LQ1")

    plt.legend()
    plt.show()


