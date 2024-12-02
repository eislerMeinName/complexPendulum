import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from sympy.strategies.branch import condition

from complexPendulum.agents import LQAgent, SwingUpAgent, CombinedAgent, ProportionalAgent
from complexPendulum.agents.NeuralAgent import NeuralAgent
from complexPendulum.agents.neuralAgents import nAgent1, nAgent2
from complexPendulum.assets import ActionType
import time

REALTIME: bool = True
GUI: bool = True
S0: np.array = np.array([0, 0, 0.01, 0], dtype=np.float64)
Q: np.array = np.eye(4)/100
R = np.ones(1)/100

if __name__ == "__main__":
    env = gym.make('complexPendulum-v0', gui=GUI, s0=S0, friction=True,
                   episode_len=30, Q=Q, R=R, render_mode="human", actiontype=ActionType.DIRECT,
                   conditionReward=False)
    lq = LQAgent(env.unwrapped)
    print(lq.K)
    neural = NeuralAgent(nAgent2, None)
    swingup = SwingUpAgent(env.unwrapped)
    agent = CombinedAgent(swingup, lq)

    state, _ = env.reset()
    done = False
    t00 = time.time()
    t0 = time.time_ns()
    while not done:
        while time.time_ns()-t0 < 10000000 and REALTIME:
            pass
        t0 = time.time_ns()
        action = agent.predict(state)
        #action = np.array([0])
        state, rew, term, trun, _ = env.step(action)
        done = term or trun
        if done:
            print(env.unwrapped.time)
            print(term)
            print(trun)

    print(str(round(time.time() - t00, 5)) + 's')

    env.unwrapped.stats()
    env.unwrapped.logger.write('test.csv')
    env.close()
