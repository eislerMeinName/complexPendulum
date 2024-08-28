import gymnasium as gym
import numpy as np
from complexPendulum.agents import LQAgent, SwingUpAgent, CombinedAgent
from complexPendulum.assets import ActionType, RewardType
from complexPendulum.envs import ComplexPendulum
import time

REALTIME: bool = True
GUI: bool = True
S0: np.array = np.array([-0.4, 0, -0.245, 0])
Q: np.array = np.eye(4)

if __name__ == "__main__":
    #env = ComplexPendulum(100, 15, "params.xml", Q, np.eye(1), True, s0=S0, actiontype=ActionType.DIRECT, friction=True)
    #env = gym.make('Lqr-v0')
    env = gym.make('complexPendulum-v0', gui=True, s0=S0)
    lq = LQAgent(env)
    swingup = SwingUpAgent(env)
    agent = CombinedAgent(swingup, lq)

    state, _ = env.reset()
    done = False
    t00 = time.time()
    t0 = time.time_ns()
    while not done:
        while time.time_ns()-t0 < 10000000 and REALTIME:
            pass
        t0 = time.time_ns()
        action = agent.sample(state)
        #print(state)
        #input()
        state, rew, done, _, _ = env.step(action)

    print(str(round(time.time() - t00, 5)) + 's')

    env.stats()
    env.logger.write('test.csv')
    env.close()
