import numpy as np
from PendelEnv import ComplexPendulum
from assets.EnvTypes import ActionType, RewardType
from agents import LQAgent, ProportionalAgent, SwingUpAgent, CombinedAgent
import time

if __name__ == "__main__":
    Q = np.eye(4)
    #Q[0, 0] = 10
    s0 = None
    s0 = np.array([0, 0, np.pi, 0])
    env = ComplexPendulum(100, 5, "params.xml", Q, np.eye(1), True, s0=s0, actiontype=ActionType.DIRECT, friction=True)
    lq = LQAgent(env)
    swingup = SwingUpAgent(env)
    agent = CombinedAgent(swingup, lq)

    state = env.reset()
    done = False
    t0 = time.time()
    while not done:
        action = agent.sample(state)
        state, rew, done, _ = env.step(action)

    print(str(round(time.time() - t0, 5)) + 's')


    env.stats()
    env.logger.write('test.csv')
    env.close()
