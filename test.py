import numpy as np
from PendelEnv import ComplexPendulum
from assets.EnvTypes import ActionType, RewardType
from agents.ProportionalAgent import ProportionalAgent
from agents.LQAgent import LQAgent
import time

if __name__ == "__main__":
    Q = np.eye(4)
    #Q[0, 0] = 10
    env = ComplexPendulum(100, 5, "params.xml", Q, np.eye(1), True, actiontype=ActionType.GAIN, s0=np.array([-0.9, 0, np.pi/8, 0]), friction=True)
    agent = LQAgent(env)
    state = env.reset()
    done = False

    t0 = time.time()
    while not done:
        action = agent.sample(state)
        state, rew, done, _ = env.step(action)

    print(str(round(time.time() - t0, 5)) + 's')
    env.stats()
    env.logger.write('test.csv')
    _ = env.reset()
    env.close()