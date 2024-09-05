import gymnasium as gym
import numpy as np
from complexPendulum.agents import LQAgent, SwingUpAgent, CombinedAgent, ProportionalAgent
import time

REALTIME: bool = True
GUI: bool = True
S0: np.array = np.array([0, 0, np.pi, 0], dtype=np.float32)
Q: np.array = np.eye(4)

if __name__ == "__main__":
    env = gym.make('complexPendulum-v0', gui=GUI, s0=S0, friction=True, episode_len=15, Q=Q)
    lq = LQAgent(env.unwrapped)
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
        state, rew, done, _, _ = env.step(action)

    print(str(round(time.time() - t00, 5)) + 's')

    env.unwrapped.stats()
    env.unwrapped.logger.write('test.csv')
    env.close()
