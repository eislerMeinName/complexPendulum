import gymnasium as gym
import numpy as np
import time

from stable_baselines3 import SAC

from complexPendulum.assets import ActionType, Setup1, RewardType
from test import REALTIME

s0: np.array = np.array([0, 0, 0.0001, 0], dtype=np.float32)
REALTIME: bool = True

def run() -> None:

    setup = Setup1
    eval_env = gym.make('complexPendulum-v0', frequency=100,
                         episode_len=15, path='params.xml',
                         Q=setup.Q, R=setup.R, actiontype=ActionType.DIRECT,
                         rewardtype=RewardType.LQ, s0=s0, gui=True,
                         friction=True, log=True, k=setup.k)

    agent = SAC.load('results/best_model.zip')

    state, _ = eval_env.reset()
    done = False
    t00 = time.time()
    t0 = time.time_ns()
    while not done:
        while time.time_ns() - t0 < 10000000 and REALTIME:
            pass
        t0 = time.time_ns()
        action, _ = agent.predict(state, deterministic=True)
        state, rew, done, _, _ = eval_env.step(action)

    print(str(round(time.time() - t00, 5)) + 's')

    eval_env.unwrapped.stats()
    eval_env.unwrapped.logger.write('test.csv')
    eval_env.close()


if __name__ == "__main__":
    run()