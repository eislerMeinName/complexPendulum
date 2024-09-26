import gymnasium as gym
import numpy as np
import time

from stable_baselines3 import SAC

from complexPendulum.assets import ActionType, RewardType, EvalSetup
from complexPendulum.assets import Setup1, Setup2, Setup3, Setup4, Setup5, Setup6

DEFAULT_STEPS: int = 1000000
DEFAULT_FREQ: int = 100
DEFAULT_EPISODE_LEN: float = 5
DEFAULT_PATH: str = 'params.xml'
DEFAULT_SETUP: EvalSetup = Setup1
DEFAULT_S0: np.array = None
DEFAULT_FRICTION: bool = True
DEFAULT_NAME: str = 'results/best_model.zip'
DEFAULT_GUI: bool = True
DEFAULT_LOG: bool = True
DEFAULT_REALTIME: bool = True


def run(frequency: float = DEFAULT_FREQ,
        episode_len: float = DEFAULT_EPISODE_LEN,
        path: str = DEFAULT_PATH,
        setup: EvalSetup = DEFAULT_SETUP,
        s0: np.array = DEFAULT_S0,
        gui: bool = DEFAULT_GUI,
        friction: bool = DEFAULT_FRICTION,
        log: bool = DEFAULT_LOG,
        name: str = DEFAULT_NAME,
        realtime: bool = DEFAULT_REALTIME) -> None:

    agent = SAC.load(name)
    actiontype = ActionType.DIRECT if agent.action_space.shape == (1,) else ActionType.GAIN
    eval_env = gym.make('directPendulum-v0', frequency=frequency,
                        episode_len=episode_len, path=path,
                        Q=setup.Q, R=setup.R, #actiontype=actiontype,
                        rewardtype=setup.func, s0=s0, gui=gui,
                        friction=friction, log=log)

    state, _ = eval_env.reset()
    done = False
    trun = False
    t00 = time.time()
    t0 = time.time_ns()
    while not done and not trun:
        while time.time_ns() - t0 < 10000000 and realtime:
            pass
        t0 = time.time_ns()
        action, _ = agent.predict(state, deterministic=True)
        state, rew, done, trun, _ = eval_env.step(action)

    print(str(round(time.time() - t00, 5)) + 's')

    eval_env.unwrapped.stats()
    eval_env.unwrapped.logger.write(name.replace('results/', 'logs/').replace('.zip', '.csv'))
    eval_env.close()


if __name__ == "__main__":
    run()
