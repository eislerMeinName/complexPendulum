import gymnasium as gym
import numpy as np
import time

from stable_baselines3 import SAC

from complexPendulum.agents import LQAgent
from complexPendulum.agents.NeuralAgent import NeuralAgent
from complexPendulum.assets import ActionType, RewardType, EvalSetup
from complexPendulum.assets import Setup1, Setup2, Setup3, Setup4, Setup5, Setup6

DEFAULT_STEPS: int = 100000
DEFAULT_FREQ: int = 100
DEFAULT_EPISODE_LEN: float = 60
DEFAULT_PATH: str = 'params.xml'
DEFAULT_SETUP: EvalSetup = Setup1
DEFAULT_S0: np.array = np.array([0.1, 0, 0.2, 0])
DEFAULT_FRICTION: bool = True
DEFAULT_NAME: str = 'results/best_model'
DEFAULT_GUI: bool = True
DEFAULT_LOG: bool = True
DEFAULT_ENV = "complexPendulum-v0"


def run(frequency: float = DEFAULT_FREQ,
        episode_len: float = DEFAULT_EPISODE_LEN,
        path: str = DEFAULT_PATH,
        setup: EvalSetup = DEFAULT_SETUP,
        s0: np.array = DEFAULT_S0,
        gui: bool = DEFAULT_GUI,
        friction: bool = DEFAULT_FRICTION,
        log: bool = DEFAULT_LOG,
        name: str = DEFAULT_NAME) -> None:

    agent = NeuralAgent({"Agent": SAC.load(name)})
    agent2 = SAC.load(name)
    #actiontype = ActionType.DIRECT if agent.action_space.shape == (1,) else ActionType.GAIN
    eval_env = gym.make(DEFAULT_ENV, frequency=frequency,
                        episode_len=episode_len, path=path,
                        Q=setup.Q, R=setup.R,
                        rewardtype=setup.func, s0=s0, gui=gui,
                        friction=friction, log=log, render_mode="human", actiontype=ActionType.GAIN)
    print(eval_env)

    state, _ = eval_env.reset()
    done = False
    trun = False
    t00 = time.time()
    while not done and not trun:
        action = agent.predict(state)
        action2, _ = agent2.predict(state, deterministic=True)
        action2 = (action2-1)*50

        state, rew, done, trun, _ = eval_env.step(action)

    print(str(round(time.time() - t00, 5)) + 's')

    eval_env.unwrapped.stats()
    eval_env.unwrapped.logger.write(name.replace('results/', 'logs/').replace('.zip', '.csv'))
    eval_env.close()


if __name__ == "__main__":
    run()
