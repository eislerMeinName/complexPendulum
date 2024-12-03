import gymnasium as gym
import numpy as np
import time

from stable_baselines3 import PPO
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray

from complexPendulum.agents import LQAgent, CombinedAgent, SwingUpAgent, ProportionalAgent
from complexPendulum.agents.NeuralAgent import NeuralAgent
from complexPendulum.agents.neuralAgents import DirectQR1
from complexPendulum.assets import ActionType, RewardType, EvalSetup
from complexPendulum.assets import Setup1, Setup2, Setup3, Setup4, Setup5

DEFAULT_STEPS: int = 100000
DEFAULT_FREQ: int = 100
DEFAULT_EPISODE_LEN: float = 60
DEFAULT_PATH: str = 'params.xml'
DEFAULT_SETUP: EvalSetup = Setup1
DEFAULT_S0= None
#DEFAULT_S0: np.array = np.array([0.177913818359375,0.27539062,0.2454369260617026,-0.46019424])
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


    eval_env = gym.make(DEFAULT_ENV, frequency=frequency,
                        episode_len=episode_len, path=path,
                        Q=setup.Q, R=setup.R,
                        rewardtype=setup.func, s0=s0, gui=gui,
                        friction=friction, log=log, render_mode="human", actiontype=ActionType.GAIN)

    agent = NeuralAgent({"Algo": "PPO", "Action": "Base" , "Agent": PPO.load(name)}, LQAgent(eval_env.unwrapped).K)
    lq = LQAgent(eval_env.unwrapped)
    #a = ProportionalAgent(np.array([-1, -3.0256, -23.7068, -4.4886]))
    #swingup = SwingUpAgent(eval_env.unwrapped)
    #agent = CombinedAgent(swingup, a)

    state, _ = eval_env.reset()
    done = False
    trun = False
    t00 = time.time()
    t0 = time.time_ns()
    while not trun:
        while time.time_ns()-t0 < 10000000:
            pass
        t0 = time.time_ns()
        action = agent.predict(state)
        print((action/lq.K)[0])
        state, rew, done, trun, _ = eval_env.step(action)

    print(str(round(time.time() - t00, 5)) + 's')
    eval_env.unwrapped.stats()
    path = name.replace('results/', 'logs/')
    eval_env.unwrapped.logger.write(path + ".csv")
    eval_env.close()


if __name__ == "__main__":
    run()
