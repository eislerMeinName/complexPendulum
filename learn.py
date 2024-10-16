import argparse
import gymnasium as gym
import numpy as np
import torch

from stable_baselines3 import SAC, PPO
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from complexPendulum.assets import ActionType, EvalSetup
from complexPendulum.assets import Setup1, Setup2, Setup3, Setup4, Setup5, Setup6

EPISODE_REWARD_THRESHOLD = 0

DEFAULT_STEPS: int = 1000000
DEFAULT_FREQ: int = 100
DEFAULT_EPISODE_LEN: float = 10
DEFAULT_PATH: str = 'params.xml'
DEFAULT_SETUP: EvalSetup = Setup1
DEFAULT_ACTIONTYPE: ActionType = ActionType.GAIN
DEFAULT_S0: np.array = None
DEFAULT_FRICTION: bool = True
DEFAULT_NAME: str = 'results/success_model.zip'
DEFAULT_CONDITION: bool = True
DEFAULT_ENV = "gainPendulum-v0"


def run(steps: int = DEFAULT_STEPS,
        frequency: int = DEFAULT_FREQ,
        episode_len: float = DEFAULT_EPISODE_LEN,
        path: str = DEFAULT_PATH,
        setup: EvalSetup = DEFAULT_SETUP,
        actiontype: ActionType = DEFAULT_ACTIONTYPE,
        s0: np.array = DEFAULT_S0,
        friction: bool = DEFAULT_FRICTION,
        name: str = DEFAULT_NAME,
        condition: bool = DEFAULT_CONDITION
        ) -> None:

    env_kwargs: dict = dict(frequency=frequency, episode_len=episode_len, path=path,
                            Q=setup.Q, R=setup.R,
                            rewardtype=setup.func, s0=s0,
                            friction=friction, log=False, conditionReward=condition)

    train_env = make_vec_env(DEFAULT_ENV, n_envs=4, seed=0, env_kwargs=env_kwargs)

    onpolicy_kwargs: dict = dict(activation_fn=torch.nn.ReLU,
                                 net_arch=dict(vf=[128, 64, 32], pi=[128, 64, 32]))

    offpolicy_kwargs: dict = dict(activation_fn=torch.nn.ReLU,
                                  net_arch=[128, 64, 32])

    agent = PPO(ActorCriticPolicy,
                train_env,
                policy_kwargs=onpolicy_kwargs,
                tensorboard_log='results/tb/',
                verbose=1
                )

    eval_env = make_vec_env(DEFAULT_ENV, env_kwargs=env_kwargs, n_envs=4, seed=0)

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=EPISODE_REWARD_THRESHOLD,
                                                     verbose=1)

    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path='results/',
                                 log_path='results/',
                                 eval_freq=200,
                                 deterministic=True,
                                 render=False
                                 )

    agent.learn(total_timesteps=steps, callback=eval_callback, log_interval=100)
    agent.save(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that allows to train your RL Model on the ComplexPendulum env.")

    parser.add_argument('-s', '--steps', default=DEFAULT_STEPS, type=float,
                        help='Amount of training time steps. (default: 1000000)')
    parser.add_argument('-f', '--frequency', default=DEFAULT_FREQ, type=int,
                        help='The control frequency. (default: 100)')
    parser.add_argument('-l', '--episode_len', default=DEFAULT_EPISODE_LEN, type=float,
                        help='The length of an episode in s. (default: 15)')
    parser.add_argument('-p', '--path', default=DEFAULT_PATH, type=str,
                        help='The path to the parameter file. (default: params.xml)')
    parser.add_argument('-se', '--setup', default=DEFAULT_SETUP, type=EvalSetup,
                        help='The training setup. (default: Setup1)')
    parser.add_argument('-a', '--actiontype', default=DEFAULT_ACTIONTYPE, type=ActionType,
                        help='The type of action. (default: ActionType.DIRECT)')
    parser.add_argument('--s0', default=DEFAULT_S0, type=np.array,
                        help='The starting state. (default: None)')
    parser.add_argument('--friction', default=DEFAULT_FRICTION, type=bool,
                        help='Use friction. (default: True)')
    parser.add_argument('-n', '--name', default=DEFAULT_NAME, type=str,
                        help='The name/path of the agent. (default: results/success_model.zip')

    ARGS = parser.parse_args()
    run(**vars(ARGS))
