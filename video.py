import imageio
import numpy as np
from stable_baselines3 import SAC
import gymnasium as gym

from complexPendulum.agents import SwingUpAgent, CombinedAgent
from complexPendulum.agents.NeuralAgent import NeuralAgent
from complexPendulum.agents.neuralAgents import nAgent2
from complexPendulum.assets import Setup1 as setup

DEFAULT_NAME: str = "results/best_model"
DEFAULT_ENV = "complexPendulum-v0"


def run(name: str = DEFAULT_NAME) -> None:
    env = gym.make(DEFAULT_ENV, frequency=100,
                   episode_len=15, path="params.xml",
                   Q=setup.Q, R=setup.R,
                   rewardtype=setup.func, s0=np.array([0, 0, np.pi + 0.01, 0]), gui=True,
                   friction=True, log=False, render_mode="rgb_array")

    neural = NeuralAgent(nAgent2)
    swingup = SwingUpAgent(env.unwrapped)
    agent = CombinedAgent(swingup, neural)

    images = []
    obs, _ = env.reset()
    img = env.render()
    for i in range(1501):
        images.append(img)
        action = agent.predict(obs)
        obs, _, _, _, _ = env.step(action)
        img = env.render()

    video_name: str = "logs/firstModel"
    imageio.mimsave(video_name + ".gif", [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=100)
    imageio.mimsave(video_name + ".mp4", [np.array(img) for i, img in enumerate(images) ], fps=100)

    env.close()

if __name__ == "__main__":
    run()