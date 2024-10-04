import imageio
import numpy as np
from stable_baselines3 import SAC
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from torch.backends.cudnn import deterministic
from complexPendulum.assets import Setup1 as setup
import moviepy.editor as mp


from learn import DEFAULT_NAME

DEFAULT_NAME: str = "results/best_model.zip"
DEFAULT_ENV = "gainPendulum-v0"


def run(name: str = DEFAULT_NAME) -> None:
    model = SAC.load(name)
    env = gym.make(DEFAULT_ENV, frequency=100,
                   episode_len=15, path="params.xml",
                   Q=setup.Q, R=setup.R,
                   rewardtype=setup.func, s0=None, gui=True,
                   friction=True, log=False, render_mode="rgb_array")

    images = []
    obs, _ = env.reset()
    img = env.render()
    for i in range(1501):
        images.append(img)
        action, _ = model.predict(obs)
        obs, _, _, _, _ = env.step(action)
        img = env.render()

    video_name: str = "logs/firstModel"
    imageio.mimsave(video_name + ".gif", [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=100)
    clip = mp.VideoFileClip(video_name + ".gif")
    clip.write_videofile(video_name + ".mp4")

    env.close()

if __name__ == "__main__":
    run()