import gymnasium as gym
import numpy as np
import gymnasium.spaces as spaces
from typing import Tuple, Any
import pygame
from gymnasium.core import ObsType
from pygame import gfxdraw
import xml.etree.ElementTree as etxml
import control as ct
from scipy.integrate import odeint


from complexPendulum.assets import Logger, ActionType, RewardType


class ComplexPendulum(gym.Env):
    """A basic RL environment that models a classic pendulum in different types."""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
        "render_fps": 100,
    }

    def __init__(self,
                 frequency: float = 100,
                 episode_len: float = 15,
                 path: str = 'params.xml',
                 Q: np.ndarray = np.eye(4),
                 R: np.ndarray = np.eye(1),
                 gui: bool = False,
                 actiontype: ActionType = ActionType.DIRECT,
                 rewardtype: RewardType = RewardType.LQ,
                 conditionReward: bool = True,
                 s0: np.array = None,
                 friction: bool = True,
                 log: bool = True,
                 render_mode: str = "human"
                 ) -> None:

        """
        Initialization.
        Input:
            frequency: float
                The control frequency.
            episode_len: float
                The length of the episode in seconds.
            path: str
                The path to the parameter file.
            Q: np.ndarray
                The Q array for the reward function.
            R: np.ndarray
                The R array for the reward function.
            gui: bool = False
                Render the gui.
            actiontype: ActionType = ActionType.Direct
                The Action Space type.
            rewardtype: RewardType = RewardType.LQ
                The type of the reward function.
            conditionReward: bool = False
                Use conditioned rewawrd.
            s0: np.array = None
                The starting state indicating random sampling or fixed starting state.
            friction: bool = True
                Use friction during simulation.
            log: bool = True
                Use logger.
            render_mode: str = human
                Render mode.
        """

        self.frequency = frequency
        self.episode_len = episode_len
        self.STEPS = episode_len * frequency
        self.time = 0

        self.friction = friction
        self.params = self._parseXML(path)

        self.Q = Q
        self.R = R
        self.t = np.linspace(0, 1/frequency, 2)
        self.conditionReward = conditionReward

        self.state = self.sampleS0() if s0 is None else s0.copy()
        self.s0 = None if s0 is None else s0.copy()

        pos: float = round(self.state[0] / self.params[8]) * self.params[8]
        angle: float = round(self.state[2] / self.params[9]) * self.params[9]
        self.last_state = np.array([pos, self.state[1], angle, self.state[3]])

        self.actiontype = actiontype
        self.rewardtype = rewardtype
        if actiontype == ActionType.DIRECT:
            self.action_space: spaces = spaces.Box(low=-np.ones(1),
                                                   high=np.ones(1), dtype=np.float32)
        else:
            self.action_space: spaces = spaces.Box(low=-np.ones(4),
                                                   high=np.ones(4), dtype=np.float32)

        self.observation_space: spaces = spaces.Box(low=np.array([-1, -np.inf, -np.pi, -np.inf]),
                                                    high=np.array([1, np.inf, np.pi, np.inf]), dtype=np.float64)

        self.render_mode = render_mode#"rgb_array" #"human"
        self.gui = gui
        self.log = log
        if self.gui:
            self.surf = None
            self.screen = None
            self.screen_dim = 500
            self.render()
        if self.log:
            self.logger = Logger(self.actiontype, self.last_state)

    @staticmethod
    def sampleS0() -> np.array:
        """Samples a random starting state with random X and θ."""
        pos : float = np.random.rand()*0.4 - 0.2
        vel: float = 0#np.random.rand()*2/3 - 1/3
        angle: float = np.random.rand()*0.5-0.25
        anglevel: float = 0#np.random.rand()*2/3 - 1/3
        #return np.array([np.random.rand()*0.4 - 0.2, 0, np.random.rand()*0.5-0.25, 0], dtype=np.float64)
        return np.array([pos, vel, angle, anglevel])

    def reward(self, u: float) -> float:
        """The reward function.
        Input:
            u: float
                The applied force.

        Return:
            reward: float
                The reward.
        """

        state = self.state.reshape(1, -1).copy()
        state[0, 2] = np.arctan2(np.sin(state[0, 2]), np.cos(state[0, 2]))
        if abs(state[0, 2]) > 0.3 or abs(self.state[0]) > 0.7 and self.conditionReward:
            return -1000
        elif self.rewardtype is RewardType.LQ:
            return -(state @ self.Q @ state.T + u * self.R * u)[0, 0]
        elif self.rewardtype is RewardType.EXP:
            return np.exp(-np.linalg.norm(self.Q @ state.T)) + np.exp(- np.linalg.norm(self.R * u)) - 2
        else:
            return -(np.linalg.norm(self.Q @ state.T) + np.linalg.norm(self.R * u))


    def done(self) -> tuple[bool, bool]:
        """Checks if simulation is finished.
        Return:
            termination: bool
                Simulation is finished due to condition.
            truncated: bool
                Simulation finished due to time.
        """

        if self.STEPS == 1:
            return False, True
        if self.conditionReward:
            angle = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))
            if abs(angle) > 0.3 or abs(self.state[0]) > 0.7:
                self.STEPS -= 1
                return True, False
        self.STEPS -= 1
        return False, False

    def intStepOde(self, state: np.array, t: float, force: float) -> np.array:
        """A single integration step of the nonlinear system dynamics.
        Input:
            x: np.array
                The current state.
            t: float
                The current time.
            force: float
                The applied force.

        Return:
            statedot: np.array
                The differential of the state.
        """

        x_dot_s = state[1]
        theta_s = state[2]
        theta_dot_s = state[3]
        F = force

        g, m, mc, mp, l, J, _, _, _, _, ep, ec, muc, mus= self.params

        d = 1 - (mp**2 * l**2 / (J * m)) * np.cos(theta_s)**2

        sinT = 0 if np.allclose(theta_s, np.pi) else np.sin(theta_s)
        cosT = -1 if np.allclose(theta_s, np.pi) else np.cos(theta_s)

        x_dot = x_dot_s
        x_ddot = F / m - mp * mp * l * l * g * sinT * cosT / (
                J * m) + mp * l * theta_dot_s * theta_dot_s * sinT / m + mp * l * ep * theta_dot_s * cosT / (J * m) - ec * x_dot_s / m
        theta_dot = theta_dot_s
        theta_ddot = mp * l * g * sinT / J - mp * l * F * cosT / (
                J * m) - mp * mp * l * l * theta_dot_s * theta_dot_s * sinT * cosT / (
                             J * m) + mp * l * ec * x_dot_s * cosT / (J * m) - ep * theta_dot_s / J

        return np.array([x_dot, x_ddot / d, theta_dot, theta_ddot / d])

    def step(self, action: np.array) -> Tuple[np.array, float, bool, bool, dict]:
        """The step function simulates a single control step in the environment.
        Input:
            action: np.array
                The action chosen by the agent.

        Return:
            state: np.array
                The current state.
            reward: float
                The current reward.
            done: bool
                Indicating if the environment is finished.
            info: dict
                The true answer.
        """

        pwm = self.preprocessAction(action)
        force = self.pwmToEffectiveForce(pwm)


        s = odeint(self.intStepOde, y0=self.state, t=self.t, args=(force, ))

        self.state = np.array(s[-1], dtype=np.float64)

        self.time += 1 / self.frequency

        rew = self.reward(force)
        term, trun = self.done()

        obs = self.observe()
        if self.gui:
            self.render()

        if self.log:
            a = action if self.actiontype is ActionType.GAIN else action[0]
            self.logger.log(self.time, obs, pwm, force, a, rew)

        return obs, rew, term, trun, {"answer": 42}

    def preprocessAction(self, a: np.array) -> float:
        """Preprocesses the action based on the actiontype.
        Input:
            a: np.array
                The action.

        Return:
            pwm: float
                The action transformed to pwm.
        """

        obs = self.last_state.reshape(1, -1).copy()
        a = a[0] if self.actiontype == ActionType.DIRECT else -(a.reshape(1, -1) @ obs.T)[0, 0]

        M0 = self.params[7]
        M1 = self.params[6]
        pwm = (a - M0) / M1

        pwm = np.clip(pwm, -0.5, 0.5)

        return pwm if self.actiontype == ActionType.GAIN else a

    def pwmToEffectiveForce(self, pwm: float) -> float:
        g, m , mc, mp, l, J, M1, M0, _, _, ep, ec, muc, mus = self.params
        deadzone = 0.001

        force = pwm * M1 + M0 if abs(pwm) > deadzone else 0

        Fstatatic = -force if abs(force) < mus else -mus * np.sign(force)
        Fcoulomb = -muc * (mc + mp) * g * np.sign(self.state[1])
        #if abs(self.state[1]) < 0.01:
        #    input(self.state[1])

        return force + Fstatatic if np.allclose(self.state[1], 0.0) else force + Fcoulomb

    def observe(self) -> np.array:
        """Returns the observation based on the current state.
        Returns:
            obs: np.array
                The observation.
        """

        pos: float = round(self.state[0] / self.params[8]) * self.params[8]
        angle: float = round(self.state[2] / self.params[9]) * self.params[9]
        vel: float = (pos - self.last_state[0]) * self.frequency
        angledif: float = 0 if np.allclose(angle, self.last_state[2]) else angle - self.last_state[2]
        anglevel: float = np.arctan2(np.sin(angledif), np.cos(angledif)) * self.frequency
        angle: float = np.arctan2(np.sin(angle), np.cos(angle))

        obs = np.array([pos, vel, angle, anglevel], dtype=np.float64)

        self.last_state = obs.copy()
        return obs

    def render(self):
        """Render the current state of the Pendulum as Pygame."""

        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(size=(self.screen_dim, self.screen_dim))
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))
        self.pygamedrawPend()
        font = pygame.font.SysFont("comicsansms", 50)

        text = font.render(str(int(self.time * 100) / 100) + "s/" + str(self.episode_len) + "s", True, (0, 0, 0))
        self.screen.blit(self.surf, (0, 0))
        self.screen.blit(text, (0, 0))

        if self.render_mode == "human":
            pygame.display.flip()
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def pygamedrawPend(self) -> None:
        """Draw the pendulum in pygame screen."""

        size = [50, 10, 100]
        scale = (self.screen_dim - 200) / 2
        OffsetX = self.state[0] * scale + self.screen_dim / 2
        OffsetY = self.screen_dim / 2

        pygame.draw.rect(self.surf, (0, 0, 200), (OffsetX - size[0] / 2, OffsetY - size[1] / 2, size[0], size[1]), 0)

        rodEndY = OffsetY - size[2] * np.cos(self.state[2])
        rodEndX = OffsetX + size[2] * np.sin(self.state[2])
        gfxdraw.filled_circle(self.surf, int(rodEndX), int(rodEndY),
                              int(size[1] / 2), (204, 77, 77))

        l, r, t, b = 0, size[2], size[1] / 2, -size[1] / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[2] - np.pi / 2)
            c = (c[0] + OffsetX, c[1] + OffsetY)
            transformed_coords.append(c)

        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_circle(self.surf, int(OffsetX), int(OffsetY), int(size[1] / 2), (190, 190, 190))

    def _parseXML(self, path: str) -> tuple:
        """Parses the parameters from xml file.
        Input:
            path: str
                The path to the xml parameter file.
        Returns:
            params: tuple of params
        """

        XML_TREE = etxml.parse(path).getroot()
        mus = float(XML_TREE.attrib['mus']) if self.friction else 0

        return float(XML_TREE.attrib['g']), \
            float(XML_TREE.attrib['m']), \
            float(XML_TREE.attrib['mc']), \
            float(XML_TREE.attrib['mp']), \
            float(XML_TREE.attrib['l']), \
            float(XML_TREE.attrib['J']), \
            float(XML_TREE.attrib['M1']), \
            float(XML_TREE.attrib['M0']), \
            float(XML_TREE.attrib['qx']), \
            float(XML_TREE.attrib['qt']), \
            float(XML_TREE.attrib['ep']), \
            float(XML_TREE.attrib['ec']), \
            float(XML_TREE.attrib['muc']), \
            mus

    def stats(self) -> None:
        """Shows the logged step response data."""

        if self.gui:
            pygame.quit()
        if self.log:
            self.logger.show()

    def close(self) -> None:
        """Closes the environment."""

        if self.gui:
            pygame.quit()

    def __str__(self) -> str:
        """Returns a string with condensed info of environment."""
        general: str = f"\nEnvironment: {self.__class__.__name__}\nObservationSpace: {self.observation_space} \nActionSpace: {self.action_space} \nActionType: {self.actiontype.__class__.__name__} \nRewardType: {self.rewardtype.__class__.__name__} \nParams: {self.params}"
        return general

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None, ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment.
        Return:
            state: np.array
                The state.
        """

        self.STEPS = self.episode_len * self.frequency
        self.time = 0
        self.state = self.sampleS0() if self.s0 is None else self.s0.copy()
        pos: float = round(self.state[0] / self.params[8]) * self.params[8]
        angle = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))
        angle: float = round(angle / self.params[9]) * self.params[9]
        self.last_state = np.array([pos, self.state[1], angle, self.state[3]], dtype=np.float64)

        if self.gui:
            self.screen = None
        if self.log:
            self.logger.reset(self.last_state)

        return self.last_state, {'answer': 42}

    def getLinearSS(self) -> ct.ss:
        """Creates the linearized state space model of the pendulum with parsed parameters.
        Return:
            ss: ct.ss
                The state space model.
        """

        g, m, mc, mp, l, J, M1, M0, _, _, ep, ec, muc, mus = self.params
        k = 1 / (1 - (mp * mp * l * l) / (J * m))

        A = [[0, 1, 0, 0],
             [0, -k * ec / m, -k * g * (mp * mp * l * l) / (J * m), k * ep * (mp * l) / (J * m)],
             [0, 0, 0, 1],
             [0, k * ec * (mp * l) / (J * m), k * g * (mp * l) / J, -k * ep / J]]

        B = [[0],
             [k / m],
             [0],
             [-k * (mp * l) / (J * m)]]

        C = [[1, 0, 0, 0],
             [0, 0, 1, 0]]

        return ct.ss(A, B, C, np.zeros((2, 1)))
