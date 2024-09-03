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
            "depth_array",
        ],
        "render_fps": 25,
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
                 s0: np.array = None,
                 friction: bool = True,
                 log: bool = True,
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
            s0: np.array = None
                The starting state indicating random sampling or fixed starting state.
            friction: bool = True
                Use friction during simulation.
            log: bool = True
                Use logger.
        """

        self.frequency = frequency
        self.episode_len = episode_len
        self.STEPS = episode_len * frequency
        self.time = 0

        self.friction = friction
        self.params = self._parseXML(path)

        self.Q = Q
        self.R = R
        self.t = np.arange(0, 1 / frequency, 1e-5)

        self.state = self.sampleS0() if s0 is None else s0.copy()
        self.s0 = None if s0 is None else s0.copy()

        self.actiontype = actiontype
        self.rewardtype = rewardtype
        if actiontype == ActionType.DIRECT:
            self.action_space: spaces = spaces.Box(low=-0.5*np.ones(1),
                                                   high=0.5*np.ones(1), dtype=np.float32)
        else:
            self.action_space: spaces = spaces.Box(low=-50 * np.ones(4),
                                                   high=np.zeros(4), dtype=np.float32)

        self.observation_space: spaces = spaces.Box(low=np.array([-1, -np.inf, -np.pi, -np.inf]),
                                                    high=np.array([1, np.inf, np.pi, np.inf]), dtype=np.float32)

        self.render_mode = "human"
        self.gui = gui
        self.log = log
        if self.gui:
            self.surf = None
            self.screen = None
            self.screen_dim = 500
            self.render()
        if self.log:
            self.logger = Logger(self.actiontype, self.state)

    @staticmethod
    def sampleS0() -> np.array:
        """Samples a random starting state with random X and θ."""

        return np.array([np.random.rand() - 0.5, 0, 2 * np.pi * (np.random.rand() - 0.5), 0], dtype=np.float32)

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
        constraint = -1000 if abs(self.state[0] > 1) else 0
        if self.rewardtype is RewardType.LQ:
            return -(state @ self.Q @ state.T + u * self.R * u)[0, 0] / self.frequency + constraint
        elif self.rewardtype is RewardType.EXP:
            return np.exp(-np.linalg.norm(self.Q @ state.T)) + np.exp(- np.linalg.norm(self.R * u)) - 2 + constraint
        else:
            return -(np.linalg.norm(self.Q @ state.T) + np.linalg.norm(self.R * u))

    def done(self) -> tuple[bool, bool]:
        """Checks if simulation is finished.
        Return:
            done: bool
                Simulation is finished.
        """

        if self.STEPS == 0:
            return True, False
        if abs(self.state[0]) > 1:
            return True, True
        self.STEPS -= 1
        return False, False

    def intStepOde(self, state: np.array, t: float, F: float) -> np.array:
        """A single integration step of the nonlinear system dynamics.
        Input:
            x: np.array
                The current state.
            t: float
                The current time.
            F: float
                The applied force u.

        Return:
            statedot: np.array
                The differential of the state.
        """

        x_dot_s = state[1]
        theta_s = state[2]
        theta_dot_s = state[3]

        mp, l, J, m, fp, fc, g, _, _ = self.params

        d = 1 - (mp ** 2 * l ** 2 / (J * m)) * np.cos(theta_s) ** 2

        x_dot = x_dot_s
        x_ddot = F / m - mp * mp * l * l * g * np.sin(theta_s) * np.cos(theta_s) / (
                    J * m) + mp * l * theta_dot_s * theta_dot_s * np.sin(
            theta_s) / m + mp * l * fp * theta_dot_s * np.cos(theta_s) / (J * m) - fc * x_dot_s / m
        theta_dot = theta_dot_s
        theta_ddot = mp * l * g * np.sin(theta_s) / J - mp * l * F * np.cos(theta_s) / (
                    J * m) - mp * mp * l * l * theta_dot_s * theta_dot_s * np.sin(theta_s) * np.cos(theta_s) / (
                                 J * m) + mp * l * fc * x_dot_s * np.cos(theta_s) / (J * m) - fp * theta_dot_s / J
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

        pwm, u = self.preprocessAction(action)
        s = odeint(self.intStepOde, y0=self.state, t=self.t, args=(u,))
        self.state = np.array(s[-1], dtype=np.float32)

        self.time += 1 / self.frequency
        if self.state[2] > np.pi:
            self.state[2] -= 2 * np.pi
        elif self.state[2] < -np.pi:
            self.state[2] += 2 * np.pi

        rew = self.reward(u)
        done, trun = self.done()
        if self.gui:
            self.render()

        if self.log:
            a = action if self.actiontype is ActionType.GAIN else action[0]
            self.logger.log(self.time, self.state, pwm, u, a, rew)

        return self.state, rew, done, trun, {"answer": 42}

    def preprocessAction(self, a: np.array) -> tuple:
        """Preprocesses the action based on the actiontype.
        Input:
            a: np.array
                The action.

        Return:
            pwm: float
                The action transformed to pwm.
            force: float
                The applied force u.
        """

        state = self.state.reshape(1, -1).copy()
        a = a[0] if self.actiontype == ActionType.DIRECT else -(a.reshape(1, -1) @ state.T)[0, 0]

        a_fric = a + np.sign(a) * self.params[8]
        pwm = a_fric / self.params[7]

        pwm = np.clip(pwm, -0.5, 0.5)

        # pwm to effective force
        force = pwm * self.params[7]
        if self.state[1] != 0 or self.state[3] != 0:
            if abs(force) < self.params[8]:
                force = 0
            elif force > -self.params[8]:
                force = force - self.params[8]
            else:
                force = force + self.params[8]

        return pwm, force - np.sign(self.state[1]) * self.params[8]

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

        pygame.display.flip()

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
            mp: float
                The mass of the pendulum.
            l: float
                The distance between the mounting point and the center of mass of the pendulum.
            J: float
                The moment of inertia.
            m: float
                The total mass.
            fp: float
                The friction coefficient for the pendulum.
            fc: float
                The friction coefficient for the cart.
            g: float
                The gravitational acceleration.
            M: float
                PWM to effective force coefficient.
            Fs: float
                The static friction.
        """

        XML_TREE = etxml.parse(path).getroot()
        Fs = float(XML_TREE.attrib['Fs']) if self.friction else 0
        return float(XML_TREE.attrib['mp']), \
            float(XML_TREE.attrib['l']), \
            float(XML_TREE.attrib['J']), \
            float(XML_TREE.attrib['m']), \
            float(XML_TREE.attrib['fp']), \
            float(XML_TREE.attrib['fc']), \
            float(XML_TREE.attrib['g']), \
            float(XML_TREE.attrib['M']), \
            Fs

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

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None, ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment.
        Return:
            state: np.array
                The state.
        """

        self.STEPS = self.episode_len * self.frequency
        self.time = 0
        self.state = self.sampleS0() if self.s0 is None else self.s0.copy()

        if self.gui:
            self.screen = None
            self.logger.reset(self.state)

        return self.state, {'answer': 42}

    def getLinearSS(self) -> ct.ss:
        """Creates the linearized state space model of the pendulum with parsed parameters.
        Return:
            ss: ct.ss
                The state space model.
        """

        mp, l, J, m, fp, fc, g, _, _ = self.params
        k = 1 / (1 - (mp * mp * l * l) / (J * m))

        A = [[0, 1, 0, 0],
             [0, -k * fc / m, -k * g * (mp * mp * l * l) / (J * m), k * fp * (mp * l) / (J * m)],
             [0, 0, 0, 1],
             [0, k * fc * (mp * l) / (J * m), k * g * (mp * l) / J, -k * fp / J]]

        B = [[0],
             [k / m],
             [0],
             [-k * (mp * l) / (J * m)]]

        C = [[1, 0, 0, 0],
             [0, 0, 1, 0]]
        return ct.ss(A, B, C, np.zeros((2, 1)))
