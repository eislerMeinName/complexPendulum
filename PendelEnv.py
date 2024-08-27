import gymnasium as gym
import numpy as np
import gymnasium.spaces as spaces
from typing import Tuple, Any
import pygame
from gymnasium.core import ObsType
from pygame import gfxdraw
import xml.etree.ElementTree as etxml
import control as ct

from assets.Logger import Logger
from assets.EnvTypes import ActionType, RewardType


class ComplexPendulum(gym.Env):
    """A basic RL environment that models a classic pendulum in different ways"""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(self,
                 frequency: float, 
                 episode_len: float,
                 path: str, 
                 Q: np.ndarray,
                 R: np.ndarray,
                 gui: bool = False,
                 actiontype: ActionType = ActionType.DIRECT,
                 rewardtype: RewardType = RewardType.LQ,
                 s0: np.array = None,
                 friction: bool = True,
                 log: bool = True
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
        self.solvesteps = 10
        self.deltat = 1 / frequency / self.solvesteps

        self.state = self.sampleS0() if s0 is None else s0.copy()
        self.s0 = None if s0 is None else s0.copy()

        self.actiontype = actiontype
        self.rewardtype = rewardtype
        if actiontype == ActionType.DIRECT:
            self.action_space: spaces = spaces.Box(low=-0.5*np.ones(1),
                                                   high=0.5*np.ones(1))
        else:
            self.action_space: spaces = spaces.Box(low=-50*np.ones(4),
                                                   high=np.zeros(4))

        self.observation_space: spaces = spaces.Box(low=-np.array([-1, -5, -np.pi, -15]),
                                                    high=np.array([1, 5, np.pi, 15]))
        
        self.render_mode = "human"
        self.k = 100
        self.gui = gui
        self.log = log
        if self.gui:
            self.screen = None
            self.screen_dim = 500
        if self.log:
            self.logger = Logger(self.actiontype, self.state)

    def sampleS0(self) -> np.array:
        """Samples a random starting state with random X and θ."""

        return np.array([np.random.rand()-0.5, 0, 2*np.pi*(np.random.rand()-0.5), 0], dtype=np.float32)

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
        constraint = -100 if abs(self.state[0] > 1) else 0
        #cost = (state@self.Q@state.T + u*self.R*u)[0, 0]
        lq = -(state @ self.Q @ state.T + u * self.R * u)[0, 0] / self.k + constraint
        #state[0, 1] = 0
        #state[0, 3] = 0
        #dist = np.linalg.norm(state)
        expo = np.exp(-np.linalg.norm(self.Q @ state.T)) + np.exp(- np.linalg.norm(self.R * u)) - 2
        return lq if self.rewardtype is RewardType.LQ else expo

    def done(self) -> bool:
        """Checks if simulation is finished.
        Return:
            done: bool
                Simulation is finished.
        """

        if self.STEPS == 0 or abs(self.state[0]) > 1:
            return True
        self.STEPS -= 1
        return False

    def intStep(self, F: float) -> None:
        """A single integration step of the nonlinear system dynamics.
        Input:
            F: float
                The applied force u.
        """

        x_dot_s = self.state[1]
        theta_s = self.state[2]
        theta_dot_s = self.state[3]
        mp, l, J, m, fp, fc, g, _, _ = self.params

        d = 1 - (mp**2 * l**2 / (J * m)) * np.cos(theta_s)**2

        x_dot = x_dot_s
        x_ddot = F/m - mp*mp*l*l*g*np.sin(theta_s)*np.cos(theta_s) / (J*m) + mp*l*theta_dot_s*theta_dot_s*np.sin(theta_s)/m + mp*l*fp*theta_dot_s*np.cos(theta_s) / (J*m) - fc*x_dot_s/m
        theta_dot = theta_dot_s
        theta_ddot = mp*l*g*np.sin(theta_s)/J - mp*l*F*np.cos(theta_s) / (J*m) - mp*mp*l*l*theta_dot_s*theta_dot_s*np.sin(theta_s)*np.cos(theta_s) / (J*m) + mp*l*fc*x_dot_s*np.cos(theta_s) / (J*m) - fp*theta_dot_s/J
        self.state += self.deltat * np.array([x_dot, x_ddot/d, theta_dot, theta_ddot/d])

    def step(self, action: np.array) -> Tuple[np.array, float, bool, dict]:
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

        if self.gui:
            self.render()

        pwm, u = self.preprocessAction(action)
        for _ in range(self.solvesteps):
            self.intStep(u)

        self.time += 1/self.frequency
        if self.state[2] > np.pi:
            self.state[2] -= 2*np.pi
        elif self.state[2] < -np.pi:
            self.state[2] += 2*np.pi

        a = action if self.actiontype is ActionType.GAIN else action[0]

        rew = self.reward(u)

        if self.log:
            self.logger.log(self.time, self.state, pwm, u, a, rew)

        return self.state, rew, self.done(), {"answer": 42}

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
        a = a[0] if self.actiontype == ActionType.DIRECT else -(a.reshape(1, -1)@state.T)[0, 0]


        a_fric = a + np.sign(a) * self.params[8]
        pwm = a_fric / self.params[7]

        pwm = np.clip(pwm, -0.5, 0.5)

        #Rail Limmiter ???
        lim1 = 1 if self.state[0] >= 0.75 else 0
        lim2 = 1 if self.state[0] <= 0.75 else 0
        lim = lim1 * 3 * (self.state[0] - 0.75) + lim2 * 3 * (self.state[0] + 0.75)
        #print(lim)

        #pwm = np.clip(pwm - lim, -0.5, 0.5)

        #pwm to effecitve force

        force = pwm * self.params[7]
        if abs(force) < self.params[8] and (self.state[1] != 0 or self.state[3] != 0):
            force = 0

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
                self.screen = pygame.display.set_mode(size=
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))
        self.pygamedrawPend()
        font = pygame.font.SysFont("comicsansms", 50)

        text = font.render(str(int(self.time * 100)/100) + "s/" + str(self.episode_len) + "s", True, (0, 0, 0))
        self.screen.blit(self.surf, (0, 0))
        self.screen.blit(text, (0, 0))

        pygame.display.flip()

    def pygamedrawPend(self) -> None:
        """Draw the pendulum in pygame screen."""

        size = [50, 10, 100]
        scale = (self.screen_dim - 200) / 2
        OffsetX = self.state[0] * scale + self.screen_dim/2
        OffsetY = self.screen_dim/2

        pygame.draw.rect(self.surf, (0, 0, 200), (OffsetX - size[0]/2, OffsetY - size[1]/2, size[0], size[1]), 0)

        rodEndY = OffsetY - size[2] * np.cos(self.state[2])
        rodEndX = OffsetX + size[2] * np.sin(self.state[2])
        gfxdraw.filled_circle(self.surf, int(rodEndX), int(rodEndY),
                              int(size[1] / 2), (204, 77, 77))

        l, r, t, b = 0, size[2], size[1] / 2, -size[1] / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[2] - np.pi/2)
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

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None,) -> tuple[ObsType, dict[str, Any]]:
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

        return self.state

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


if __name__ == "__main__":
    Q = np.array([[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 1000, 0], [0, 0, 0, 0.1]])
    env = ComplexPendulum(48, 2, "params.xml", Q, np.eye(1), True, rewardtype=RewardType.EXP)
    #env2 = ComplexPendulum(48, 5, "params.xml", Q, np.eye(1), True, rewardtype=RewardType.EXP, s0=np.pi / 8)
    done = False

    while not done:
        state, rew, done, _ = env.step(np.array([0]))

    env.close()
    env.stats()
