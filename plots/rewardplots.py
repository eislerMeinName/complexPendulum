import numpy as np
import matplotlib.pyplot as plt

class RewardPlotter:

    def __init__(self, Q: list, R: list, k: float) -> None:
        self.Q = Q
        self.R = R
        self.k = k

    def LQRreturn(self, state: np.array, u: float, Q: np.array, R: np.array):
        return -(state @ Q @ state.T + u * R * u)[0, 0] / self.k

    def EXPreturn(self, state, u, Q, R):
        return np.exp(- np.linalg.norm(Q@state)) + np.exp(- np.linalg.norm(R*u)) - 2

    def plotSingleSurf(self, ax, title, func, l) -> None:
        x = np.linspace(-1, 1, 100)
        theta = np.linspace(-np.pi, np.pi, 100)
        x, theta = np.meshgrid(x, theta)
        rewards = np.zeros(x.shape)
        for i in range(0, 100):
            for j in range(0, 100):
                state = np.array([x[i, j], 0, theta[i, j], 0])
                rewards[i, j] = func(state, 0, self.Q[l], self.R[l])

        ax.set_xlabel('$X$', fontsize=20)
        ax.set_ylabel(r'$\theta$', fontsize=20)
        ax.set_zlabel('reward', fontsize=20)
        ax.plot_surface(x, theta, np.array(rewards))
        ax.set_title(title)

    def plotLQRPlanes(self, fig, axs) -> None:
        x = np.linspace(-1, 1, 100)
        theta = np.linspace(-np.pi, np.pi, 100)
        x, theta = np.meshgrid(x, theta)
        print(x.shape)
        rewards = np.zeros((100, 100))
        for i in range(0, 100):
            for j in range(0, 100):
                state = np.array([x[i, j], 0, theta[i, j], 0])
                rewards[i, j] = self.LQRreturn(state, 0, self.Q[0], self.R[0])

        axs[0].set_xlabel('$X$', fontsize=20, rotation=150)
        axs[0].set_ylabel('$Y$')
        axs[0].set_zlabel(r'$\gamma$', fontsize=30, rotation=60)
        axs[0].plot_surface(x, theta, np.array(rewards))
        axs[0].set_title("LQ reward 1")

        rewards = np.zeros((100, 100))
        for i in range(0, 100):
            for j in range(0, 100):
                state = np.array([x[i, j], 0, theta[i, j], 0])
                rewards[i, j] = self.LQRreturn(state, 0, self.Q[1], self.R[1])

        axs[1].plot_surface(x, theta, np.array(rewards))
        axs[1].set_title("LQ reward 2")

        rewards = np.zeros((100, 100))
        for i in range(0, 100):
            for j in range(0, 100):
                state = np.array([x[i, j], 0, theta[i, j], 0])
                rewards[i, j] = self.LQRreturn(state, 0, self.Q[2], self.R[2])

        axs[2].plot_surface(x, theta, np.array(rewards))
        axs[2].set_title("LQ reward 3")

    def plotEXPPLanes(self, fig, axs) -> None:
        x = np.linspace(-1, 1, 100)
        theta = np.linspace(-np.pi, np.pi, 100)
        x, theta = np.meshgrid(x, theta)
        print(x.shape)
        rewards = np.zeros((100, 100))
        for i in range(0, 100):
            for j in range(0, 100):
                state = np.array([x[i, j], 0, theta[i, j], 0])
                rewards[i, j] = self.EXPreturn(state, 0, self.Q[0], self.R[0])

        axs[0].plot_surface(x, theta, np.array(rewards))
        axs[0].set_title("EXP reward 1")

        rewards = np.zeros((100, 100))
        for i in range(0, 100):
            for j in range(0, 100):
                state = np.array([x[i, j], 0, theta[i, j], 0])
                rewards[i, j] = self.EXPreturn(state, 0, self.Q[1], self.R[1])

        axs[1].plot_surface(x, theta, np.array(rewards))
        axs[1].set_title("EXP reward 2")

        rewards = np.zeros((100, 100))
        for i in range(0, 100):
            for j in range(0, 100):
                state = np.array([x[i, j], 0, theta[i, j], 0])
                rewards[i, j] = self.EXPreturn(state, 0, self.Q[2], self.R[2])

        axs[2].plot_surface(x, theta, np.array(rewards))
        axs[2].set_title("EXP reward 3")

if __name__ == "__main__":
    q1 = np.eye(4)
    q1[2, 2] = 0.5
    q2 = np.eye(4)
    q2[2, 2] = 0.1
    Q = [np.eye(4), q1, q2]
    plotter = RewardPlotter(Q, [np.eye(1), np.eye(1), np.eye(1)], 2)
    #fig, axs = plt.subplots(2, 3, subplot_kw={'projection': '3d'})
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    plotter.plotSingleSurf(ax,"", plotter.EXPreturn, 2)
    plt.show()