import numpy as np
import gymnasium as gym
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO

from complexPendulum.agents import ProportionalAgent, LQAgent
from complexPendulum.agents.NeuralAgent import NeuralAgent
from complexPendulum.agents.neuralAgents import *
from complexPendulum.assets import Setup1, Setup2, Setup3, Evaluator, ActionType, EvaluationDataType
from complexPendulum.envs import ComplexPendulum
from complexPendulum.plots.s0 import calcS0

def run(amount: int = 200, agent: NeuralAgent | ProportionalAgent = NeuralAgent(DirectQR1, None)) -> None:

    evaluationData = {"Sucess": [],
                      "LQR 1": [],
                      "LQR 2": [],
                      "LQR 3": [],
                      "pwm": [],
                      "pwm_max": [],
                      "pwm_min": [],
                      "pwm_d": [],
                      "pwm_dmax": [],
                      "xTr": [],
                      "xTm": [],
                      "xdh": [],
                      "xTe": [],
                      "xeinf": [],
                      "tTr": [],
                      "tTm": [],
                      "tdh": [],
                      "tTe": [],
                      "teinf": []}

    failX = []
    failT = []
    failtimes = []
    
    #swingup s0
    path: str = "complexPendulum/data/s0/s0.xlsx"
    df = pd.read_excel(path, None)
    vals = calcS0(df)
    def sampleS0Swing() -> np.array:
        c = np.random.choice([0, 1])
        sign = np.array([-1, -1, -1, 1]) if c == 0 else np.array([1, 1, 1, -1])
        s0 = np.array([np.random.normal(vals['x'][0], vals['x'][1]),
                       np.random.normal(vals['xd'][0], vals['xd'][1]),
                       np.random.normal(vals['t'][0], vals['t'][1]),
                       np.random.normal(vals['td'][0], vals['td'][1])])
        s0 = np.multiply(sign, s0)
        return s0

    env = gym.make('complexPendulum-v0', gui=False, s0=None, friction=True,
                   episode_len=10, actiontype=ActionType.GAIN, log=True,
                   conditionReward=True)

    #env.unwrapped.sampleS0 = sampleS0Swing
    
    setups = [Setup1, Setup2, Setup3]

    for i in tqdm(range(amount)):

        state, _ = env.reset()
        term, trun = False, False
        while not term and not trun:
            action = agent.predict(state)
            state, rew, term, trun, _ = env.step(action)

        if term:
            evaluationData["Sucess"].append(0)
            failX.append(env.unwrapped.logger.X[0])
            failT.append(env.unwrapped.logger.Theta[0])
            failtimes.append(env.unwrapped.time)

        else:
            evaluationData["Sucess"].append(1)
            env.unwrapped.logger.write('test.csv')
            eval = Evaluator("test.csv", setups=setups, datatype=EvaluationDataType.STEP, epsilon=(0.03, np.pi / 1024))
            evaluationData["LQR 1"].append(eval.data["LQR 1"])
            evaluationData["LQR 2"].append(eval.data["LQR 2"])
            evaluationData["LQR 3"].append(eval.data["LQR 3"])
            evaluationData["pwm"].append(eval.data["pwm"])
            evaluationData["pwm_max"].append(eval.data["max pwm"])
            evaluationData["pwm_min"].append(eval.data["min pwm"])
            evaluationData["pwm_d"].append(eval.data["pwm delta"])
            evaluationData["pwm_dmax"].append(eval.data["pwm delta max"])

            if eval.data["X: Tr"] is not None: evaluationData["xTr"].append((eval.data["X: Tr"]))
            if eval.data["X: Tm"] is not None: evaluationData["xTm"].append(eval.data["X: Tm"])
            if eval.data["X: Δh"] is not None: evaluationData["xdh"].append(eval.data["X: Δh"])
            if eval.data["X: Te"] is not None: evaluationData["xTe"].append(eval.data["X: Te"])
            if eval.data["X: e∞"] is not None: evaluationData["xeinf"].append(eval.data["X: e∞"])

            if eval.data["θ: Tr"] is not None: evaluationData["tTr"].append((eval.data["θ: Tr"]))
            if eval.data["θ: Tm"] is not None: evaluationData["tTm"].append(eval.data["θ: Tm"])
            if eval.data["θ: Δh"] is not None: evaluationData["tdh"].append(eval.data["θ: Δh"])
            if eval.data["θ: Te"] is not None: evaluationData["tTe"].append(eval.data["θ: Te"])
            if eval.data["θ: e∞"] is not None: evaluationData["teinf"].append(eval.data["θ: e∞"])

    for k in evaluationData.keys():
        if len(evaluationData[k]) != 0:
            print(f"{k}: {np.mean(evaluationData[k])} ± {np.std(evaluationData[k])}")

    print(failX)
    print(failT)
    if not len(failX) == 0:
        plt.scatter(failX, failT, c=failtimes)
        plt.xlim(-0.5, 0.5)
        plt.ylim(-0.3, 0.3)
        plt.colorbar()
        plt.show()



if __name__ == "__main__":
    agent = LQAgent(ComplexPendulum(Q=Setup1.Q, R=Setup1.R))
    #print(agent.K)
    #agent = NeuralAgent({"Agent": PPO.load("results/best_model"), "Action": "Direct"}, None)
    #agent = NeuralAgent(DirectQR2UnCon_best, None)
    run(100, agent)

