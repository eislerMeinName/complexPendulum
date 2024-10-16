from scipy.io import savemat
import numpy as np
import torch
from stable_baselines3 import SAC, PPO
from sympy.physics.vector.printing import params

from complexPendulum.agents import LQAgent
from complexPendulum.envs import ComplexPendulum, GainPendulum, DirectPendulum, BaselineGainPendulum


class NeuralAgent:
    """Neural Agent class."""

    def __init__(self, agentDict: dict = None, K: np.array = None) -> None:
        """Initialization.

        Input:
            agentDict: dict
                A dict with important agent parameters. Needs either Path and Algo or Agent (the direct sb3 model).
            K: np.array
                Potential baseline gains.
        """

        self.algo = None
        self.K = K

        if "Path" in agentDict.keys() and "Algo" in agentDict.keys():
            self.algo = agentDict["Algo"]
            match agentDict["Algo"]:
                case "SAC":
                    model = SAC.load(agentDict["path"])
                case "PPO":
                    model = PPO.load(agentDict["path"])
        elif "Agent" in agentDict.keys():
            model = agentDict["Agent"]
            self.algo = "SAC" if type(model) == SAC else "PPO"
        else:
            raise Exception("Dictionary has not the needed fields for init.")
        if not "Action" in agentDict.keys():
            raise Exception("Dictionary has not the needed fields for init.")
        if agentDict["Action"] == "Baseline" and K is None:
            raise Exception("You need to specify the baselined controller. K is None")

        self.AS = agentDict["Action"]
        self.agentDict = agentDict
        self.loadSAC(model) if self.algo == "SAC" else self.loadPPO(model)
        if torch.cuda.is_available():
            self.policy = self.policy.to('cpu')
            self.mu = self.mu.to('cpu')

    def loadSAC(self, model) -> None:
        self.policy = model.policy.actor.latent_pi
        self.mu = model.policy.actor.mu

    def loadPPO(self, model) -> None:
        self.policy = model.policy.mlp_extractor.policy_net
        self.mu = model.policy.action_net

    def predict(self, state: np.array) -> np.array:
        """Method to samples the action of the policy.
        Input:
            state: np.array
                The state.

        Return:
            action: np.array
                The action.
        """
        
        with torch.no_grad():
            s = torch.from_numpy(state).to(torch.float32)
            p = self.policy(s)
            mu = self.mu(p)
            a = torch.tanh(mu).numpy()
            if self.AS == "Direct":
                return a/2
            elif self.AS == "Gain":
                return (a - 1) * 50
            elif self.AS == "Base":
                return self.K + a * self.K

            raise Exception("No matching actionspace.")


    def saveMat(self, path: str) -> None:
        """Method that saves the parameters of the NeuralAgent to a .mat file. Allows import in Matlab/ Simulink model.
        Input:
            path: str
                The path of the .mat file.
        """

        mdic = {}
        mdic.update(self.policy.state_dict())
        mdic.update(self.mu.state_dict())
        mdicS = {}
        for k in mdic.keys():
            key = k[2:] + k[0] if k[0].isdigit() else k
            val = mdic[k]
            mdicS[key] = val.numpy() if type(val) is torch.Tensor else val
        mdicS["Algo"] = self.algo
        savemat(path, mdicS)

    def __str__(self) -> str:
        """Method constructing a string representation of the Neural Agent."""

        dictstr =  ''.join(["\n" + str(k) + ": " + str(v) for k, v in self.agentDict.items()])
        return f"NeuralAgent:{dictstr}\n{self.policy}\n{self.mu}"
