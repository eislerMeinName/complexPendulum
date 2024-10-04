from scipy.io import savemat
import numpy as np
import torch
from stable_baselines3 import SAC, PPO


class NeuralAgent:
    """Neural Agent class."""

    def __init__(self, agentDict: dict = None) -> None:
        """Initialization.

        Input:
            agentDict: dict
                A dict with important agent parameters. Needs either Path and Algo or Agent (the direct sb3 model).
        """

        if "Path" in agentDict.keys() and "Algo" in agentDict.keys():
            match agentDict["Algo"]:
                case "SAC":
                    model = SAC.load(agentDict["path"])
                case "PPO":
                    model = SAC.load(agentDict["path"])
        elif "Agent" in agentDict.keys():
            model = agentDict["Agent"]
        else:
            raise Exception("Dictionary has not the needed fields for init.")

        self.agentDict = agentDict
        self.policy = model.policy.actor.latent_pi
        self.mu = model.policy.actor.mu

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
            p = self.policy(torch.from_numpy(state).to(torch.float32))
            mu = self.mu(p)
            a = torch.tanh(mu).numpy()
            return (a - 1) * 50 if a.size == 4 else a/2


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
        mdicS["Algo"] = self.agentDict["Algo"]
        savemat(path, mdicS)

    def __str__(self) -> str:
        """Method constructing a string representation of the Neural Agent."""

        dictstr =  ''.join(["\n" + str(k) + ": " + str(v) for k, v in self.agentDict.items()])
        return f"NeuralAgent:{dictstr}\n{self.policy}\n{self.mu}"

