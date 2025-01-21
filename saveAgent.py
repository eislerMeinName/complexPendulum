from complexPendulum.agents import NeuralAgent
from complexPendulum.agents.NeuralAgent import NeuralAgent
from complexPendulum.agents.neuralAgents import *

if __name__ == "__main__":
    agent = NeuralAgent(DirectQR3, None)
    print(agent)
    agent.saveMat("DirectQR3.mat")
