from complexPendulum.agents import NeuralAgent
from complexPendulum.agents.NeuralAgent import NeuralAgent
from complexPendulum.agents.neuralAgents import nAgent1, nAgent2, nAgent3, nAgent4

if __name__ == "__main__":
    agent = NeuralAgent(nAgent3)
    print(agent)
    agent.saveMat("Setup2PPODirect.mat")
