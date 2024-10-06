from stable_baselines3 import SAC, PPO

nAgent1 = {"Setup" : 1, "Algo": "SAC", "Action": "Gain", "Constrained": False, "Agent": SAC.load("complexPendulum/agents/neuralAgents/Setup1SacGain.zip")}
nAgent2 = {"Setup" : 2, "Algo": "SAC", "Action": "Gain", "Constrained": True, "Agent": SAC.load("complexPendulum/agents/neuralAgents/Setup2SacGainConstrain.zip")}
