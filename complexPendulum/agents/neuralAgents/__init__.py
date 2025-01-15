from stable_baselines3 import SAC, PPO

#Direct Models
DirectQR1 = {"Setup": 1, "Algo": "PPO", "Action": "Direct", "Constrained": True, "Agent": PPO.load("complexPendulum/agents/neuralAgents/direct/QR1Direct_last.zip")}
#DirectQR1best = {"Setup": 1, "Algo": "PPO", "Action": "Direct", "Constrained": True, "Agent": PPO.load("complexPendulum/agents/neuralAgents/direct/QR1Direct_best.zip")}

#DirectQR2 = {"Setup": 2, "Algo": "PPO", "Action": "Direct", "Constrained": True, "Agent": PPO.load("complexPendulum/agents/neuralAgents/direct/QR2Direct_last.zip")}
DirectQR2 = {"Setup": 2, "Algo": "PPO", "Action": "Direct", "Constrained": True, "Agent": PPO.load("complexPendulum/agents/neuralAgents/direct/QR2Direct_best.zip")}


#DirectQR2_1m = {"Setup": 1, "Algo": "PPO", "Action": "Direct", "Constrained": True, "Agent": PPO.load("complexPendulum/agents/neuralAgents/direct/QR2Direct1m_last.zip")}
#DirectQR2_1m_best = {"Setup": 1, "Algo": "PPO", "Action": "Direct", "Constrained": True, "Agent": PPO.load("complexPendulum/agents/neuralAgents/direct/QR2Direct1m_best.zip")}

DirectQR3 = {"Setup": 3, "Algo": "PPO", "Action": "Direct", "Constrained": True, "Agent": PPO.load("complexPendulum/agents/neuralAgents/direct/QR3Direct_last.zip")}
DirectQR3_best = {"Setup": 3, "Algo": "PPO", "Action": "Direct", "Constrained": True, "Agent": PPO.load("complexPendulum/agents/neuralAgents/direct/QR3Direct_best.zip")}

#Gain Models
GainQR1 = {"Setup": 1, "Algo": "PPO", "Action": "Gain", "Constrained": True, "Agent": PPO.load("complexPendulum/agents/neuralAgents/gain/QR1Gain_last.zip")}
GainQR1best = {"Setup": 1, "Algo": "PPO", "Action": "Gain", "Constrained": True, "Agent": PPO.load("complexPendulum/agents/neuralAgents/gain/QR1Gain_best.zip")}

GainQR2 = {"Setup": 2, "Algo": "PPO", "Action": "Gain", "Constrained": True, "Agent": PPO.load("complexPendulum/agents/neuralAgents/gain/QR2Gain_last.zip")}
GAINQR2best = {"Setup": 2, "Algo": "PPO", "Action": "Gain", "Consrained": True, "Agent": PPO.load("complexPendulum/agents/neuralAgents/gain/QR2Gain_best.zip")}

GainQR3 = {"Setup": 3, "Algo": "PPO", "Action": "Gain", "Constrained": True, "Agent": PPO.load("complexPendulum/agents/neuralAgents/gain/QR3Gain_last.zip")}
GainQR3best = {"Setup": 3, "Algo": "PPO", "Action": "Gain", "Constrained": True, "Agent": PPO.load("complexPendulum/agents/neuralAgents/gain/QR3Gain_best.zip")}


#Baselined Models
BaseQR1 = {"Setup": 1, "Algo": "PPO", "Action": "Base", "Constrained": True, "Agent": PPO.load("complexPendulum/agents/neuralAgents/base/QR1Base_last.zip")}
BaseQR1best = {"Setup": 1, "Algo": "PPO", "Action": "Base", "Constrained": True, "Agent": PPO.load("complexPendulum/agents/neuralAgents/base/QR1Base_best.zip")}

BaseQR2 = {"Setup": 2, "Algo": "PPO", "Action": "Base", "Constrained": True, "Agent": PPO.load("complexPendulum/agents/neuralAgents/base/QR2Base_last.zip")}
BaseQR2best = {"Setup": 2, "Algo": "PPO", "Action": "Base", "Constrained": True, "Agent": PPO.load("complexPendulum/agents/neuralAgents/base/QR2Base_best.zip")}

BaseQR3 = {"Setup": 3, "Algo": "PPO", "Action": "Base", "Constrained": True, "Agent": PPO.load("complexPendulum/agents/neuralAgents/base/QR3Base_last.zip")}
BaseQR3best = {"Setup": 3, "Algo": "PPO", "Action": "Base", "Constrained": True, "Agent": PPO.load("complexPendulum/agents/neuralAgents/base/QR3Base_best.zip")}
