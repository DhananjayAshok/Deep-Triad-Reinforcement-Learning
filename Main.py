from Configs import ACTION_CLASS, GAME_CLASS, ENVIRONMENT_CLASS
###############################################################
from Gyms.Gyms import ForwardTDLambdaGym, BatchDQLearningGym
from Opponents.Opponents import RandomOpponent, HyperionOpponent
from Agents.OpponentAgent import OpponentAgent
from Agents.TrainableAgents.QAgents.DictionaryAgent import DictionaryAgent
from time import time



r = RandomOpponent(blocking=True, winning=True)
hyp = HyperionOpponent()
agent = DictionaryAgent(0.8, 0.1)
g = ENVIRONMENT_CLASS()

gym = BatchDQLearningGym()
start = time()
gym.simulate(agent, g, episodes=20_001, opponent_1=r)
end = time()
print(f"Seconds taken -> end-start")