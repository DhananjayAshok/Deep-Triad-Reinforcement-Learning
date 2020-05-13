from Configs import ACTION_CLASS, GAME_CLASS, ENVIRONMENT_CLASS
###############################################################
from Gyms.Gyms import ForwardTDLambdaGym, BatchDQLearningGym
from Opponents.Opponents import HumanOpponent, RandomOpponent
from Agents.OpponentAgent import OpponentAgent
from Agents.TrainableAgents.QAgents.DictionaryAgent import DictionaryAgent




g = ENVIRONMENT_CLASS()
r = RandomOpponent(winning=True, blocking=True)
h = HumanOpponent()
a = DictionaryAgent(0.8, 0.2)
gym = BatchDQLearningGym(epsilon=0.8)
gym.simulate(a, g, opponent=r, episodes = 10_001, show_every=1000, save_every=2500)