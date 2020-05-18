from Configs import ACTION_CLASS, GAME_CLASS, ENVIRONMENT_CLASS
###############################################################
from Gyms.Gyms import ForwardTDLambdaGym, BatchDQLearningGym
from Opponents.Opponents import HumanOpponent, RandomOpponent
from Agents.OpponentAgent import OpponentAgent
from Agents.TrainableAgents.QAgents.DictionaryAgent import DictionaryAgent
from Agents.TrainableAgents.QAgents.DeepQAgents import ConvAgent





g = ENVIRONMENT_CLASS()
r = RandomOpponent(winning=True, blocking=True)
h = HumanOpponent()
a = ConvAgent(0.8, 0.2)
a.load()
gym = BatchDQLearningGym(epsilon=0.0)
gym.simulate(a, g, opponent=r, episodes = 501, show_every=50, training=False, save_every=2500)