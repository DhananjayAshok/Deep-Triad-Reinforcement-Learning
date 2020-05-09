from Configs import ACTION_CLASS, GAME_CLASS, ENVIRONMENT_CLASS
###############################################################
from Gyms.Gyms import ForwardTDLambdaGym, BatchDQLearningGym
from Opponents.Opponents import HumanOpponent, HyperionOpponent, RandomOpponent
from Agents.TrainableAgents.QAgents.DeepQAgents import SimpleDeepAgent
from Agents.OpponentAgent import OpponentAgent
from Agents.TrainableAgents.QAgents.DictionaryAgent import TicTacToe3DDictionaryAgent





g = ENVIRONMENT_CLASS()
h = HumanOpponent()
r = RandomOpponent(blocking=True, winning=True)
hyp = HyperionOpponent()
agent = SimpleDeepAgent(1, 0.2)

gym = BatchDQLearningGym()
#agent.load()
gym.simulate(agent, g, opponent_1=r, opponent_2=r, training=True, episodes = 1001)
agent.save()

