from Configs import ACTION_CLASS, GAME_CLASS, ENVIRONMENT_CLASS
###############################################################
from Gyms.Gyms import ForwardTDLambdaGym, BatchDQLearningGym
from Opponents.Opponents import HumanOpponent, RandomOpponent
from Agents.OpponentAgent import OpponentAgent




g = ENVIRONMENT_CLASS()
r = RandomOpponent(winning=True, blocking=True)
h = HumanOpponent()
a = OpponentAgent(RandomOpponent, winning=True, blocking=True)
gym = ForwardTDLambdaGym(epsilon=0.5)
gym.simulate(a, g, opponent=r, training=False, episodes = 1001, show_every=50)