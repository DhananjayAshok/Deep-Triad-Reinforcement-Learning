from Configs import ACTION_CLASS, GAME_CLASS, ENVIRONMENT_CLASS
###############################################################
from Gyms.Gyms import ForwardTDLambdaGym, BatchDQLearningGym
from Opponents.Opponents import RandomOpponent, HyperionOpponent
from Agents.OpponentAgent import OpponentAgent


r = RandomOpponent(blocking=True, winning=True)
hyp = HyperionOpponent()
agent = OpponentAgent(HyperionOpponent)
g = ENVIRONMENT_CLASS()

gym = ForwardTDLambdaGym()
gym.simulate(agent, g, episodes=1000, show_every=50, training=False, opponent_1=r)