import numpy as np
from Opponents import HumanOpponent, RandomOpponent, HyperionOpponent , MMOpponent
from GameSystem.game import GameEnvironment, Game
from Agents import HumanAgent,  QLinearAgent, RandomAgent, HyperionAgent, NaiveDeepQAgent, MMagent, AssistedDeepQAgent, DictionaryAgent
from Gyms import ForwardTDLambdaGym, BatchDQLearningGym


agent = None
g = GameEnvironment()
rdumb = RandomOpponent()
rsmart = RandomOpponent(blocking=True, winning=True)
h = HyperionOpponent()
MMagent = MMAgent()
gym = BatchDQLearningGym(epsilon=0.999, avoid_illegal=True, clear_after_episode=True)
gym.simulate(MMagent, g, rdumb, rsmart, episodes = 10, training=False)
#gym.simulate(d, g, r, r, episodes = 3000, training=True)
