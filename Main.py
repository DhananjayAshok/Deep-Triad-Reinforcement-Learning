import numpy as np
from Opponents import HumanOpponent, RandomOpponent, HyperionOpponent
from GameSystem.game import GameEnvironment, Game
from Agents import HumanAgent,  QLinearAgent, RandomAgent, HyperionAgent, NaiveDeepQAgent, AssistedDeepQAgent, DictionaryAgent
from Gyms import ForwardTDLambdaGym, BatchDQLearningGym



g = GameEnvironment()
rdumb = RandomOpponent()
rsmart = RandomOpponent(blocking=True, winning=True)
h = HyperionOpponent()

d = DictionaryAgent(0.5, 0.1)



gym = BatchDQLearningGym(epsilon=0.999, avoid_illegal=True, clear_after_episode=True)

d.load_model(alternate_name="bigdict")
gym.simulate(d, g, rdumb, rdumb, episodes = 500_000, training=True)
#gym.simulate(d, g, r, r, episodes = 3000, training=True)
d.save_model(alternate_name="dict")
nz, il, win, inter = d.stats()