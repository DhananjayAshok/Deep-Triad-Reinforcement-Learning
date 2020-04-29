import numpy as np
from Opponents import HumanOpponent, RandomOpponent, HyperionOpponent , MMOpponent
from GameSystem.game import GameEnvironment, Game
from Agents import HumanAgent,  QLinearAgent, RandomAgent, HyperionAgent, NaiveDeepQAgent, MMAgent, AssistedDeepQAgent, DictionaryAgent
from Gyms import ForwardTDLambdaGym, BatchDQLearningGym, NEATArena

g = GameEnvironment()
rdumb = RandomOpponent()
rsmart = RandomOpponent(blocking=True, winning=True)
h = HyperionOpponent()

agent = None
gym = NEATArena("models/NEATAgent")
gym.simulate("config-feedforward.txt", GameEnvironment, rsmart, rsmart)
