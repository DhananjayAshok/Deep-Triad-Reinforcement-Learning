import numpy as np
from Opponents import HumanOpponent, RandomOpponent, HyperionOpponent , MMOpponent
from GameSystem.game import GameEnvironment, Game
from Agents import SimpleDeepAgent, ConvAgent, FeaturedDeepAgent
from Gyms import ForwardTDLambdaGym, BatchDQLearningGym, NEATArena

g = GameEnvironment()
rdumb = RandomOpponent()
rsmart = RandomOpponent(blocking=True, winning=True)
h = HyperionOpponent()
path = "models//DeepQAgent"

agent = FeaturedDeepAgent(1, 0.2, min_replay_to_fit=1_000)
gym = BatchDQLearningGym(0.9)
gym.simulate(agent, g, rsmart, rsmart, 5_000)
agent.save_model(path)

