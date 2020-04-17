from Opponents import HumanOpponent, RandomOpponent, HyperionOpponent
from GameSystem.game import GameEnvironment, Game
from Agents import HumanAgent,  QLinearAgent, RandomAgent, HyperionAgent, NaiveDeepQAgent, AssistedDeepQAgent
from Gyms import ForwardTDLambdaGym, BatchDQLearningGym



g = GameEnvironment()
r = RandomOpponent(blocking=True, winning=True)
h = HyperionOpponent()
hyp = HyperionAgent()
qagent = NaiveDeepQAgent(None, 0.35, avoid_assist=True, win=True, block=True)
#rando = RandomAgent()
aqagent = AssistedDeepQAgent(None, 0.35)
agent_path = "models/AssistedDeepQAgent"


gym = BatchDQLearningGym(epsilon=0.999, avoid_illegal=True)

<<<<<<< HEAD
agent.load_model(path=agent_path)
#gym.simulate(agent, g, h, h, episodes = 3000, training=False)
gym.simulate(agent, g, r, r, episodes = 1000, training=False)
#g.play_slow(agent, h, h)
=======
aqagent.load_model(path=agent_path)
gym.simulate(aqagent, g, r, r, episodes = 1_500, training=False)
#gym.simulate(hyp, g, r, r, episodes = 3000, training=False)
#aqagent.save_model(path=agent_path)
g.play_slow(aqagent, r, r, avoid_illegal=False)
>>>>>>> c58027305d451ba6f505f5317f0a54b6e0cdaae4
