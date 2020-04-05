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

aqagent.load_model(path=agent_path)
gym.simulate(aqagent, g, r, r, episodes = 1_500, training=False)
#gym.simulate(hyp, g, r, r, episodes = 3000, training=False)
#aqagent.save_model(path=agent_path)
g.play_slow(aqagent, r, r, avoid_illegal=False)