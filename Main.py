from Opponents import HumanOpponent, RandomOpponent, HyperionOpponent
from GameSystem.game import GameEnvironment, Game
from Agents import HumanAgent,  QLinearAgent, RandomAgent, HyperionAgent, DeepQAgent
from Gyms import ForwardTDLambdaGym, BatchDQLearningGym



g = GameEnvironment()
r = RandomOpponent(blocking=True, winning=True)
h = HyperionOpponent()
hyp = HyperionAgent()
qagent = DeepQAgent(None, 0.35, avoid_assist=True, win=True, block=True, model_name="AssistedDQA")
rando = RandomAgent()

agent_path = "models/DeepQAgent"


gym = BatchDQLearningGym(epsilon=0.999, avoid_illegal=False)

qagent.load_model(path=agent_path)
#gym.simulate(qagent, g, r, r, episodes = 500, training=True)
#gym.simulate(hyp, g, r, r, episodes = 3000, training=False)
#qagent.save_model(path=agent_path)
g.play_slow(qagent, r, r, avoid_illegal=False)