from Opponents import HumanOpponent, RandomOpponent, HyperionOpponent
from GameSystem.game import GameEnvironment, Game
from Agents import HumanAgent,  QLinearAgent, RandomAgent
from Gyms import ForwardTDLambdaGym


g = GameEnvironment()
r = RandomOpponent(blocking=True, winning=True)
h = HyperionOpponent()

agent = QLinearAgent(0.007, 0.2)
rando = RandomAgent()
agent_path = "models/QLinearAgent"


gym = ForwardTDLambdaGym()

agent.load_model(path=agent_path)
#gym.simulate(agent, g, h, h, episodes = 3000, training=False)
gym.simulate(agent, g, r, r, episodes = 3000, training=False)
#g.play_slow(agent, h, h)