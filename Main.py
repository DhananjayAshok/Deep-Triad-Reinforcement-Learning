from Opponents import HumanOpponent, RandomOpponent, HyperionOpponent
from GameSystem.game import GameEnvironment, Game
from Agents import HumanAgent,  QLinearAgent
from Gyms import ForwardTDLambdaGym


g = GameEnvironment()
r = RandomOpponent()
h = HyperionOpponent()

agent = QLinearAgent(0.007, 0.2)
agent_path = "models/QLinearAgent"


gym = ForwardTDLambdaGym()

gym.train(agent, g, h, h, episodes = 30000)
agent.save_model(path=agent_path)