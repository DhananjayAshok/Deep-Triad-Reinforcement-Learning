from Opponents import HumanOpponent, RandomOpponent, HyperionOpponent
from GameSystem.game import GameEnvironment, Game
from Agents import HumanAgent,  QLinearAgent
from Gyms import ForwardTDLambdaGym


g = GameEnvironment()
r = RandomOpponent()
agent = QLinearAgent(0.07, 0.2)
agent_path = "models/QLinearAgent"


gym = ForwardTDLambdaGym()

gym.train(agent, g, r, r, episodes = 3000)
agent.save_model(path=agent_path)