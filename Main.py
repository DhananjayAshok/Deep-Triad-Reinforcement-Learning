from GameSystem.Environments import TicTacToeEnvironment
from GameSystem.Games import TicTacToeGame
from GameSystem.Actions import TicTacToeAction
from GameSystem.States import TicTacToeState

###############################################################
from Gyms.Gyms import ForwardTDLambdaGym, BatchDQLearningGym
from Opponents.Opponents import RandomOpponent, HyperionOpponent
from Agents.Agent import DictionaryAgent
from time import time



r = RandomOpponent(blocking=True, winning=True)
hyp = HyperionOpponent()
agent = DictionaryAgent(0.8, 0.1)
g = TicTacToeEnvironment()


gym = BatchDQLearningGym()
start = time()
gym.simulate(agent, g, episodes=20_001, opponent_1=r)
end = time()
print(f"Seconds taken -> end-start")