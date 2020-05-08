from Opponents import HumanOpponent
from GameSystem.Environments import TicTacToe3DEnvironment
from GameSystem.Actions import TicTacToe3DAction


g = TicTacToe3DEnvironment()
h = HumanOpponent()
g.reset(h, h)
