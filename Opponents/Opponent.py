#Imports Here
import numpy as np
from GameSystem.Environments import TicTacToeEnvironment
from GameSystem.Games import TicTacToeGame
from GameSystem.Actions import TicTacToeAction
from GameSystem.States import TicTacToeState




##############################################################################################

class Opponent(object):
    """
    Will be an abstract parent class for various other children who implement different strategies
    """
    def __init__(self, **kwargs):
        self.g = TicTacToeGame(**kwargs)
        self.g_env = TicTacToeEnvironment(**kwargs)


    def play(self, state, **kwargs):
        """
        Given the state of the game return a move that the opponent plays
        """
        raise NotImplementedError

