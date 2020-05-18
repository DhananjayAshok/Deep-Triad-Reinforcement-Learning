#Imports Here
import numpy as np
from GameSystem.Actions import Connect4Action
from GameSystem.Games import Connect4Game
from GameSystem.Environments import Connect4Environment





##############################################################################################

class Opponent(object):
    """
    Will be an abstract parent class for various other children who implement different strategies
    """
    def __init__(self, **kwargs):
        self.g = Connect4Game(**kwargs)
        self.g_env = Connect4Environment(**kwargs)


    def play(self, state, **kwargs):
        """
        Given the state of the game return a move that the opponent plays
        """
        raise NotImplementedError

