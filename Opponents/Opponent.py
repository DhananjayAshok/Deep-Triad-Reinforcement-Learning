#Imports Here
import numpy as np
from Configs import ACTION_CLASS, GAME_CLASS, ENVIRONMENT_CLASS




##############################################################################################

class Opponent(object):
    """
    Will be an abstract parent class for various other children who implement different strategies
    """
    def __init__(self, **kwargs):
        self.g = GAME_CLASS(**kwargs)
        self.g_env = ENVIRONMENT_CLASS(**kwargs)


    def play(self, state, **kwargs):
        """
        Given the state of the game return a move that the opponent plays
        """
        raise NotImplementedError

