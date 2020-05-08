#Imports Here
import numpy as np




##############################################################################################

class Action(object):
    """Abstract base action class"""
    def get_action_space():
        """
        Returns a list of valid input parameters to create an action
        Not always applicable
        """
        raise NotImplementedError

    def print(self):
        return str(self)

# Implement Your Custom Classes Below
##############################################################################################
class TicTacToe3DAction(Action):
    def get_action_space():
        """
        Returns the list of numbers from 1 through 9
        """
        return [i for i in range(1, 10)]
    
    def __init__(self, inp):
        Action.__init__(self)
        self.act = inp

    def __str__(self):
        return str(self.act)


