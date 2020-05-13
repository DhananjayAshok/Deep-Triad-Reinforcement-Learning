#Core Imports Here
##############################################################################################

class State(object):
    """Abstract base state class"""
    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        return str(self)

    def get_data(self):
        """
        Returns the details of the state that is relevant in some form described by the User
        """
        raise NotImplementedError


# Implement Your Custom Classes Below
##############################################################################################
import numpy as np
class Connect4State(State):
    def __init__(self, board, current_player):
        self.rep = np.append(board.reshape((36,)), current_player)
        State.__init__(self)

    def __str__(self):
        return str(self.rep[:36].reshape((6,6))) + "\n" + "Current Player: " + str(self.rep[36])

    def get_data(self):
        return self.rep
    
    def get_board(self):
        return self.rep[:36].reshape((6,6))

    def get_turn(self):
        return self.rep[36]

    def get_induviduals(self):
        return self.get_board(), self.get_turn()



