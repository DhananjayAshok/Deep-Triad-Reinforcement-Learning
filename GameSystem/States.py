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

class TicTacToeState(State):
    
    def __init__(self, board, current_player, next_player):
        """
        Creates a new state
        """
        board = np.reshape(board, (9,))
        self.rep = np.append(board, [current_player, next_player])
        return

    def __str__(self):
        """
        will return the state vector in a way that humans can read
        """
        board = self.rep[0:9].reshape((3,3))
        turn = self.rep[9]
        next = self.rep[-1]
        s = f"Board\n {board}\n "+ f"Current Turn - {turn}| Next turn - {next}"
        return s

    def get_induviduals(self):
        """
        Returns the unpacked tuple - board, current player, next player
        """
        return self.rep[:9].reshape((3,3)), self.rep[9], self.rep[-1]

    def get_data(self):

        return self.rep.copy()


