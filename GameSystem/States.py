#Core Imports Here
##############################################################################################

class State(object):
    """Abstract base state class"""
    def __str__(self):
        raise NotImplementedError


# Implement Your Custom Classes Below
##############################################################################################
import numpy as np

class TicTacToe3DState(State):
    
    def __init__(self, board, current_player, next_player):
        """
        Creates a new state
        """
        board = np.reshape(board, (27,))
        self.rep = np.append(board, [current_player, next_player])
        return

    def __str__(self):
        """
        will return the state vector in a way that humans can read
        """
        board = self.rep[0:27].reshape((3,3,3))
        turn = self.rep[27]
        next = self.rep[-1]
        s = f"Board\n {board[2]} \n \t3 \n {board[1]} \n \t2 \n {board[0]} \n \t1" + "\n" + f"Current Turn - {turn}| Next turn - {next}"
        return s

    def get_induviduals(self):
        """
        Returns the unpacked tuple - board, current player, next player
        """
        return self.rep[:27].reshape((3,3,3)), self.rep[27], self.rep[28]



