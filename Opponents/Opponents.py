#Core Imports
from GameSystem.Environments import TicTacToeEnvironment
from GameSystem.Games import TicTacToeGame
from GameSystem.Actions import TicTacToeAction
from GameSystem.States import TicTacToeState
from .Opponent import Opponent
from Utility import get_state_data
##############################################################################################
import numpy as np


class TicTacToeOpponent(Opponent):
    def __init__(self):
        Opponent.__init__(self)

    def blocking_move(self, state):
        """
        If the next player has any move that wins them the game return a move that blocks at least one of those moves else returns -1
        """
        board, turn, next = get_state_data(state)

        for action in range(1, 10):
            if self.g.is_legal(action, board) and self.g.check_for_win(action, next, board) == next:
                return action
        return -1

    def winning_move(self, state):
        """
        If there is an object that will win the AI the game this move it returns the move that wins else it returns -1
        """
        board, turn, next = get_state_data(state)

        for action in range(1,10):
            if self.g.is_legal(action, board) and self.g.check_for_win(action, turn, board) == turn:
                return action
        return -1

class SelfOpponent(TicTacToeOpponent):
    """
    Is an Opponent object for use to train the Agent against a former version of itself
    """
    pass

class HumanOpponent(TicTacToeOpponent):
    """
    Is an Opponent object for use to test the Agent against a human
    """
    def __init__(self):
        TicTacToeOpponent.__init__(self)

    def play(self, state):
        print("Before Human Turn State is -")
        print(state)

        while True:
            inp = input("Enter a legal move from 1-9")
            try:
                int(inp)
            except:
                print("That move was not an integer")
            else:
                act = TicTacToeAction(int(inp))
                board, curr, next = get_state_data(state)
                legal = self.g.is_legal(act, state)
                if legal:
                    return act
                else:
                    print("That move was not legal")

class RandomOpponent(TicTacToeOpponent):
    """
    Is an Opponent that plays truly random moves (that are legal)
    Can be parameterized to also
        block immediate winning moves from the next player
        win if move exists
    """
    def __init__(self, blocking=False, winning=False, **kwargs):
        TicTacToeOpponent.__init__(self)
        if kwargs.get('blocking', None) is not None:
            self.blocking = kwargs.get('blocking', False)
            self.winning = kwargs.get('winning', False)
        else:
            self.blocking = blocking
            self.winning = winning


    def play(self, state):
        """
        Plays a random move (unless winning and/or blocking was true
        """
        board, turn, next = get_state_data(state)
        if self.winning:
            winmove = self.winning_move(state)
            if (winmove != -1):
                #print("Tries Winning Move")
                return winmove
        if self.blocking:
            blockmove = self.blocking_move(state)
            if blockmove != -1:
                #print("Tries Blocking Move")
                return blockmove
        #print("Does Neither")
        choices = []
        for action in range(1, 10):
            if self.g.is_legal(action, board):
                choices.append(action)
        #print(f"Thinks its choices are {choices}")
        return np.random.choice(choices)

class HyperionOpponent(TicTacToeOpponent):
    """
    Plays against Hyperion the Greedy
    """
    def __init__(self):
        TicTacToeOpponent.__init__(self)

    def play(self, state):
        board, player, next_player = get_state_data(state)
        winmove = self.winning_move(state)
        if winmove != -1:
            return winmove
        blockmove = self.blocking_move(state)
        if blockmove != -1:
            return blockmove
        choices = [-1 for i in range(0, 10)]
        for action in range(1, 10):
            if self.g.is_legal(action, board):
                choices[action.act-1] = self.g.get_attack_score(action, player, board)
        return random_highest_index(choices)+1


def random_highest_index(scores):
    """
    Returns a random index out of all the maximum values of scores
    """
    max_indexes = np.where(np.asarray(scores) == max(scores))[0]
    return np.random.choice(max_indexes)

