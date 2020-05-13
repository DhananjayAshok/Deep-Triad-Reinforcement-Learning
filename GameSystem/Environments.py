#region
#Core Imports Here

##############################################################################################

class Environment(object):
    """
    Abstract Base Class to store the most generic possible Environment.
    All Environments will be children of this class
    """
    def reset(self, **kwargs):
        raise NotImplementedError

    def step(self, **kwargs):
        raise NotImplementedError

    def play_slow(self, **kwargs):
        """
        Meant to play the game at a pace which human debuggers can comprehend
        """
        raise NotImplementedError

    def get_state(self, **kwargs):
        raise NotImplementedError


# Implement Your Custom Classes Below
##############################################################################################
#endregion
from .Games import Connect4Game
from .States import Connect4State
import numpy as np

class Connect4Environment(Environment):
    def __init__(self, win_reward=10, loss_reward=-10, draw_reward=0, illegal_reward=-50):
        self.win_reward=win_reward
        self.loss_reward=loss_reward
        self.draw_reward=draw_reward
        self.illegal_reward=illegal_reward
        self.g = Connect4Game()

    def reset(self, **kwargs):
        """
        Requires argument opponent of class Opponent (instance)
        returns a state object
        """
        self.opponent = kwargs.get('opponent', None)
        if self.opponent is None:
            raise ValueError("Reset called without opponent")
        self.turn = 1
        turns = [1, 2]
        self.agent_turn = np.random.choice(turns)
        self.g.restart()
        if self.agent_turn != 1:
            self.opponent_move()
        return self.get_state()

    def opponent_move(self):
        """
        Returns new_state, reward, done
        """
        act = self.opponent.play(self.get_state())
        winner = self.g.play(act, self.turn)
        if winner == self.turn:
            self.increment_turn()
            return self.get_state(), self.loss_reward, True
        self.increment_turn()
        if winner == 0:
            return self.get_state(), 0, False
        elif winner == -1:
            return self.get_state(), self.draw_reward, True
        else:
            return self.get_state(), self.win_reward, True

    def step(self, action):
        """
        Returns new_state, reward, done
        """
        

        if not self.g.is_legal(action):
            print(f"Illegal Move was made")
            return self.get_state(), self.illegal_reward, False

        winner = self.g.play(action, self.turn)
        if winner == self.turn:
            self.increment_turn()
            return self.get_state(), self.win_reward, True
        self.increment_turn()
        if winner == -1:
            return self.get_state(), self.draw_reward, True
        else:
            return self.opponent_move()

    def get_state(self):
        """
        Returns a state object
        """
        return Connect4State(self.g.grid, self.turn)

    def increment_turn(self):
        """
        in place changes the turn from given turn
        """
        self.turn = self.turn%2 + 1