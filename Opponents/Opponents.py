#Core Imports
from .Opponent import Opponent
##############################################################################################
from GameSystem.Actions import Connect4Action
from GameSystem.Games import Connect4Game
from GameSystem.Environments import Connect4Environment
import numpy as np

class HumanOpponent(Opponent):
    def __init__(self):
        Opponent.__init__(self)

    def play(self, state, **kwargs):
        print(state)
        inp = input("please enter a legal number move")
        try:
            move = int(inp)
        except:
            print("move not an integer try again")
            self.play(state)
        action = Connect4Action(move)
        if not self.g.is_legal(action, provided_state=state):
            print("Move not valid try again")
            self.play(state)
        else:
            return action


class RandomOpponent(Opponent):
    def __init__(self, winning=False, blocking=False):
        self.winning = winning
        self.blocking = blocking
        Opponent.__init__(self)

    def wincheck(self, state, legals):
        for action in legals:
            if self.g.check_for_win(action, state.get_turn(), provided_state=state):
                return action, True
        return None, False

    def blockchoices(self, state, legals):
        board = state.get_board()
        losers = []
        for action in legals:
            self.g.grid = board.copy()
            if not self.g.is_legal(action):
                losers.append(action)
                break
            else:
                status = self.g.play(action, state.get_turn())
                if status != 0:
                    break
                for act in Connect4Action.get_action_space():
                    if self.g.check_for_win(act, state.get_turn()%2+1):
                        losers.append(action)
                        break

        return list(set(legals).difference(set(losers)))

    def play(self, state):
        choices = Connect4Action.get_action_space()
        legals = []
        for choice in choices:
            if self.g.is_legal(choice, provided_state=state):
                legals.append(choice)
        if len(legals) == 0:
                raise ValueError(f"Called Play on an opponent when there is no legal move to be made \n {state}")
        if self.winning:
            move, make = self.wincheck(state, legals)
            if make:
                return move
        if self.blocking:
            choices = self.blockchoices(state, legals)
            if len(choices) > 0:
                return np.random.choice(choices)
        return np.random.choice(legals)