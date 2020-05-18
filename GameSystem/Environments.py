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
from .Games import TicTacToeGame
from .States import TicTacToeState
import numpy as np


class TicTacToeEnvironment(Environment):
    """
    This object will contain all the working of the game environment but not the game itself. It must have the following functionalities


    Will probably need the following variables
        Game object -> to actually have a board and make moves
        MoveStack -> a stack of all moves that were played so it can be undone if we want

    Currently thinking the following will be the state vector
    [piece in 1st cell, ..... piece in 9th cell, who's turn is it, who plays next]


    """

    def __init__(self):
        """
        """
        Environment.__init__(self)
        self.g = TicTacToeGame()

    def reset(self, opponent_1=None, **kwargs):
        """
        will randomly assign order and present an empty field. If the first k players are opponent AIs it will return a state with k moves played.
        """
        if opponent_1 is None:
            opponent_1 = kwargs.get('opponent_1', None)
        if opponent_1 is None:
            raise ValueError("Opponents not entered correctly")
        players = [1,2]
        self.agent_turn = np.random.choice(players)
        players.remove(self.agent_turn)
        self.opponent_1 = opponent_1
        self.opponent_1_turn = np.random.choice(players)
        players.remove(self.opponent_1_turn)
        self.g.restart()
        self.turn = 1
        self.turn_dict = {self.agent_turn: "AGENT", self.opponent_1_turn: self.opponent_1}
        fin_state, reward, done = self.opponents_move_sequence()
        return fin_state

    def step(self, action, verbose=False):
        """
        will take in an action from the agent and then play 2 other moves (either opponent AI or Random or human) and then return - new_state vector, rewards, done
        """
        if not self.g.is_legal(action):
            print("Player Tried Illegal Move")
            return self.get_state(), -10, False
        agent_turn_reward, agent_turn_done = self.play_move(action)
        if agent_turn_done:
            return self.get_state(), agent_turn_reward, agent_turn_done
        else:
            return self.opponents_move_sequence(verbose=verbose)

    def play_move(self, move):
        """
        Makes a call to the play of the Game object. Assumes the move is legal and all checking for winning buisness is done
        We then return the reward and Done status
        """
        winner = self.g.play(move, self.turn)
        win_reward = 1
        loss_reward = -1
        if winner == -1:
            reward = 0
            done = True
        elif winner == 0:
            reward = 0
            done = False
        elif winner == self.agent_turn:
            reward = win_reward
            done = True
        else:
            reward = loss_reward
            done = True

        self.turn = self.turn%2+1
        return reward, done

    def opponents_move_sequence(self, verbose=False):
        """
        Simulates both opponents playing until the next agent turn
        Returns next_state, reward and done after either an opponent wins or draws or the agents turn is reached.
        """
        while self.turn != self.agent_turn:
            reward = 0
            done = False
            move = self.turn_dict[self.turn].play(self.get_state())
            if verbose:
                print(f"Player {self.turn} plays {move}")
            reward, done = self.play_move(move)
            if done:
                return self.get_state(), reward, done
        return self.get_state(), 0, False

    def get_opponent_move(self, verbose=False):
        if self.turn == self.opponent_1_turn:
            opponent = self.opponent_1
        else:
            # Will crash but honestly will never get here
            opponent = self.opponent_2
        move = opponent.play(self.get_state())
        return move

    def play_slow(self, agent, opponent_1, avoid_illegal=True, verbose=False):
        """
        will create a new game but to be played slowly where each turn is only completed if a human presses an input
        """
        self.reset(opponent_1)
        counter = 0
        done=False

        while not done:
            print(f"Currently on turn {counter}")
            self.print_state()
            action = agent.play(self.get_state(), verbose=True, avoid_illegal=avoid_illegal)
            print(f"Player takes action {action}")
            new_state, reward, done = self.step(action, verbose=True)
            self.print_state()
            counter += 1
            input()
        return

    def get_state(self):
        """
        will return the state vector of the game as it is
        """
        return TicTacToeState(self.g.matrix, self.turn, self.turn%2+1)

    def print_state(self, provided=None):
        """
        will print the state vector in a way that humans can read
        """
        if provided is None:
            provided = self.get_state()
        print(provided)