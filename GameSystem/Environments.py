#Imports Here
import numpy as np
from .Games import TicTacToe3DGame
from .States import TicTacToe3DState




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

class TicTacToe3DEnvironment(Environment):
    """
    This object will contain all the working of the game environment but not the game itself. It must have the following functionalities


    Will probably need the following variables
        Game object -> to actually have a board and make moves
        MoveStack -> a stack of all moves that were played so it can be undone if we want

    Currently thinking the following will be the state vector
    [piece in 1st cell, ..... piece in 27th cell, who's turn is it, who plays next]


    """

    def __init__(self):
        """
        Initialize a new 3D Tic Tac Toe Environment and assign in a game object
        """
        Environment.__init__(self)
        self.g = TicTacToe3DGame()

    def reset(self, opponent_1, opponent_2):
        """
        will randomly assign order and present an empty field. If the first k players are opponent AIs it will return a state with k moves played.
        """
        players = [1,2,3]
        self.agent_turn = np.random.choice(players)
        players.remove(self.agent_turn)
        self.opponent_1 = opponent_1
        self.opponent_1_turn = np.random.choice(players)
        players.remove(self.opponent_1_turn)
        self.opponent_2 = opponent_2
        self.opponent_2_turn = players[0]
        self.g.restart()
        self.turn = 1
        self.turn_dict = {self.agent_turn: "AGENT", self.opponent_1_turn: self.opponent_1, self.opponent_2_turn: self.opponent_2}
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

    def play_move(self, action):
        """
        Makes a call to the play of the Game object. Assumes the move is legal and all checking for winning buisness is done
        We then return the reward and Done status
        """
        
        winner = self.g.play(action, self.turn)
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

        self.turn = self.turn%3+1
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
            opponent = self.opponent_2
        move = opponent.play(self.get_state())
        return move

    def play_slow(self, agent, opponent_1, opponent_2, avoid_illegal=True, verbose=False):
        """
        will create a new game but to be played slowly where each turn is only completed if a human presses an input
        """
        self.reset(opponent_1, opponent_2)
        counter = 0
        done=False

        while not done:
            print(f"Currently on turn {counter}")
            print(self.get_state())
            action = agent.play(self.get_state(), verbose=True, avoid_illegal=avoid_illegal)
            print(f"Player takes action {action}")
            new_state, reward, done = self.step(action, verbose=True)
            counter += 1
            input()
        return

    def get_state(self):
        """
        will return the state vector of the game as it is
        """
        board = self.g.get_board()
        board = np.reshape(board, (27,))
        board = np.append(board, [self.turn, self.turn%3+1])
        return TicTacToe3DState(self.g.get_board(), self.turn, self.turn%3+1)