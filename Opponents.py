import numpy as np
from GameSystem.game import Game, GameEnvironment
from Utility import MaxN, random_highest_index

class Opponent(object):
    """
    Will be an abstract parent class for various other children who implement different strategies
    """
    def __init__(self):
        self.g = Game()
        self.g_env = GameEnvironment()


    def play(self, state):
        """
        Given the state of the game return a move that the opponent plays
        """
        raise NotImplementedError


    def blocking_move(self, state):
        """
        If the next player has any move that wins them the game return a move that blocks at least one of those moves else returns -1
        """
        board = state[:27].reshape((3,3,3))
        turn = state[27]
        next = state[28]

        for move in range(1, 10):
            if self.g.is_legal(move, board) and self.g.check_for_win(move, next, board) == next:
                return move
        return -1

    def winning_move(self, state):
        """
        If there is an object that will win the AI the game this move it returns the move that wins else it returns -1
        """
        board = state[:27].reshape((3,3,3))
        turn = state[27]
        next = state[28]

        for move in range(1, 10):
            if self.g.is_legal(move, board) and self.g.check_for_win(move, turn, board) == turn:
                return move
        return -1

class SelfOpponent(Opponent):
    """
    Is an Opponent object for use to train the Agent against a former version of itself
    """
    pass

class HumanOpponent(Opponent):
    """
    Is an Opponent object for use to test the Agent against a human
    """
    def play(self, state):
        print("Before Human Turn State is -")
        self.g_env.print_state(provided=state)

        while True:
            inp = input("Enter a legal move from 1-9")
            try:
                int(inp)
            except:
                print("That move was not an integer")
            else:
                legal = self.g.is_legal(int(inp), state[:27].reshape(3,3,3))
                if legal:
                    return int(inp)
                else:
                    print("That move was not legal")

class RandomOpponent(Opponent):
    """
    Is an Opponent that plays truly random moves (that are legal)
    Can be parameterized to also
        block immediate winning moves from the next player
        win if move exists
    """
    def __init__(self, blocking=False, winning=False):
        Opponent.__init__(self)
        self.blocking = blocking
        self.winning = winning

    def play(self, state):
        """
        Plays a random move (unless winning and/or blocking was true
        """
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
        for move in range(1, 10):
            if self.g.is_legal(move, state[:27].reshape((3,3,3))):
                choices.append(move)
        #print(f"Thinks its choices are {choices}")
        return np.random.choice(choices)

class HyperionOpponent(Opponent):
    """
    Plays against Hyperion the Greedy
    """
    def play(self, state):
        board = state[:27].reshape((3,3,3)).copy()
        player = state[27]
        next_player = state[28]
        winmove = self.winning_move(state)
        if winmove != -1:
            return winmove
        blockmove = self.blocking_move(state)
        if blockmove != -1:
            return blockmove
        choices = [-1 for i in range(0, 10)]
        for move in range(1,10):
            if self.g.is_legal(move, board):
                choices[move] = self.g.get_attack_score(move, player, board)
        return random_highest_index(choices)

class MMOpponent(Opponent):
    """
    Plays against a minimax AI bot
    """
    def play(self,state):
        best_score=-10
        x = 1
        for action in range(1,10):
            evaluation= MaxN(state)
            score=evaluation[0]
            if score>best_score:
                best_score=score
                x =action
        #play peice on x
        return x
