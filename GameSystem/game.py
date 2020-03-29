import numpy as np
from Line import get_win_lines

class GameEnvironment(object):
    """
    This object will contain all the working of the game environment but not the game itself. It must have the following functionalities
    

    Will probably need the following variables
        Game object -> to actual have a board and make moves
        MoveStack -> a stack of all moves that were played so it can be undone if we want

    Currently thinking the following will be the state vector
    [piece in 1st cell, ..... piece in 27th cell, who are you, who plays next]


    """
    def reset(human_players):
        """
        will randomly assign order and present an empty field. If the first k players are opponent AIs it will return a state with k moves played.
        """
    def step(action):
        """
        will take in an action from the agent and then play 2 other moves (either opponent AI or Random or human) and then return - new_state vector, rewards, done
        """

    def play_slow(player_policy, enemy1, enemy_2):
        """
        will create a new game but to be played slowly where each turn is only completed if a human presses an input
        """
    
    def get_state():
        """
        will return the state vector of the game as it is
        """

    def print_state():
        """
        will print the state vector in a way that humans can read
        """



class Game(object):
    """
    This object will have the board representation and implement all the basic rules of the games. Will need the following methods:

    Matrix is of dimensions [height layer, row layer, column layer ]
    """
    def __init__(self):
        self.restart()
        self.win_lines = get_win_lines()

    def restart(self):
        """
        resets its internal representation
        """
        self.matrix = np.zeros(shape=(3,3,3), dtype=np.int8)
        self.move_stack = []

    def get_board(self):
        """
        returns the board as it currently is
        """
        return self.matrix

    def is_legal(self, move, board=None):
        """
        returns true iff the proposed move is legal given the board at its current state
        """
        if not (1 <= move <= 9):
            return False
        else:
            decoded = self._decode_move(move, board)
            return decoded[0] <= 2

    def undo_move(self):
        """
        Undoes move
        """
        move = self.move_stack.pop()
        decoded = self._decode_move(move)
        if (self.matrix[2, decoded[1], decoded[2]] != 0):
            self.matrix[2, decoded[1], decoded[2]] = 0
        elif (self.matrix[1, decoded[1], decoded[2]] != 0):
            self.matrix[1, decoded[1], decoded[2]] = 0
        else:
            self.matrix[0, decoded[1], decoded[2]] = 0
        return move

    def play(self, move, player):
        """
        make a move with that player and return winner - (-1 if draw, 0 if game is still going on, else x if player x has won)
        """
        self.move_stack.append(move)
        decoded = self._decode_move(move)
        winner = self.check_for_win(move, player)
        self.matrix[decoded[0], decoded[1], decoded[2]] = player
        return winner

    def _decode_move(self,move, board=None):
        """
        Takes in a move and returns [height, row col] that is intended
        """
        if board is None:
            board = self.matrix
        col = (move-1)%3
        if move > 6:
            row = 2
        elif move > 3:
            row = 1
        else:
            row = 0
        if board[0, row, col] == 0:
            height = 0
        elif board[1, row, col] == 0:
            height = 1
        elif board[2, row, col] == 0:
            height = 2
        else:
            height = 3
        return [height, row, col]

    def piece_at(self, point, board=None):
        """
        Returns the piece at that point
        """
        if board is None:
            board = self.matrix
        return board[point[0], point[1], point[2]]

    def check_for_win(self, proposed_move, proposed_player, board=None):
        """
        returns winner as required by the above play function but its hypothetical so doesn't change anything. Useful for AI logic
        Prerequisite: Before checkforwin is called there is no winner
        """
        if board is None:
            board = self.matrix.copy()
        else:
            board = board.copy()

        move = self._decode_move(proposed_move, board)
        board[move[0], move[1], move[2]] = proposed_player

        for line in self.win_lines:
            if(self.piece_at(line.point0, board) == self.piece_at(line.point1, board) == self.piece_at(line.point2, board)) and self.piece_at(line.point0, board)!=0:
                return self.piece_at(line.point0, board)
        if 0 not in board[2]:
            return -1
        else:
            return 0
