#Core Imports Here
##############################################################################################

class Game(object):
    """
    Base Class for all games to inherit from 
    """
    def __init__(self, **kwargs):
        self.restart(**kwargs)

    def restart(self, **kwargs):
        """
        Restart the game
        """
        raise NotImplementedError

    def play(self, action, provided_state=None, **kwargs):
        raise NotImplementedError

    def is_legal(self, action, provided_state=None, **kwargs):
        raise NotImplementedError
# Implement Your Custom Classes Below
##############################################################################################
import numpy as np
from Utility import get_state_data

class TicTacToeGame(Game):
    """
    This object will have the board representation and implement all the basic rules of the games. Will need the following methods:

    Matrix is of dimensions [height layer, row layer, column layer ]
    """
    def __init__(self):
        Game.__init__(self)
        self.restart()
        self.win_lines = get_win_lines()

    def restart(self):
        """
        resets its internal representation
        """
        self.matrix = np.zeros(shape=(3,3), dtype=np.int8)


    def get_board(self):
        """
        returns the board as it currently is
        """
        return self.matrix

    def is_legal(self, action, board=None, state=None):
        """
        returns true iff the proposed move is legal given the board at its current state
        """
        if not (1 <= action <= 9):
            return False
        else:
            if state is not None:
                board, turn, next = get_state_data(state)
            decoded = self._decode_action(action, board)
            return decoded[0] <= 0

    def play(self, action, player, board = None):
        """
        make a move with that player and return winner - (-1 if draw, 0 if game is still going on, else x if player x has won)
        """
        target_matrix = self.matrix
        if board is not None:
            target_matrix = board
        decoded = self._decode_action(action, board=target_matrix)
        winner = self.check_for_win(action, player, board=target_matrix)
        target_matrix[decoded[1], decoded[2]] = player
        return winner

    def _decode_action(self, action, board=None):
        """
        Takes in a move and returns [height, row col] that is intended
        """
        move = action
        if board is None:
            board = self.matrix
        col = (move-1)%3
        if move > 6:
            row = 2
        elif move > 3:
            row = 1
        else:
            row = 0
        if board[row, col] == 0:
            height = 0
        else:
            height = 3
        return [height, row, col]

    def piece_at(self, point, board=None):
        """
        Returns the piece at that point
        """
        if board is None:
            board = self.matrix
        return board[point[1], point[2]]

    def check_for_win(self, proposed_action, proposed_player, board=None):
        """
        returns winner as required by the above play function but its hypothetical so doesn't change anything. Useful for AI logic
        Prerequisite: Before checkforwin is called there is no winner
        """
        if board is None:
            board = self.matrix.copy()
        else:
            board = board.copy()

        move = self._decode_action(proposed_action, board)
        board[move[1], move[2]] = proposed_player

        for line in self.win_lines:
            if(self.piece_at(line.point0, board) == self.piece_at(line.point1, board) == self.piece_at(line.point2, board)) and self.piece_at(line.point0, board)!=0:
                return self.piece_at(line.point0, board)
        if 0 not in board:
            return -1
        else:
            return 0

    def get_attack_score(self, proposed_action, proposed_player, board=None):
        """
        This assumes that the move is legal
        """
        if board is None:
            board = self.matrix.copy()
        else:
            board = board.copy()
        move = self._decode_action(proposed_action, board)
        board[move[1], move[2]] = proposed_player

        score = 1
        for line in self.win_lines:
            miniscore = 1
            if self.piece_at(line.point0, board)!=0  and self.piece_at(line.point0, board)!= proposed_player:
                pass
            elif self.piece_at(line.point1, board) != 0 and self.piece_at(line.point1, board) != proposed_player:
                pass
            elif self.piece_at(line.point2, board) != 0 and self.piece_at(line.point2, board) != proposed_player:
                pass
            else:
                counter = 0
                for l in [line.point0, line.point1, line.point2]:
                    if self.piece_at(l, board) == proposed_player:
                        counter += 1
                if counter == 0:
                    miniscore = 1
                if counter == 1:
                    miniscore = 2
                if counter == 2:
                    miniscore = 5
                if counter == 3:
                    miniscore = 10000
            score *= miniscore
        return score

    def get_player_eval(self,proposed_player, board=None):
        """
        This assumes that the move is legal and returns an integer value for players evaluation
        """
        if board is None:
            board = self.matrix.copy()
        else:
            board = board.copy()
        for line in self.win_lines:
            if self.piece_at(line.point0, board)!=0  and self.piece_at(line.point0, board)!= proposed_player:
                pass
            elif self.piece_at(line.point1, board) != 0 and self.piece_at(line.point1, board) != proposed_player:
                pass
            elif self.piece_at(line.point2, board) != 0 and self.piece_at(line.point2, board) != proposed_player:
                pass
            else:
                counter = 0
                for l in [line.point0, line.point1, line.point2]:
                    if self.piece_at(l, board) == proposed_player:
                        counter += 1
        return counter

class Line(object):
    """
    Convinience Class
    """
    def __init__(self, start, direction):
        self.point0 = start
        self.point1 = self.add(start, direction)
        self.point2 = self.add(start, self.scale(direction, 2))
      

    def add(self, a, b):
        return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
    def scale(self, a, c):
        return [c*a[0], c*a[1], c*a[2]]

    def onLine(self, point):
        if point == point0 or point == point1 or point == point2:
            return True
        else:
            return False

    def __str__(self):
        return f"Line: Start - {self.point0}, mid {self.point1}, end {self.point2}"
    
def get_win_lines():
    """
    Returns a list of all winning lines in the board
    """
    win_lines = []
    win_lines.append(Line([0,0,0], [0,0,1]))
    win_lines.append(Line([0,1,0], [0,0,1]))
    win_lines.append(Line([0,2,0], [0,0,1]))
    win_lines.append(Line([0,0,0], [0,1,0]))
    win_lines.append(Line([0,0,1], [0,1,0]))
    win_lines.append(Line([0,0,2], [0,1,0]))
    win_lines.append(Line([0,0,0], [0,1,1]))
    win_lines.append(Line([0,0,2], [0,1,-1]))

    return win_lines
