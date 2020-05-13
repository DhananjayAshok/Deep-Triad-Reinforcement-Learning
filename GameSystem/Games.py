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

class Connect4Game(Game):
    def __init__(self):
        self.winlines = get_winlines()
        Game.__init__(self)

    def restart(self):
        self.grid = np.zeros((6,6), dtype=np.int32)

    def play(self, action, player, provided_state=None, provided_board=None):
        """
        Play a move on the grid of the object and return a number to indicate the winner
        1 -> player 1
        2 -> player 2
        0 -> ongoing
        -1 -> draw

        priority given to provided_board
        """
        board = self.grid
        if provided_board is not None:
            board = provided_board
        elif provided_state is not None:
            board = provided_state.get_board()
        if self.is_legal(action, provided_state=provided_state):
            move_y, move_x = self.decode_action(action, provided_state)
            board[move_y, move_x] = player
        return self.winner(provided_board=board)

    def piece_at(self, point, provided_state=None, provided_board=None):
        """
        if both state and board are provided priority is given to board
        """
        board = self.grid
        if provided_board is not None:
            board = provided_board
        elif provided_state is not None:
            board = provided_state.get_board()
        return board[point[0], point[1]]

    def decode_action(self, action, provided_state=None):
        """
        Returns y, x coordinates of intended action
        """
        board = self.grid
        if provided_state is not None:
            board = provided_state.get_board()
        move = action.get_data()
        col = move
        top = 5
        while board[top, col] != 0 and top >= 0:
            top -= 1
        return top, col

    def is_legal(self, action, provided_state=None):
        """
        Returns true iff the following is a valid move
        """
        move = action.get_data()
        if not (0 <= move <= 5):
            return False

        board = self.grid
        if provided_state is not None:
            board = provided_state.get_board()
        col = move
        return board[0, col] == 0

    def winner(self, provided_state=None, provided_board=None):
        """
        Returns 1 or 2 if a player has won, -1 if draw and 0 if ongoing
        if both state and board are provided priority is given to board
        """
        board = self.grid
        if provided_board is not None:
            board = provided_board
        elif provided_state is not None:
            board = provided_state.get_board()

        for win in self.winlines:
            temp = self.piece_at(win.point0, provided_board=board)
            if temp != 0:
                if temp == self.piece_at(win.point1, provided_board=board) == self.piece_at(win.point2, provided_board=board) == self.piece_at(win.point3, provided_board=board):
                    return temp

        if 0 not in board:
            return -1

        return 0

    def check_for_win(self, action, proposed_player, provided_state=None):
        """
        Returns True iff the proposed action from the proposed player wins the game
        """
        if not self.is_legal(action, provided_state):
            return False

        temp = self.grid.copy()
        if provided_state is not None:
            temp = provided_state.get_board().copy()

        return self.play(action,proposed_player, provided_board=temp) == proposed_player


        



        




class Line:
    def __init__(self, start, direction):
        self.point0 = start
        self.point1 = self.add(start, direction)
        self.point2 = self.add(start, self.scale(direction, 2))
        self.point3 = self.add(start, self.scale(direction, 3))

    def add(self, v1, v2):
        """
        Adds two vectors
        """
        return [v1[0]+v2[0], v1[1]+v2[1]]

    def scale(self, v, c):
        """
        Scales the vector
        """
        return [v[0]*c, v[1]*c]


def get_winlines():
    """
    Returns all possible lines of victory
    """
    lines = []

    # All vertical downs
    dir = [1, 0]
    for item in [0, 1, 2]:
        for i in range(6):
            lines.append(Line([item, i], dir))

    #All horizontal rights
    dir = [0, 1]
    for item in [0, 1, 2]:
        for i in range(6):
            lines.append(Line([i, item], dir))

    # All diag down rights
    dir = [1, 1]
    for item in [0, 1, 2]:
        for i in [0, 1, 2]:
            lines.append(Line([item, i], dir))

    # All diag down lefts
    dir = [1, -1]
    for item in [0, 1, 2]:
        for i in [5, 4, 3]:
            lines.append(Line([item, i], dir))

    # In case

    #All vertical ups
    dir = [-1, 0]
    for item in [5, 4, 3]:
        for i in range(6):
            lines.append(Line([item, i], dir))

    # All horizontal lefts
    dir = [0, -1]
    for item in [5, 4, 3]:
        for i in range(6):
            lines.append(Line([i, item], dir))

    return lines