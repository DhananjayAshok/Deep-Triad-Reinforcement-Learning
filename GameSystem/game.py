import numpy as np
from .Line import get_win_lines

class GameEnvironment(object):
    """
    This object will contain all the working of the game environment but not the game itself. It must have the following functionalities
    

    Will probably need the following variables
        Game object -> to actual have a board and make moves
        MoveStack -> a stack of all moves that were played so it can be undone if we want

    Currently thinking the following will be the state vector
    [piece in 1st cell, ..... piece in 27th cell, who's turn is it, who plays next]


    """

    def __init__(self):
        """
        """
        self.g = Game()

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
            return self.get_state(), -1000000000, False

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
        if winner == -1:
            reward = 0
            done = True
        elif winner == 0:
            reward = 0
            done = False
        elif winner == self.agent_turn:
            reward = 5
            done = True
        else:
            reward = -5
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

    def play_slow(self, agent, opponent_1, opponent_2, avoid_illegal=True):
        """
        will create a new game but to be played slowly where each turn is only completed if a human presses an input
        """
        self.reset(opponent_1, opponent_2)
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
        board = self.g.get_board()
        board = np.reshape(board, (27,))
        board = np.append(board, [self.turn, self.turn%3+1])
        return board

    def print_state(self, provided=None):
        """
        will print the state vector in a way that humans can read
        """
        if provided is None:
            temp = self.get_state()
        else:
            temp = provided
        board = temp[0:27].reshape((3,3,3))
        turn = temp[27]
        next = temp[-1]
        print(f"Board\n {board[2]} \n \t3 \n {board[1]} \n \t2 \n {board[0]} \n \t1")
        print(f"Current Turn - {turn}| Next turn - {next}")


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

    def play(self, move, player, board = None):
        """
        make a move with that player and return winner - (-1 if draw, 0 if game is still going on, else x if player x has won)
        """
        target_matrix = self.matrix
        if board is not None:
            target_matrix = board
        else:
            self.move_stack.append(move)
        decoded = self._decode_move(move, board=target_matrix)
        winner = self.check_for_win(move, player, board=target_matrix)
        target_matrix[decoded[0], decoded[1], decoded[2]] = player
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
            if (line.point0 == [0,0,0] and line.point1 == [1,0,0] and line.point2 == [2,0,0]):
                #print(f"Yup the thing is here and in this case we get the if condition is {(self.piece_at(line.point0, board) == self.piece_at(line.point1, board) == self.piece_at(line.point2, board)) and self.piece_at(line.point0, board)!=0)} because")
                #print(self.piece_at(line.point0, board), self.piece_at(line.point1, board), self.piece_at(line.point2, board))
                pass
            if(self.piece_at(line.point0, board) == self.piece_at(line.point1, board) == self.piece_at(line.point2, board)) and self.piece_at(line.point0, board)!=0:
                return self.piece_at(line.point0, board)
        if 0 not in board[2]:
            return -1
        else:
            return 0

    def get_attack_score(self, proposed_move, proposed_player, board=None):
        """
        This assumes that the move is legal
        """
        if board is None:
            board = self.matrix.copy()
        else:
            board = board.copy()
        move = self._decode_move(proposed_move, board)
        board[move[0], move[1], move[2]] = proposed_player

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




