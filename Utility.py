import numpy as np
from GameSystem.game import Game


def MaxN(state):
    """
    returns a tuple --> (players evaluation,move to be made)
    players evaluation --> largest number of n-in-a-row peices, that player has
    returns -11 for illegal move, 3 for win and 0 for loss
    """
    current_player = state[27]
    next_player = state[28]
    board = np.reshape(state[:27],(3,3,3)).copy()
    game=Game()
    game.matrix = board.copy()
    for action in range(1,10):
        if game.is_legal(action):
            #check terminal state --> Win/Loss/Draw
            winner = game.check_for_win(action, current_player)
            if winner==-1 or winner==0:
                #return the tuple (static evaluation of this terminal state,move)
                return (game.get_player_eval(current_player),None)
            else:
                #look at possible moves (MaxN)
                current_player = next_player
                next_player = next_player%3+1
                new_state = convert_data_to_state(game, current_player, next_player)
                result=-10
                evaluation=MaxN(new_state)
                best_result=evaluation[0]
                if best_result>result:
                    result,best_action=best_result,action
        else:
            #illegal move so worst reward (-11)
            return (-11,None)
    return (best_result,best_action)


def convert_data_to_state(game, current_player, next_player):
    board = game.matrix.copy()
    return np.append(board, [current_player, next_player])

def random_highest_index(scores):
    """
    Returns a random index out of all the maximum values of scores
    """
    max_indexes = np.where(np.asarray(scores) == max(scores))[0]
    return np.random.choice(max_indexes)