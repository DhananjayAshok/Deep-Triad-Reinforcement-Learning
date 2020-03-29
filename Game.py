class GameEnvironment(object):
    """
    This object will contain all the working of the game environment but not the game itself. It must have the following functionalities
    

    Will probably need the following variables
        Game object -> to actual have a board and make moves
        MoveStack -> a stack of all moves that were played so it can be undone if we want


    """
    def reset(human_players):
        """
        will randomly assign order and present an empty field. If the first k players are opponent AIs it will return a state with k moves played.
        """
    def step(action):
        """
        will take in an action from the agent and then play 2 other moves (either opponent AI or Random or human) and then return - new_state vector, rewards, done
        """

    def undo_step():
        """
        For use only in slow plays
        will return the last k moves until it reaches a human player or the agent. 
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

    Will probably need the following variables:
        Matrix Object -> a 3 * 3 * 3 matrix to store the game 
    """
    def restart():
        """
        resets its internal representation
        """

    def get_board():
        """
        returns the board as it currently is
        """

    def is_illegal(move):
        """
        returns true iff the proposed move is illegal given the board at its current state
        """

    def play(move, player):
        """
        make a move with that player and return winner - (-1 if draw, 0 if game is still going on, else x if player x has won)
        """



"""
Utility Functions
    
"""
def checkforwin(board, proposed_move, proposed_player):
    """
    returns winner as required by the above play function but its hypothetical so doesn't change anything. Useful for AI logic
    """