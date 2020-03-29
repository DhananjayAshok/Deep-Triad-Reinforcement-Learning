class Opponent(object):
    """
    Will be an abstract parent class for various other children who implement different strategies
    """
    def play(state):
        """
        Given the state of the game return a move that the opponent plays
        """
        raise NotImplementedError


    def blocking_move(state):
        """
        If the next player has any move that wins them the game return a move that blocks at least one of those moves else returns -1
        """

    def winning_move(state):
        """
        If there is an object that will win the AI the game this move it returns the move that wins else it returns -1
        """

class SelfOpponent(Opponent):
    """
    Is an Opponent object for use to train the Agent against a former version of itself
    """
    pass


class HumanOpponent(Opponent):
    """
    Is an Opponent object for use to test the Agent against a human
    """
    pass


class RandomOpponent(Opponent):
    """
    Is an Opponent that plays truly random moves (that are legal)
    Can be parameterized to also
        block immediate winning moves from the next player
        win if move exists
    """
    pass


class HyperionOpponent(Opponent):
    """
    Plays against Hyperion the Greedy
    """
    pass