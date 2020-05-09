from .Agent import Agent

class OpponentAgent(Agent):
    """
    Creates an Agent that mimics an opponent class.
    """
    def __init__(self, OPPONENT_CLASS, **kwargs):
        Agent.__init__(self)
        self.opp = OPPONENT_CLASS(**kwargs)

    def play(self, state, **kwargs):
        return self.opp.play(state)




