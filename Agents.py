from Opponents import HumanOpponent
class Agent(object):
    """
    Parent Class for all Agents
    
    """
    def play(self, state):
        raise NotImplementedError

class HumanAgent(Agent):
    def __init__(self):
        self.human = HumanOpponent()

    def play(self, state):
        return self.human.play(state)

