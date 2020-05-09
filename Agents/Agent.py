
class Agent(object):
    """
    Abstract Base Agent Class
    """
    def play(self, state, **kwargs):
        raise NotImplementedError


