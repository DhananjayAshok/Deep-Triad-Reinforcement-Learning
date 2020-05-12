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
