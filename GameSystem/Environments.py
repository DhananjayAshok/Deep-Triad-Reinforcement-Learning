#region
#Core Imports Here

##############################################################################################

class Environment(object):
    """
    Abstract Base Class to store the most generic possible Environment.
    All Environments will be children of this class
    """
    def reset(self, **kwargs):
        raise NotImplementedError

    def step(self, **kwargs):
        raise NotImplementedError

    def play_slow(self, **kwargs):
        """
        Meant to play the game at a pace which human debuggers can comprehend
        """
        raise NotImplementedError

    def get_state(self, **kwargs):
        raise NotImplementedError


# Implement Your Custom Classes Below
##############################################################################################
#endregion
