#Core Imports Here
##############################################################################################

class Action(object):
    """Abstract base action class"""
    def get_action_space():
        """
        Returns a list of all possible actions
        Not always applicable
        """
        raise NotImplementedError

    def get_data(self):
        """
        Returns the details of the action that is relevant in some form described by the User
        """
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        return str(self)

# Implement Your Custom Classes Below
##############################################################################################
class Connect4Action(Action):
    def get_action_space():
        return [Connect4Action(i) for i in range(6)]

    def __init__(self, act):
        self.act = act
        Action.__init__(self)

    def get_data(self):
        return self.act

    def __str__(self):
        return str(self.act)