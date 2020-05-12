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
