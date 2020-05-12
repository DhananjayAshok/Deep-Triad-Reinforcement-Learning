#Core Imports Here
##############################################################################################

class State(object):
    """Abstract base state class"""
    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        return str(self)

    def get_data(self):
        """
        Returns the details of the state that is relevant in some form described by the User
        """
        raise NotImplementedError


# Implement Your Custom Classes Below
##############################################################################################



