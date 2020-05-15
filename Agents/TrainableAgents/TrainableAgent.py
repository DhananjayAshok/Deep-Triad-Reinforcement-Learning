from Agents.Agent import Agent
from GameSystem.Actions import Connect4Action
from GameSystem.Games import Connect4Game
from GameSystem.Environments import Connect4Environment

class TrainableAgent(Agent):
    """
    Subclass of agents that are meant to go through the training process
    """
    def __init__(self, learning_rate=1, decay_rate=0.2, model_path="models/TrainableAgents", model_name="TrainableAgent"):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.model_name = model_name
        self.model_path = model_path
        self.g = Connect4Game()

    def learn(self, dataset, **kwargs):
        raise NotImplementedError

    def save(self, **kwargs):
        raise NotImplementedError

    def load(self, **kwargs):
        raise NotImplementedError

