from Agents.Agent import Agent
from GameSystem.Games import TicTacToeGame

class TrainableAgent(Agent):
    """
    Subclass of agents that are meant to go through the training process
    """
    def __init__(self, learning_rate=1, decay_rate=0.2, model_path="models/TrainableAgents", model_name="TrainableAgent"):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.model_name = model_name
        self.model_path = model_path
        self.g = TicTacToeGame()

    def learn(self, dataset, **kwargs):
        raise NotImplementedError

    def save(self, **kwargs):
        raise NotImplementedError

    def load(self, **kwargs):
        raise NotImplementedError

