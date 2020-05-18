from Agents.TrainableAgents.QAgents.QAgent import QAgent
import random
##############################################################################################

class DeepQAgent(QAgent):
    """
    Abstract Neural Network Class To predict QValue
    """
    def __init__(self, learning_rate, decay_rate, min_replay_to_fit=1_001, minibatch_size=1_000, avoid_assist=False, win=False, block=False, model_path="models/DeepQAgents", model_name="DQA"):
        QAgent.__init__(self, learning_rate, decay_rate, model_path, model_name)
        self.minibatch_size = minibatch_size
        self.min_replay_to_fit=min_replay_to_fit


    def estimate_from_state_action(self, state, action):
        raise NotImplementedError

    def learn(self, **kwargs):
        """
        Performs a Single Neural Net fit on a random sample of minibatch
        """
        queue = kwargs.get('queue', None)
        if queue is None:
            raise ValueError("Called learn on empty input")
        if len(queue) < self.min_replay_to_fit:
            print(f"DID NOT FIT because queue length {len(queue)}")
            return
        #print(len(queue), self.minibatch_size, self.min_replay_to_fit)
        minibatch = random.sample(queue, self.minibatch_size)
        self.model.train(minibatch, self.decay_rate)
        return

    def save(self, **kwargs):
        """
        Saves model to the path given
        """
        import os
        save_name = self.model_name
        save_path = self.model_path
        if kwargs.get('model_name', None) is not None:
            save_name = kwargs.get('model_name')
        if kwargs.get('model_path', None) is not None:
            save_path = kwargs.get('model_path')
        final_path = os.path.join(save_path, save_name)
        self.model.save(final_path)
        return

    def load(self):
        """
        Loads model if its in the path provided
        """
        import os
        final_path = os.path.join(self.model_path, self.model_name)
        self.model.load(final_path)
        return

    def get_model(self, **kwargs):
        raise NotImplementedError

    def estimate_from_q_vector(self, q_vector):
        return self.estimate_from_state_action(q_vector[0], q_vector[1])

    def estimate_from_state_action(self, state, action):
        pred = self.model.predict([state], actions=[action])[0]
        return pred

##############################################################################################
from Agents.Models.NeuralNetworks import ConvNet

class ConvAgent(DeepQAgent):
    def __init__(self, learning_rate, decay_rate, min_replay_to_fit=1_000, minibatch_size=1000, model_path="models/DeepQAgents", model_name="ConvAgent"):
        DeepQAgent.__init__(self, learning_rate, decay_rate, min_replay_to_fit=min_replay_to_fit, minibatch_size=minibatch_size, model_path=model_path, model_name=model_name)
        self.model = ConvNet(update_every=500)