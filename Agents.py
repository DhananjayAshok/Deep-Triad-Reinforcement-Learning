import numpy as np
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

class TrainableAgent(Agent):
    """
    Subclass of agents that are meant to go through the training process
    """
    def __init__(self, learning_rate, decay_rate, model_name):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.model_name = model_name
    
    def update_from_state_action(self, state, action, target, prediction):
        raise NotImplementedError

    def estimate_from_state_action(self, state, action):
        raise NotImplementedError

    def play(self, state, real_epsilon=0.0):
        """
        Epsilon Greedy
        """
        if np.random.random() > real_epsilon:
            choices = [i for i in range(1,10)]
            scores = [self.estimate_from_state_action(state, i) for i in choices]
            max_indexes = np.where(np.asarray(scores) == max(scores))[0]
            #print(f"Scores are {scores}")
            if np.isnan(scores[0]):
                raise ArithmeticError(f"Scores are {scores}. There are exploding gradients and you are getting NaN values")
            return np.random.choice(max_indexes) + 1     
        else:
            return np.random.choice(range(1,10))

    def learn(self, dataset):
        raise NotImplementedError

    def save_model(self, path, model_name):
        pass

class QLinearAgent(TrainableAgent):
    """
    We start with a naive approach of just x(S,A) = state vector + [action]
    We make our Q-Value approximator -> x(S,A)T * w = sum(from 1 to n)(x(S,A)[i]*w[i]) i.e a linear function approximator
    We use a stochastic Gradient Descent Update
    """
    def __init__(self, learning_rate, decay_rate, model_name="QLinearAgent"):
        TrainableAgent.__init__(self, learning_rate, decay_rate, model_name)
        self.w = np.random.uniform(low=1, high=2,size=(30,)).astype(np.longdouble)

    def estimate_from_state_action(self, state, action):
        return self.estimate_from_q_vector(self.create_q_vector(state, action))

    def estimate_from_q_vector(self, q_vector):
        value = 0
        for i in range(len(self.w)):
            value += q_vector[i]*self.w[i]
        return value

    
    def create_q_vector(self, state, action):
        """
        Retursn x(S,A) as described above
        """
        return np.append(state, action)

    def update(self, vector, target, prediction):
        """
        target and prediction should be scalars
        """
        error = target - prediction
        self.w += self.learning_rate * error * vector

    def update_from_state_action(self, state, action, target, prediction):
        return self.update(self.create_q_vector(state, action), target, prediction)

    def learn(self, dataset):
        """
        Performs batch updates with that dataset
        """
        for X, y in dataset:
            self.update(X, y, self.estimate_from_q_vector(X))

    def save_model(self, path, model_name=None):
        """
        Save model to the path - path + model_name + _weights.npy
        """
        import os
        save_name = self.model_name
        if mode_name is not None:
            save_name = model_name
        final_path = os.path.join(path, save_name + "_weights.npy")
        np.save(final_path, self.w)

    def load_model(self, path):
        """
        Loads weights from a given path
        """
        import os
        self.w = np.load(os.path.join(path, self.model_name+"_weights.npy"))
        print(f"Model {self.model_name} imported without issues")

