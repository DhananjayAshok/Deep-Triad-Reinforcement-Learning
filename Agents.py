import numpy as np
from Opponents import HumanOpponent, RandomOpponent, HyperionOpponent
from GameSystem.game import Game

class Agent(object):
    """
    Parent Class for all Agents
    
    """
    def play(self, state):
        raise NotImplementedError

class HumanAgent(Agent):
    def __init__(self):
        self.human = HumanOpponent()

    def play(self, state, real_epsilon=0.5):
        return self.human.play(state)

class RandomAgent(Agent):
    def __init__(self, blocking=True, winning=True):
        self.random = RandomOpponent(blocking=blocking, winning=winning)

    def play(self, state, real_epsilon=0.5):
        return self.random.play(state)

class HyperionAgent(Agent):
    def __init__(self):
        self.hyp = HyperionOpponent()

    def play(self, state, real_epsilon=0.5):
        return self.hyp.play(state)


class TrainableAgent(Agent):
    """
    Subclass of agents that are meant to go through the training process
    """
    def __init__(self, learning_rate, decay_rate, model_name):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.model_name = model_name
        self.g = Game()
    
    def update_from_state_action(self, state, action, target, prediction):
        raise NotImplementedError

    def estimate_from_state_action(self, state, action):
        raise NotImplementedError

    def play(self, state, real_epsilon=0.0):
        """
        Epsilon Greedy
        Manually Avoids Illegal Moves
        """
        choices = []
        for i in range(1,10):
            if self.g.is_legal(i, state[:27].reshape((3,3,3))):
                choices.append(i)
        
        if np.random.random() > real_epsilon:
            scores = [self.estimate_from_state_action(state, i) for i in choices]
            max_indexes = np.where(np.asarray(scores) == max(scores))[0]
            #print(f"Scores are {scores}")
            if np.isnan(scores[0]):
                raise ArithmeticError(f"Scores are {scores}. There are exploding gradients and you are getting NaN values")
            return choices[np.random.choice(max_indexes)]     
        else:
            return np.random.choice(choices)

    def learn(self, dataset):
        raise NotImplementedError

    def save_model(self, path, model_name):
        pass


class QAgent(TrainableAgent):
    """
    We start with a naive approach of just x(S,A) = state vector + [action]
    """
    def __init__(self, learning_rate, decay_rate, model_name="QLinearAgent"):
        TrainableAgent.__init__(self, learning_rate, decay_rate, model_name)

    def estimate_from_state_action(self, state, action):
        return self.estimate_from_q_vector(self.create_q_vector(state, action))

    def create_q_vector(self, state, action):
        """
        Retursn x(S,A) as described above
        """
        return np.append(state, action)

    def estimate_from_q_vector(self, q_vector):
        try:
            return self.model.predict([q_vector])[0]
        except:
            print(f"Could not predict (Likely because model was not fitted) returned 0")
            return 0


class QSKLearnAgent(QAgent):
    """
    Is a class that works for any SKLearn model that used partial_fit
    """
    def __init__(self, learning_rate, decay_rate, model_name="QSKlearnAgent"):
        QAgent.__init__(self, learning_rate, decay_rate, model_name)
        self.model = self.get_unfitted_model()

    def get_unfitted_model(self):
        raise NotImplementedError

    def learn(self,X_train, y_train):
        """
        Performs batch updates with that dataset
        """
        self.model.partial_fit(X_train, y_train)

    def save_model(self, path, model_name=None):
        """
        Save model to the path - path + model_name + .sav
        """
        import os
        import pickle
        save_name = self.model_name
        if model_name is not None:
            save_name = model_name
        final_path = os.path.join(path, save_name + ".sav")
        pickle.dump(self.model, open(final_path, 'wb'))

    def load_model(self, path):
        """
        Loads weights from a given path
        """
        import os
        import pickle
        self.model = pickle.load(open(os.path.join(path, self.model_name+".sav"), 'rb'))
        print(f"Model {self.model_name} imported without issues")

class QLinearAgent(QSKLearnAgent):
    """
    Stochastic Gradient Descent with Linear Approximator (SKLearn SGD)
    """
    def __init__(self, learning_rate, decay_rate, model_name="QLinearAgent"):
        QSKLearnAgent.__init__(self, learning_rate, decay_rate, model_name)

    def get_unfitted_model(self):
        from sklearn.linear_model import SGDRegressor
        return SGDRegressor()

