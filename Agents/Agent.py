import numpy as np
import os
import pickle
from GameSystem.Environments import TicTacToeEnvironment
from GameSystem.Games import TicTacToeGame
from GameSystem.Actions import TicTacToeAction
from GameSystem.States import TicTacToeState

class Agent(object):
    """
    Abstract Base Agent Class
    """
    def play(self, state, **kwargs):
        raise NotImplementedError

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


class QAgent(TrainableAgent):
    """
    Assumes q_vector is simply state, action

    """
    def __init__(self, learning_rate, decay_rate, model_path = "models/QAgents", model_name="QAgent"):
        TrainableAgent.__init__(self, learning_rate, decay_rate, model_path, model_name)


    def play(self, state, real_epsilon=0.0, avoid_illegal = True, verbose=False, **kwargs):
        """
        Epsilon Greedy
        Manually Avoids Illegal Moves if avoid_illegal is True
        """
        choices = []
        for act in range(1, 10):
            if self.g.is_legal(act, state=state) or not avoid_illegal:
                choices.append(act)

        if np.random.random() > real_epsilon:
            scores = [self.estimate_from_state_action(state, act) for act in choices]
            max_indexes = np.where(np.asarray(scores) == max(scores))[0]
            if verbose:
                s = ""
                for i, score in enumerate(scores):
                    s = s + f"Action {choices[i]} has score: {score}\n"
                print(s)
            if np.isnan(scores[0]):
                raise ArithmeticError(f"Scores are {scores}. There are exploding gradients and you are getting NaN values")
            return choices[np.random.choice(max_indexes)]

        else:
            return np.random.choice(choices)

    def estimate_from_state_action(self, state, action):
        return self.estimate_from_q_vector(self.create_q_vector(state, action))

    def create_q_vector(self, state, action):
        """
        Retursn x(S,A) as described above
        """
        return np.append(state, action)

    def estimate_from_q_vector(self, q_vector):
        try:
            return self.model.predict([np.append(q_vector[0], q_vector[1])])[0]
        except:
            print(f"Could not predict (Likely because model was not fitted or because model does not exist) returned 0")
            return 0



class DictionaryAgent(QAgent):
    """
    Supposed to be trained with a BatchLearning Gym with a clear_every_episode set to True
    Has a smart learning function where whenever it sees an illegal move it tries to learn that all similar configurations are also illegal
    """
    def __init__(self, learning_rate, decay_rate, model_path="models/Dictionary", model_name="dict"):
        QAgent.__init__(self, learning_rate, decay_rate, model_path, model_name)
        self.d = {}

    def estimate_from_q_vector(self, q_vector):
        key = tuple(np.append(q_vector[0], q_vector[1]))
        return self.d.get(key, 0)

    def learn(self, **kwargs):
        """
        Updates value such that dict[tuple(q_vector)] = reward + q_value(next_state)
        """
        queue = kwargs.get("queue", None)
        heavy_learn = kwargs.get("heavy_learn", False)

        for i, (state, action, reward, next_state, done) in enumerate(queue):
            current_vec = self.create_q_vector(state, action)
            vectuple = tuple(np.append(current_vec[0], current_vec[1]))
            if done:
                self.d[vectuple] = reward
            else:
                next_actions = range(1, 10)
                values = []
                for act in next_actions:
                    next_vec = self.create_q_vector(next_state, act)
                    temp_vectuple = tuple(np.append(next_vec[0], next_vec[1]))
                    values.append(self.estimate_from_q_vector(next_vec))
                if reward <= -10:
                    pass
                else:
                    self.d[vectuple] = reward + self.decay_rate* max(values)
                if kwargs.get('heavylearn', False):
                    self.heavy_learn(state)
                    self.heavy_learn(next_state)

        return

    def save(self, **kwargs):
        """
        Model name optional parameter
        """
        save_name = self.model_name
        if kwargs.get('model_name', None) is not None:
            save_name = kwargs.get('model_name')
        with open(os.path.join(self.model_path, save_name), 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.d, f, pickle.HIGHEST_PROTOCOL)

    def load(self, **kwargs):
        save_name = self.model_name
        if kwargs.get('model_name', None) is not None:
            save_name = kwargs.get('model_name')
        final = os.path.join(self.model_path, save_name)
        with open(final, 'rb') as f:
            self.d = pickle.load(f)
        print(f"Loaded without error - Number of estimates is {len(self.d)}")
        return

    def stats(self):
        """
        Meant to analyze the dictionary and return statistics
        """
        raise NotImplementedError

    def heavy_learn(self, state, **kwargs):
        """
        Manually Learn a set of states values from a given state
        """
        raise NNotImplementedError