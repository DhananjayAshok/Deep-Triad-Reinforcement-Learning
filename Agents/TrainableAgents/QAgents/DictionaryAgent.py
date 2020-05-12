#Core Imports Here
from Agents.TrainableAgents.QAgents.QAgent import QAgent
from Configs import ACTION_CLASS
import os
import pickle
import numpy as np

##############################################################################################

class DictionaryAgent(QAgent):
    """
    Supposed to be trained with a BatchLearning Gym with a clear_every_episode set to True
    Has a smart learning function where whenever it sees an illegal move it tries to learn that all similar configurations are also illegal
    """
    def __init__(self, learning_rate, decay_rate, model_path="models/Dictionary", model_name="dict"):
        QAgent.__init__(self, learning_rate, decay_rate, model_path, model_name)
        self.d = {}

    def estimate_from_q_vector(self, q_vector):
        key = tuple(np.append(q_vector[0].get_data(), q_vector[1].get_data()))
        return self.d.get(key, 0)

    def learn(self, **kwargs):
        """
        Updates value such that dict[tuple(q_vector)] = reward + q_value(next_state)
        """
        queue = kwargs.get("queue", None)
        heavy_learn = kwargs.get("heavy_learn", False)

        for i, (state, action, reward, next_state, done) in enumerate(queue):
            current_vec = self.create_q_vector(state, action)
            vectuple = tuple(np.append(current_vec[0].get_data(), current_vec[1].get_data()))
            if done:
                self.d[vectuple] = reward
            else:
                next_actions = ACTION_CLASS.get_action_space()
                values = []
                for act in next_actions:
                    next_vec = self.create_q_vector(next_state, act)
                    temp_vectuple = tuple(np.append(next_vec[0].get_data(), next_vec[1].get_data()))
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


# Implement Your Custom Classes Below
##############################################################################################



