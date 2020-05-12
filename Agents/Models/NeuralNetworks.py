#Core Imports Here
from Configs import GAME_CLASS
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input
from keras.initializers import Zeros
import numpy as np
from Configs import ACTION_CLASS
##############################################################################################
class DeepQNetwork(object):
    """
    Implements Replay Memory Learning in Neural Network
    """
    def __init__(self, update_every=1000):
        """
        NEEDS
            save
            load
            predict from state vector
            train from input minibatch of format [state, action, reward, new_state, done]

        """
        self.g = GAME_CLASS()
        self.update_counter = 0
        self.update_every = 1000
        self.main_model = self.create_model()
        self.prev_model = self.create_model()
        self.prev_model.set_weights(self.main_model.get_weights())

    def update_prev_model(self):
        self.prev_model.set_weights(self.main_model.get_weights())
        self.update_counter = 0

    def update_prev_model_checks(self):
        """
        Checks if we should update prev model
        """
        return self.update_counter >= self.update_every

    def train(self, minibatch, discount):
        """
        For every data point in the minibatch does the following:
            estimate the value of each action of the current_state using the prev_model
            estimate the value of each action of next_state using prev_model and get the max of that
            For the action that was taken set the target q value to be return + discount * max(q_values of next state) rest is as predicted
            use the target as Y and states as X
        then fit to the main model
        Check to see if prev model needs to be updated
        """
        X = []
        y = []
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            X.append(self.prepare_X(state=state, action=action))
            y.append(self.prepare_y(state=state, reward=reward, next_state=next_state, discount=discount, done=done))
        X = np.array(X, dtype=np.int8)
        y = np.array(y)

        
        self.main_model.fit(X, y, batch_size = len(minibatch), verbose=0)
        self.update_counter+=1
        if self.update_prev_model_checks():
            self.update_prev_model()

    def predict(self, states, **kwargs):
        """
        Returns the predictions as per the main model
        states is just the list of state representations
        actions provided is a list of actions with same size as states
        """
        actions =  kwargs['actions']
        X = []
        for i, state in enumerate(states):
            X.append(self.transform_input(state , action=actions[i]))
        X = np.array(X)
        return self.main_model.predict(X)

    def save(self, final_path):
        self.main_model.save(final_path)

    def load(self, final_path):
        """
        Loads model into main model and updates prev
        """
        self.main_model = load_model(final_path)
        self.update_prev_model()
        return
    
    def prepare_X(self, **kwargs):
        """
        Use state action pair
        """
        state = kwargs['state']
        action = kwargs['action']
        return self.transform_input(state, action=action)

    def prepare_y(self, **kwargs):
        state = kwargs['state']
        next_state = kwargs['next_state']
        discount = kwargs['discount']
        reward = kwargs['reward']
        done = kwargs['done']
        if not done:
            next_values = []
            for act in ACTION_CLASS.get_action_space():
                inp = np.array([self.transform_input(state=next_state, action=act)])
                #print(inp.shape)
                pred = self.prev_model.predict(inp)
                next_values.append(pred[0])
            return reward + discount * max(next_values)
        else:
            return reward


    def create_model(self):
        raise NotImplementedError

    def transform_input(self, state, **kwargs):
        raise NotImplementedError

    def additional_q_target_processes(self, state, q_target):
        raise NotImplementedError


# Implement Your Custom Classes Below
##############################################################################################
