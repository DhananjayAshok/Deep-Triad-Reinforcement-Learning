from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.models import load_model
from keras.initializers import Zeros
from collections import deque
from GameSystem.game import Game

class NeuralNetwork(object):
    """description of class"""

class DeepQNetwork(object):
    """
    Implements Replay Memory Learning in Neural Network
    """
    def __init__(self, update_every=1000, avoid_assist=False, win_assist=False, block_assist=False):
        """
        Creates a model with architecture
        INPUT(32) -> Dense(64) -> Dense(1) 
        NEEDS
            save
            load
            predict from state vector
            train from input minibatch of format [state, action, reward, new_state, done]

        """
        self.avoid_assist = avoid_assist
        self.win_assist = win_assist
        self.block_assist = block_assist
        self.g = Game()
        self.main_model = self.create_model()
        self.prev_model = self.create_model()
        self.prev_model.set_weights(self.main_model.get_weights())
        self.update_counter = 0
        self.update_every = 1000


    def update_prev_model(self):
        self.prev_model.set_weights(self.main_model.get_weights())
        self.update_counter = 0

    def update_prev_model_checks(self):
        """
        Checks if we should update prev model
        """
        return self.update_counter >= self.update_every


    def create_model(self):
        """
        Returns a model of the following archtiecture
        """
        model = Sequential()
        model.add(Dense(32, activation="elu", input_shape=(29,), kernel_initializer=Zeros()))
        model.add(Dense(64, activation="elu", kernel_initializer=Zeros()))
        model.add(Dense(32, activation="elu", kernel_initializer=Zeros()))
        model.add(Dense(9, activation="linear", kernel_initializer=Zeros()))
        model.compile("adam", loss="mse", metrics=["accuracy"])
        return model

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
        states = [transition[0] for transition in minibatch]
        next_states = [transition[3] for transition in minibatch]
        q_targets = self.prev_model.predict(np.array(states))
        next_q_values = self.prev_model.predict(np.array(next_states))

        X = []
        y = []
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            X.append(state)

            q_target = q_targets[i]
            if not done:
                q_target[action-1] = reward + discount * max(next_q_values[i])
            else:
                q_target[action-1] = reward
            if self.avoid_assist:
                q_target = self.avoid_assistance(state, q_target)
            if self.win_assist:
                q_target = self.win_assistance(state, q_target)
            if self.block_assist:
                q_target = self.win_assistance(state, q_target)

            y.append(q_target)

        X = np.array(X, dtype=np.int8)
        y = np.array(y)

        
        self.main_model.fit(X, y, batch_size = len(minibatch), verbose=0)
        self.update_counter+=1
        if self.update_prev_model_checks():
            self.update_prev_model()

    def avoid_assistance(self, state, q_target):
        """
        Returns a q_target list with very low values for an illegal action
        """
        board = state[:27].reshape((3,3,3))
        illegals = []
        for action in range(1, 10):
            if not self.g.is_legal(action, board=board):
                illegals.append(action-1)
        for illegal in illegals:
            q_target[illegal] = -1000000
        return q_target

    def win_assistance(self, state, q_target):
        """
        Returns a q_target with a reward of 5 for all states that win
        """
        board = state[:27].reshape((3,3,3))
        curr = state[27]
        winners = []
        for action in range(1, 10):
            if self.g.is_legal(action, board=board) and self.g.check_for_win(action, curr, board = board) == curr:
                winners.append(action-1)
        for winner in winners:
            q_target[winner] = 5
        return q_target

    def block_assistance(self, state, q_target):
        """
        Does an advanced block assistance i.e looks if after a move a the opponent can win
        Returns a q_target with a reward of -5 for all states that lose to the next player
        Does this only if after that action the game doesn't end
        """
        board = state[:27].reshape((3,3,3))
        curr = state[27]
        next = state[28]
        losers = []
        for action in range(1, 10):
            board = state[:27].reshape((3,3,3))
            if not self.g.is_legal(action, board=board):
                continue
            else:
                w = self.g.play(action, curr, board=board)
                if w != 0:
                    continue
                else:
                    for act in range(1,10):
                        if self.g.is_legal(act, board=board) and self.g.check_for_win(act, next, board=board) == next:
                            losers.append(action-1)
                            break
        for loser in losers:
            q_target[winner] = -5
        return q_target

    def predict(self, X):
        """
        Returns the predictions as per the main model
        """
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