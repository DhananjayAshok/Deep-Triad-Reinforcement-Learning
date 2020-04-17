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
    def __init__(self, update_every=1000):
        """
        Creates a model with architecture
        INPUT(32) -> Dense(64) -> Dense(1) 
        NEEDS
            save
            load
            predict from state vector
            train from input minibatch of format [state, action, reward, new_state, done]

        """
        self.g = Game()
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
        
        states = np.array([self.transform_state(transition[0]) for transition in minibatch])
        next_states = np.array([self.transform_state(transition[0]) for transition in minibatch])
        q_targets = self.prev_model.predict(states)
        next_q_values = self.prev_model.predict(next_states)

        X = []
        y = []
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            X.append(self.transform_state(state))

            q_target = q_targets[i]
            if not done:
                q_target[action-1] = reward + discount * max(next_q_values[i])
            else:
                q_target[action-1] = reward

            q_target = self.additional_q_target_process(state, q_target)

            y.append(q_target)

        X = np.array(X, dtype=np.int8)
        y = np.array(y)

        
        self.main_model.fit(X, y, batch_size = len(minibatch), verbose=0)
        self.update_counter+=1
        if self.update_prev_model_checks():
            self.update_prev_model()

    def predict(self, X):
        """
        Returns the predictions as per the main model
        X is just the state representation
        """

        X = np.array([self.transform_state(i) for i in X])
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

    def create_model(self):
        raise NotImplementedError

    def transform_state(self, state):
        raise NotImplementedError
    def additional_q_target_processes(self, state, q_target):
        raise NotImplementedError


class NaiveNetwork(DeepQNetwork):
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
        DeepQNetwork.__init__(self, update_every)

    def transform_state(self, state):
        return state

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

    def additional_q_target_processes(self, state, q_target):
        if self.avoid_assist:
                q_target = self.avoid_assistance(state, q_target)
        if self.win_assist:
                q_target = self.win_assistance(state, q_target)
        if self.block_assist:
                q_target = self.win_assistance(state, q_target)
    def avoid_assistance(self, state, q_target):
        """
        Returns a q_target list with very low values for an illegal action
        """
        board = np.reshape(state[:27],(3,3,3))
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
        board = np.reshape(state[:27],(3,3,3))
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
        board = np.reshape(state[:27],(3,3,3))
        curr = state[27]
        next = state[28]
        losers = []
        for action in range(1, 10):
            board = np.reshape(state[:27],(3,3,3))
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

class AssistedNetwork(DeepQNetwork):
    def __init__(self, update_every=1000):
        """
        
        
        """
        self.main_model = self.create_model()
        self.prev_model = self.create_model()
        self.prev_model.set_weights(self.main_model.get_weights())
        DeepQNetwork.__init__(self, update_every)

    def create_model(self):
        """
        Architecture
        """
        model = Sequential()
        model.add(Dense(300, activation="relu", kernel_initializer=Zeros(), input_shape=(65,)))
        model.add(Dense(150, activation="relu"))
        model.add(Dense(50, activation="relu"))
        model.add(Dense(9, activation="linear", kernel_initializer=Zeros()))
        model.compile("adam", loss="mse", metrics=["accuracy"])
        return model

    def transform_state(self, state):
        """
        Will do the following feature transformation and addition
            Will add the following vector for every action 1 if the statements are true, 0 if false , [move_legal, legal and winning, legal and losing]
            Will also add the following vector for every action -ve if illegal [attack_score]
        This makes the state vector go from (29,) to (65,)
        """
        board = np.reshape(state[:27],(3,3,3)).copy()
        curr = state[27]
        next = state[28]
        move_status_list = []
        for action in range(1,10):
            board = np.reshape(state[:27],(3,3,3)).copy()
            if not self.g.is_legal(action, board=board): # Then it should be 0, low value, high value, -1
                move_status_list.extend([0, -500, 500, -1])
                #print("Section 0")
            else:
                move_status_list.append(1)
                if self.g.check_for_win(action, curr, board=board) == curr: # Then win is 1, lose is 0 and attack score is a very high number
                    move_status_list.append(1)
                    move_status_list.append(0)
                    move_status_list.append(50000000)
                    #print("Section 1")
                else:
                    move_status_list.append(0)
                    a = self.g.play(action, curr, board=board)
                    # If Game is draw lose is 0 and attack score is 0
                    if a == -1:
                        move_status_list.append(0)
                        move_status_list.append(0)
                        #print("Section 2")
                    elif a == 0:
                        flag = False
                        for next_action in range(1,10):
                            if self.g.is_legal(next_action, board=board) and self.g.check_for_win(next_action, next, board=board):
                                flag = True
                                move_status_list.append(1) # Then lose is 1 and attack score  is negative
                                move_status_list.append(-1)
                                #print("Section 3")
                                break
                        if not flag:
                            move_status_list.append(0)
                            #print("Section 4")
                            # Finally check attack score and append
                            board = np.reshape(state[:27],(3,3,3)).copy()
                            #print("State: ", state)
                            move_status_list.append(self.g.get_attack_score(action, curr, board=board))

        
        return np.append(state, move_status_list)
                    
    def additional_q_target_process(self, state, q_target):
        return q_target

