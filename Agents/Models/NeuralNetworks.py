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

class SimpleNetwork(DeepQNetwork):
    """
    Takes as input state action pairs and outputs the q value expected
    """
    def __init__(self, update_every=1000):
        DeepQNetwork.__init__(self, update_every)

    def transform_input(self, state, **kwargs):
        """
        Returns state action
        """
        return np.append(state.get_data(), kwargs['action'].get_data())

    def create_model(self):
        """
        Has an architecture with inputs -> 32 -> 64 -> 32 -> 1
        """
        model = Sequential()
        model.add(Dense(32, activation="relu", kernel_initializer=Zeros(), input_shape=(30,)))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1, activation="linear", kernel_initializer=Zeros()))
        model.compile("adam", loss="mse", metrics=["accuracy"])
        return model


class ConvNetwork(DeepQNetwork):
    def __init__(self, update_every=1000):
        DeepQNetwork.__init__(self, update_every)

    def transform_input(self, state, **kwargs):
        """
        Returns in the shape
        [
        [l0], 
        [l1],
        [l2], 
        [action * 9]
        ]
        with 1 being player, 2 being next and 3 being 3rd after player
        """
        action = kwargs['action']
        minilist = [action.get_data() for i in range(9)]
        board = self.board_review(state)
        return np.append(board, minilist).reshape((4, 3, 3))



    def board_review(self, state):
        """
        Takes in a board and subs in values such that the player is always 1 and next player is always 2
        """
        board, player, next = state.get_induviduals()
        third = next%3+1
        player_token = 10
        next_token = 20
        third_token = 30
        

        for i, item in enumerate(board):
            if item == player:
                board[i] = player_token
            elif item == next:
                board[i] = next_token
            elif item == third:
                board[i] = third_token
            else:
                pass

        for i, item in enumerate(board):
            if item == player_token:
                board[i] = 1
            elif item == next_token:
                board[i] = 2
            elif item == third_token:
                board[i] = 3
            else:
                pass

        return board



    def create_model(self):
        """
        Has an architecture with inputs -> 32 -> 64 -> 32 -> 1
        """
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, activation="relu", input_shape=(4,3,3)))
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dense(1, activation="linear"))
        model.compile("adam", loss="mse", metrics=["accuracy"])
        return model


class AlphaZeroNetwork(DeepQNetwork):
    def __init__(self, update_every=1000):
        DeepQNetwork.__init__(self, update_every)

    def transform_input(self, state, **kwargs):
        """
        Performs a feature map simlilar to AlphaZero convolutions - 
        3 cubes each with indicators if there exists a piece of player i in that spot for all i cubes
        2 cubes indicating whose turn it is and who's next 
        1 cube indicating the action which the user is taking
        """
        board, curr, next = state.get_induviduals()
        last = next%3+1
        playerboard = self.create_player_indicator_cube(board, curr)
        nextboard = self.create_player_indicator_cube(board, next)
        return np.array([self.create_player_indicator_cube(board, curr), self.create_player_indicator_cube(board, next), self.create_player_indicator_cube(board, last), self.create_value_cube(curr), self.create_value_cube(next), self.create_value_cube(kwargs['action'])])

    def create_player_indicator_cube(self, state_slice, player):
        """
        Creates indicator cube as specified in transform state docstring
        """
        state_slice_copy = state_slice.copy()
        for i, item in enumerate(state_slice_copy):
            if item == player:
                state_slice_copy[i] = 1
            else:
                state_slice_copy[i] = 0
        return np.reshape(state_slice_copy, (3,3,3))

    def create_value_cube(self, value):
        """
        Returns a cube populated only with that value
        """
        temp = [value for i in range(27)]
        return np.reshape(temp, (3,3,3))

    def create_model(self):
        inp = Input(shape=(6, 3, 3, 3))
        flat = Flatten()(inp)
        x = Conv3D(2, 1, activation="elu")(inp)
        x = Conv3D(2, 1, activation="elu")(x)
        x = Flatten() (x)
        together = Concatenate() ([x, flat])
        final = Dense(128, activation="elu") (together)
        final = Dense(32, activation="elu") (final)
        outputs = Dense(1, activation= "linear") (final)
        model = Model(inputs=inp, outputs=outputs)
        model.compile("adam", loss="mse", metrics=["accuracy"])
        return model