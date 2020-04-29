import numpy as np
from Opponents import HumanOpponent, RandomOpponent, HyperionOpponent , MMOpponent
from GameSystem.game import Game
import os
import pickle
from NeuralNetworks import NaiveNetwork, AssistedNetwork
import random



# Non Trainable/ Present Agents
#region
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


class MMAgent(Agent):
    def __init__(self):
        self.MM = MMOpponent()

    def play(self, state, real_epsilon=0, **kwargs):
        
        return self.MM.play(state)
#endregion



class TrainableAgent(Agent):
    """
    Subclass of agents that are meant to go through the training process
    """
    def __init__(self, learning_rate=1, decay_rate=0.2, model_name="TrainableAgent"):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.model_name = model_name
        self.g = Game()

    def update_from_state_action(self, state, action, target, prediction):
        raise NotImplementedError

    def estimate_from_state_action(self, state, action):
        raise NotImplementedError

    def learn(self, dataset):
        raise NotImplementedError

    def save_model(self, path, model_name):
        pass

class QAgent(TrainableAgent):
    """
    We start with a naive approach of just x(S,A) = state vector + [action]

    """
    def __init__(self, learning_rate, decay_rate, model_name="TrainableAgent"):
        TrainableAgent.__init__(self, learning_rate, decay_rate, model_name)


    def play(self, state, real_epsilon=0.0, avoid_illegal = True, verbose=False):
        """
        Epsilon Greedy
        Manually Avoids Illegal Moves if avoid_illegal is True
        """
        choices = []
        for i in range(1,10):
            if self.g.is_legal(i, state[:27].reshape((3,3,3))) or not avoid_illegal:
                choices.append(i)

        if np.random.random() > real_epsilon:
            scores = [self.estimate_from_state_action(state, i) for i in choices]
            max_indexes = np.where(np.asarray(scores) == max(scores))[0]
            if verbose:
                if len(scores) == 9:
                    print(f"Player Q_values are \n{np.reshape(scores, (3,3))}\n")
                else:
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
            return self.model.predict([q_vector])[0]
        except:
            print(f"Could not predict (Likely because model was not fitted) returned 0")
            return 0

class DictionaryAgent(QAgent):
    """
    Supposed to be trained with a BatchLearning Gym with a clear_every_episode set to True
    Has a smart learning function where whenever it sees an illegal move it tries to learn that all similar configurations are also illegal
    """
    def __init__(self, learning_rate, decay_rate, model_name="dict"):
        QAgent.__init__(self, learning_rate, decay_rate, model_name)
        self.d = {}
        self.g = Game()

    def estimate_from_q_vector(self, q_vector):
        return self.d.get(tuple(q_vector), 0)

    def learn(self, queue, heavylearn=True):
        """
        Updates value such that dict[tuple(q_vector)] = reward + q_value(next_state)
        """
        for i, (state, action, reward, next_state, done) in enumerate(queue):
            vectuple = tuple(self.create_q_vector(state, action))
            if done:
                self.d[vectuple] = reward
            else:
                next_actions = range(1, 10)
                values = []
                for act in next_actions:
                    tempvec = tuple(self.create_q_vector(next_state, act))
                    values.append(self.estimate_from_q_vector(tempvec))
                if reward <= -10:
                    pass
                else:
                    self.d[vectuple] = reward + self.decay_rate* max(values)
                if heavylearn:
                    self.heavy_learn(state)
                    self.heavy_learn(next_state)

        return

    def heavy_learn(self, state):
        """
        Manually Learn all illegal moves and winning moves from a given state
        """
        self.g.matrix = state[:27].reshape((3,3,3)).copy()
        player = state[27]
        for act in range(1,10):
            if not self.g.is_legal(act):
                #tempvec = tuple(self.create_q_vector(state, act))
                #self.d[tempvec] = -10
                pass
            else:
                winner = self.g.check_for_win(act, player)
                if winner > 0:
                    tempvec = tuple(self.create_q_vector(state, act))
                    self.d[tempvec] = 5
        return

    def save_model(self, alternate_name=None):
        if alternate_name is None:
            name = self.model_name
        else:
            name = alternate_name
        #print(os.listdir())
        with open(os.path.join("models", "Dictionary", name), 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.d, f, pickle.HIGHEST_PROTOCOL)

    def load_model(self, alternate_name = None):
        if alternate_name is None:
            name = self.model_name
        else:
            name = alternate_name
        final = os.path.join("models", "Dictionary", name)
        with open(final, 'rb') as f:
            self.d = pickle.load(f)
        print(f"Loaded without error - Number of estimates is {len(self.d)}")
        return

    def stats(self):
        """
        Returns lists nonzeros, illegals, winners, intermediates
        """
        l = len(self.d)
        print(f"Number of Entries: {l}")
        non_zeroes = 0
        illegals = 0
        winners = 0
        intermediates = 0
        nz = []
        il = []
        win = []
        inter = []
        for key in self.d:
            val = self.d[key]
            if val != 0:
                non_zeroes += 1
                nz.append(key)
                if val <= -10:
                    illegals += 1
                    il.append(key)
                elif val == 5:
                    winners += 1
                    win.append(key)
                elif val > 0:
                    intermediates += 1
                    inter.append(key)

        print(f"Number of non zero entries: {non_zeroes} this is {100*non_zeroes/l}% of the dictionary")
        print(f"Number of illegal entries: {illegals} this is {100*illegals/non_zeroes}% of the non zero entries")
        print(f"Number of winning entries: {winners} this is {100*winners/non_zeroes}% of the non zero entries")
        print(f"Number of intermediate entries: {intermediates} this is {100*intermediates/non_zeroes}% of the non zero entries")
        return nz, il, win, inter


# SKLearn Agents
#region
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
#endregion

# Deep Q Agents
#region

class DeepQAgent(QAgent):
    """
    Naive Simple Neural Net
    """
    def __init__(self, learning_rate, decay_rate, min_replay_to_fit=1_000, minibatch_size=1_000, avoid_assist=False, win=False, block=False, model_name="DQA"):
        QAgent.__init__(self, learning_rate, decay_rate, model_name)
        self.minibatch_size = minibatch_size
        self.min_replay_to_fit=min_replay_to_fit


    def estimate_from_state_action(self, state, action):
        pred = self.model.predict([state])[0]
        return pred[action-1]

    def learn(self, queue):
        """
        Performs a Single Neural Net fit on a random sample of minibatch
        """
        if len(queue) < self.min_replay_to_fit:
            print(f"DID NOT FIT because queue length {len(queue)}")
            return

        minibatch = random.sample(queue, self.minibatch_size)
        self.model.train(minibatch, self.decay_rate)
        return

    def save_model(self, path, model_name=None):
        """
        Saves model to the path given
        """
        import os
        save_name = self.model_name
        if model_name is not None:
            save_name = model_name
        final_path = os.path.join(path, save_name)
        self.model.save(final_path)
        return

    def load_model(self, path, model_name=None):
        """
        Loads model if its in the path provided
        """
        import os
        save_name = self.model_name
        if model_name is not None:
            save_name = model_name
        final_path = os.path.join(path, save_name)
        self.model.load(final_path)
        return

    def get_model(self, **kwargs):
        raise NotImplementedError

    def create_q_vector(self, state, action):
        """
        Retursn x(S,A) as described above
        """
        raise NotImplementedError("Not supposed to have q_vector")

    def estimate_from_q_vector(self, q_vector):
        raise NotImplementedError("Not supposed to have q_vector")

class NaiveDeepQAgent(DeepQAgent):
    """
    Naive Simple Neural Net
    """
    def __init__(self, learning_rate, decay_rate, min_replay_to_fit=1_000, minibatch_size=1_000, avoid_assist=False, win=False, block=False, model_name="DQA"):
        DeepQAgent.__init__(self, learning_rate, decay_rate, model_name=model_name, min_replay_to_fit=min_replay_to_fit, minibatch_size=minibatch_size)
        self.model = NaiveNetwork(avoid_assist=avoid_assist, win_assist=win, block_assist = block)

class AssistedDeepQAgent(DeepQAgent):
    """
    Heavily customized feature inputs to try to make the best performing network
    """
    def __init__(self, learning_rate, decay_rate, min_replay_to_fit=1_000, minibatch_size=1_000, model_name="ADQA"):
        DeepQAgent.__init__(self, learning_rate, decay_rate, model_name=model_name, min_replay_to_fit=min_replay_to_fit, minibatch_size=minibatch_size)
        self.model = AssistedNetwork()

#endregion

class NEATAgent(TrainableAgent):
    import neat
    def __init__(self, genome, config, model_name="NEATAgent: Genome - "):
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        TrainableAgent.__init__(self, model_name=model_name+str(genome))
    
    def play(self, state):
        """
        Play a move from the genome network
        """
        output = self.net.activate(tuple(state))
        return np.argmax(output)+1

    def update_fitness(self, value):
        """
        Increments genome fitness by given value
        """
        self.genome.fitness += value

