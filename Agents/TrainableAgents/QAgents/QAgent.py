from Agents.TrainableAgents.TrainableAgent import TrainableAgent
from GameSystem.Environments import TicTacToeEnvironment
from GameSystem.Games import TicTacToeGame
from GameSystem.Actions import TicTacToeAction
from GameSystem.States import TicTacToeState
import numpy as np

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
            return self.model.predict([np.append(q_vector[0].get_data(), q_vector[1].get_data())])[0]
        except:
            print(f"Could not predict (Likely because model was not fitted or because model does not exist) returned 0")
            return 0


