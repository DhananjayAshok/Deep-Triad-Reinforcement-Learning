#Core Imports Here
import numpy as np
from Agents.TrainableAgents.QAgents.QAgent import QAgent

##############################################################################################
class QSKLearnAgent(QAgent):
    """
    Is a class that works for any SKLearn model that used partial_fit
    """
    def __init__(self, learning_rate, decay_rate, model_path= "models/SKLearnAgents", model_name="QSKlearnAgent"):
        QAgent.__init__(self, learning_rate, decay_rate, model_path, model_name)
        self.model = self.get_unfitted_model()

    def get_unfitted_model(self):
        raise NotImplementedError

    def learn(self, **kwargs):
        """
        Performs batch updates with that dataset
        """
        X_train = kwargs.get('X_train', None)
        y_train = kwargs.get('y_train', None)
        if X_train is None or y_train is None:
            raise ValueError("Passed in No Value to learn function")
        X_train = np.array([np.append(X[0].get_data(), X[1].get_data()) for X in X_train])
        self.model.partial_fit(X_train, y_train)

    def save(self, **kwargs):
        """
        Save model to the path - path + model_name + .sav
        """
        import os
        import pickle
        save_name = self.model_name
        save_path = self.model_path
        if kwargs.get('model_name', None) is not None:
            save_name = kwargs.get('model_name')
        if kwargs.get('model_path', None) is not None:
            save_path = kwargs.get('model_path')
        final_path = os.path.join(save_path, save_name + ".sav")
        pickle.dump(self.model, open(final_path, 'wb'))

    def load(self):
        """
        Loads weights from a given path
        """
        import os
        import pickle
        self.model = pickle.load(open(os.path.join(self.model_path, self.model_name+".sav"), 'rb'))
        print(f"Model {self.model_name} imported without issues")

class QLinearAgent(QSKLearnAgent):
    """
    Stochastic Gradient Descent with Linear Approximator (SKLearn SGD)
    """
    def __init__(self, learning_rate, decay_rate, model_path="models/SKLearnAgents", model_name="QLinearAgent"):
        QSKLearnAgent.__init__(self, learning_rate, decay_rate, model_path, model_name)

    def get_unfitted_model(self):
        from sklearn.linear_model import SGDRegressor
        return SGDRegressor()

# Implement Your Custom Classes Below
##############################################################################################
