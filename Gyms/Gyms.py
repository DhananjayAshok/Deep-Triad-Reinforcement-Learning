# Core Imports Here
from .StandardGym import StandardGym
import numpy as np
from collections import deque



class ForwardTDLambdaGym(StandardGym):
    """
    A Gym to Train an Agent in the TDLambda Method
    Assumes that the agent has the following functions implemented for training
    Agent:
        agent.learn(X_train, y_train)
        agent.decay_rate
        agent.create_q_vector(state, action)
        
    For testing requires 
        agent.play(state, real_epsilon)


    """
    def __init__(self, epsilon=0.5, lamb=27, avoid_illegal=True):
        """
        Default is TD(inf) because max turns is 27
        """
        StandardGym.__init__(self, epsilon, avoid_illegal)
        self.lamb = lamb

    def update_dataset(self, **kwargs):
        """
        Create an update a dataset in the form  -
        dataset : [[q_vector, actual_reward], [q_vector, actual_reward]....]
        """
        turn = kwargs['turn']
        agent = kwargs['agent']
        decay = agent.decay_rate
        state = kwargs['state']
        action = kwargs['action']
        reward = kwargs['reward']
        if turn == 1:
            self.dataset = []
        q_vector = agent.create_q_vector(state, action)
        for n in range(len(self.dataset)):
            turn_added = n+1
            turn_diff = turn - turn_added
            if (turn_diff > self.lamb + 1):
                self.dataset[n][1] = self.dataset[n][1] + (decay**turn_diff)*reward
            else:
                pass
        self.dataset.append([q_vector, reward])
        return
    
    def deploy_dataset(self, **kwargs):
        """
        Makes the dataset split into two numpy ndarrays - X_train which has all the input vectors and y_train which has all the experienced values.
        Feeds this to the agents learn function
        """
        agent = kwargs['agent']
        X_train = []
        y_train = []
        for X, y in self.dataset:
            X_train.append(X)
            y_train.append(y)

        agent.learn(X_train=np.asarray(X_train), y_train=np.asarray(y_train))
        return

    def clear_dataset(self, **kwargs):
        """
        Reset the dataset
        """
        self.dataset = None

class BatchDQLearningGym(StandardGym):
    """
    A Gym to Train an Agent in the TDLambda Method
    Assumes that the agent has the following functions implemented for training
    Agent:
        agent.learn(Queue)
        agent.play(state, real_epsilon)
        
    For testing requires 
        agent.play(state, real_epsilon)

    """
    def __init__(self, epsilon=0.5, max_replay_size = 50_000, avoid_illegal=True, clear_after_episode=False):
        StandardGym.__init__(self, epsilon, avoid_illegal)
        self.max_replay_size = max_replay_size
        self.clear_after_episode = clear_after_episode
        self.dataset = deque(maxlen=self.max_replay_size)

    def update_dataset(self, **kwargs):
        """
        Create an update a dataset in the form  -
        dataset : queue(of some maximum size) [state, action, reward, new_state, done] 
        """
        turn = kwargs['turn']
        state = kwargs['state']
        action = kwargs['action']
        reward = kwargs['reward']
        new_state = kwargs['new_state']
        done = kwargs['done']

        
        transition = [state, action, reward, new_state, done]
        self.dataset.append(transition)
        return
    
    def deploy_dataset(self, **kwargs):
        """
        Feeds the queue to the agent
        """
        agent = kwargs.get('agent', 0)
        agent.learn(queue=self.dataset)
        return

    def clear_dataset(self):
        """
        We do not wan't to reset the database and so we just don't
        """
        if self.clear_after_episode:
            self.dataset = deque(maxlen=self.max_replay_size)


# Implement Your Custom Classes Below
##############################################################################################