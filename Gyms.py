import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from collections import deque

class Gym(object):
    """
    """
    def __init__(self, epsilon=0.5, avoid_illegal=True):
        self.epsilon = epsilon
        self.avoid_illegal = avoid_illegal

    def epsilon_scheduler(self, episodes, training = True):
        """
        Return a lower epsilon as episdoes go higher
        """
        if not training:
            return 0.0
        return self.epsilon / ((episodes+1)**0.125)

    def simulate(self, agent, env, opponent_1, opponent_2, episodes=10000, show_every = 1000, training = True):
        """
        Simulate episodes -
               Across training we also aggregate statistics across the rolling window of show_every and display with a graph after training
               In this system assuming there are no draws and the rewards for loss and win is equal we get Win % = ((Avg + 5)*10)%
        If Training is True:
            Loops through each episode and creates a dataset of [q_vector, qlambda value experienced]
            After every episode we deploy the dataset as (X_train, y_train) to agents learn function

        If Training is False:
            
        """
        aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': [] }
        rewardlist = []


        for n in tqdm(range(episodes)):
            #print(f"Episode {n}")
            if n %show_every == 0:
                flag = True
            else:
                flag = False

            state = env.reset(opponent_1, opponent_2)
            done = False
            total_rewards = 0
            turn = 0
            while not done:
                turn += 1
                action = agent.play(state, self.epsilon_scheduler(n, training), avoid_illegal=self.avoid_illegal)
                new_state, rewards, done = env.step(action)
                total_rewards += rewards
                if training:
                    self.update_dataset(agent=agent, state=state, action=action, reward=rewards, turn=turn, new_state=new_state, done=done)
                state = new_state
            if training:
                self.deploy_dataset(agent=agent)
                self.clear_dataset()
            rewardlist.append(total_rewards)
            if flag:
                aggr_ep_rewards['ep'].append(n)
                if len(rewardlist) > 0:
                    aggr_ep_rewards['avg'].append(sum(rewardlist)/len(rewardlist))
                    aggr_ep_rewards['min'].append(min(rewardlist))
                    aggr_ep_rewards['max'].append(max(rewardlist))
                rewardlist.clear()

        print("Win Ratio is (AVG+5)/10")
        input()
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label = 'avg')
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label = 'min')
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label = 'max')
        plt.legend()
        plt.show()
        
        return

    def update_dataset(self, **kwargs):
        pass
    def clear_dataset(self, **kwargs):
        pass
    def deploy_dataset(self, **kwargs):
        pass


class ForwardTDLambdaGym(Gym):
    """
    A Gym to Train an Agent in the TDLambda Method
    Assumes that the agent has the following functions implemented for training
    Agent:
        agent.learn(X_train, y_train)
        agent.play(state, real_epsilon)
        agent.decay_rate
        agent.create_q_vector(state, action)
        
    For testing requires 
        agent.play(state, real_epsilon)


    """
    def __init__(self, epsilon=0.5, lamb=27, avoid_illegal=True):
        """
        Default is TD(inf) because max turns is 27
        """
        Gym.__init__(self, epsilon, avoid_illegal)
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

        agent.learn(np.asarray(X_train), np.asarray(y_train))
        return

    def clear_dataset(self, **kwargs):
        """
        Reset the dataset
        """
        self.dataset = None

class BatchDQLearningGym(Gym):
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
        Gym.__init__(self, epsilon, avoid_illegal)
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
        agent.learn(self.dataset)
        return

    def clear_dataset(self):
        """
        We do not wan't to reset the database and so we just don't
        """
        if self.clear_after_episode:
            self.dataset = deque(maxlen=self.max_replay_size)



