from tqdm import tqdm
import matplotlib.pyplot as plt

class StandardGym(object):
    """description of class"""
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

    def simulate(self, agent, env, episodes=10000, show_every=1000, save_every=5_000, training=True, display_plot=True, **kwargs):
        """
        Prerequistes:
            Assumes **kwargs will have all required arguments to call environment.reset
            Assumes agent has play function which either takes in avoid_illegal and real_epsilon optional parameters or has **kwargs

        Simulate episodes -
               Across training we also aggregate statistics across the rolling window of show_every and display with a graph after training
        

        If Training is True:
               Assumes the agent has learn function implemented
               Assumes the agent has save_model function implemented that can be called on with **kwargs

        """
        aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': [] }
        rewardlist = []


        for n in tqdm(range(episodes)):
            #print(f"Episode {n}")
            if n %show_every == 0:
                flag = True
            else:
                flag = False

            if n % save_every == 0:
                if training:
                    try:
                        agent.save(**kwargs)
                    except:
                        print("Could not save")
                else:
                    pass

            state = env.reset(**kwargs)
            done = False
            total_rewards = 0
            turn = 0
            while not done:
                turn += 1
                action = agent.play(state, real_epsilon=self.epsilon_scheduler(n, training), avoid_illegal=self.avoid_illegal)
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

        if display_plot:
            input("Press any key to show plot....")
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

