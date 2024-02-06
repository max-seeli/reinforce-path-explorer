import os
from game import QLearningGame
from cell import CELL
import numpy as np
import itertools
import random
import matplotlib.pyplot as plt

class QLearning:
    
    def __init__(self, n_episodes: int,
                 eps: float = 0.1, 
                 eps_decay_factor: float = 1.0, 
                 min_eps: float = 0.1,
                 gamma: float = 0.99, 
                 lr:float = 0.1, 
                 map_file = os.path.join(os.getcwd(), "maps/map1.txt")):
        
        """
        Q_learning algorithm implementation to learn an optimal path to the target on a given map.
        
        Parameters
        ----------
        n_episodes : int
            number of episodes to use for training
        eps : float
            probability to use a random action
        eps_decay_factor : float
            factor by which the epsilon should decay, default = 1.0 (no decay)
        min_eps: float
            minimal possible values for eps
        gamma : float
            discounted factor
        lr : float
            learning rate
        map_file : str
            file name where map is stored that should be used for training        
        """

        self.game = QLearningGame(width= 500, height=600, map_file=map_file)
        self.n_episodes = n_episodes
        self.eps = max(eps, min_eps)
        self.eps_decay_factor = eps_decay_factor
        self.min_eps = min_eps
        self.gamma = gamma
        self.learning_rate = lr
        self.q_table = np.zeros((self.game.grid.shape[0], self.game.grid.shape[1], 9), dtype=float)

        # set a positive reward for finding the target large enough to compensate costs
        #target_pos = np.where(self.game.grid == CELL.TARGET)
        #self.q_table[target_pos[0][0], target_pos[1][0],:] = [self.game.grid.size]*9

        self.actions = list(itertools.product([0, 1, -1], [0, 1, -1]))

        self.rewards_per_episode = []

    def train(self):
        """
        Implementation of the Q-Learning algorithm to play the path finding game on a definied map
        """

        for episode in range(self.n_episodes):
            if (episode%10 == 0):
                print(f"Episode {episode}")
                
                self.eps = max(self.min_eps, self.eps*self.eps_decay_factor)

            total_reward = 0.0

            while not self.game.is_game_finished():

                current_pos = self.game.agent

                if np.random.uniform(0,1) < self.eps:
                    action_index = random.randint(0,8)
                else:
                    action_index = np.nanargmax(self.q_table[current_pos[0], current_pos[1], :])

                while not self.game.is_valid_action(np.array(self.actions[action_index])):
                    #if chosen action is not valid set its q-value to a NaN such that cannot be chosen and choose other action

                    self.q_table[current_pos[0], current_pos[1], action_index] = np.nan

                    if np.random.uniform(0,1) < self.eps:
                        action_index = random.randint(0,8)
                    else:
                        action_index = np.nanargmax(self.q_table[current_pos[0], current_pos[1], :])

                self.game.move(self.actions[action_index])

                # update total reward for this episode and q_table entry for previous position 
                total_reward -= 1
                self.q_table[current_pos[0], current_pos[1], action_index] = (1 - self.learning_rate) * self.q_table[current_pos[0], current_pos[1], action_index] + self.learning_rate*(-1 + self.gamma * max(self.q_table[self.game.agent[0], self.game.agent[1], :]))

            self.rewards_per_episode.append(total_reward)
            self.game.reset_finished()


if __name__ == "__main__":

    n_episodes = 100000
    step_size = 100
    q_learning = QLearning(n_episodes, eps=1.0, eps_decay_factor=0.9)

    q_learning.train()
    
    rewards = np.array(q_learning.rewards_per_episode)
    
    plt.plot(range(0,n_episodes, step_size), np.average(rewards.reshape(-1, step_size), axis=1))
    plt.title("Rewards per episode")
    plt.xlabel("Number of episodes")
    plt.ylabel("Reward")
    plt.show()