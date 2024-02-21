import numpy as np
from tqdm import tqdm
from collections import defaultdict
import warnings
from operator import itemgetter

from cell import CELL
import itertools
import matplotlib.pyplot as plt


class Policy:
    def __init__(self, train_epsilon, test_epsilon,
                 min_eps = 0.05, eps_decay = 1.0):
        self.policy = {}
    
        self.train_epsilon = train_epsilon
        self.test_epsilon = test_epsilon
        self.epsilon = train_epsilon
        self.min_eps = min_eps
        self.eps_decay = eps_decay

    def __getitem__(self, state):
        if state not in self.policy:
            self.policy[state] = np.random.choice(QLearning.get_legal_actions(state))
        if np.random.random() < self.epsilon:
            return np.random.choice(QLearning.get_legal_actions(state))
        return self.policy[state]
    
    def __setitem__(self, state, action):
        self.policy[state] = action

    def __len__(self):
        return len(self.policy)
    
    def set_trained(self):
        self.epsilon = self.test_epsilon

    def reduce_eps(self):
        self.epsilon = max(self.min_eps, self.epsilon * self.eps_decay)

    def load(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                state = tuple(map(int, line.split(',')[:4]))
                action = line.split(',')[4].strip()
                self.policy[state] = action

    def save(self, filename):
        with open(filename, 'w') as f:
            for state, action in self.policy.items():
                f.write(f"{state[0]},{state[1]},{state[2]},{state[3]},{action}\n")

class QLearning:

    actions = {'increase_vx': (1, 0), 'decrease_vx': (-1, 0),
           'increase_vy': (0, 1), 'decrease_vy': (0, -1),
           'increase_vx_vy': (1, 1), 'decrease_vx_vy': (-1, -1),
           'increase_vx_decrease_vy': (1, -1), 'decrease_vx_increase_vy': (-1, 1),
           'no_change': (0, 0)}
    max_velocity = 2

    def __init__(self, grid,
                 n_episodes: int,
                 gamma: float = 0.99, 
                 lr:float = 0.1, 
                 step_size: int = 10,
                 train_epsilon=0.9, 
                 test_epsilon=0.05,
                 eps_decay_factor: float = 1.0, 
                 min_eps: float = 0.1, 
                 policy_filename=None, 
                 load_policy=False):
        
        """
        Q_learning algorithm implementation to learn an optimal path to the target on a given map.
        
        Parameters
        ----------
        grid
            map that should be used for the game
        n_episodes : int
            number of episodes to use for training
        gamma : float
            discounted factor
        lr : float
            learning rate
        map_file : str
            file name where map is stored that should be used for training
        step_size : int
            number of steps in which frequency eps should be decayed
        train_epsilon : float
            starting epsilon in case of training
        test_epsilon : float
            epsilon in case of testing
        eps_decay_factor : float
            factor by which the epsilon should decay, default = 1.0 (no decay)
        min_eps: float
            minimal possible values for eps
        policy_filename : ndarray
            file where pre-trained policy is stored, default None
        load_policy : bool
            whether or not pre-trained policy should be loaded    
        """

        self.grid = grid
        self.n_episodes = n_episodes
        self.eps_decay_factor = eps_decay_factor
        self.min_eps = min_eps
        self.gamma = gamma
        self.learning_rate = lr
        self.step_size = step_size

        self.starts = np.where(self.grid == CELL.START)
        self.starts = list(zip(self.starts[0], self.starts[1]))
        self.goal = np.where(self.grid == CELL.TARGET)
        self.goal = (self.goal[0][0], self.goal[1][0])

        self.policy = Policy(train_epsilon, test_epsilon)
        self.policy_filename = policy_filename
        self.is_trained = False

        if load_policy:
            self.policy.load(policy_filename)
            self.is_trained = True
            self.policy.set_trained()

        self.q_table = defaultdict(lambda: defaultdict(float))

        self.velocities = [[i,j] for i in range(-self.max_velocity, self.max_velocity + 1) for j in range(-self.max_velocity, self.max_velocity + 1)]

        self.rewards_per_episode = []

    @staticmethod
    def get_legal_actions(state):
        candidates = QLearning.actions.values()
        legal_actions = []
        for c in candidates:
            combined = (state[2] + c[0], state[3] + c[1])
            if np.all(np.abs(combined) <= QLearning.max_velocity) and np.any(combined):
                legal_actions.append(c)

        legal_actions = [list(QLearning.actions.keys())[list(QLearning.actions.values()).index(a)] for a in legal_actions]
        return legal_actions
    
    def train(self):
        """
        Implementation of the Q-Learning algorithm to play the path finding game on a definied map
        """
        if self.is_trained:
            warnings.warn("The agent is already trained.")
            return
        
        for episode in tqdm(range(self.n_episodes)):

            episode_info = self.generate_episode(use_policy=False)
            episode_reward = sum(map(lambda x: x[2], episode_info))
            self.rewards_per_episode.append(episode_reward)

            if (episode%self.step_size == 0):
                #reduce eps for randomization
                self.policy.reduce_eps()

        self.is_trained = True
        self.policy.set_trained()
        if self.policy_filename:
            self.policy.save(self.policy_filename)
    
    def generate_episode(self, start=None, use_policy = True):
    
        episode = []
        state = start + (0, 0) if start else self.gen_random_start() + (0, 0)

        while state[:2] != self.goal:

            if use_policy:
                action = self.policy[state]
            else: 
                #still training the policy
                legal_actions = QLearning.get_legal_actions(state)

                if np.random.random() < self.policy.epsilon:
                    action = np.random.choice(legal_actions) 
                else:
                    action = max(legal_actions, key=lambda a: self.q_table[state][a])

            next_state = self.step(state, action)
            
            if not self.is_step_legal(state, next_state):
                reward = -1  # Penalty for hitting boundaries or obstacles
                next_state = self.gen_random_start() + (0, 0)
            elif next_state[:2] == self.goal:
                reward = 1
            else:
                reward = -1  # Small penalty for each move

            #update q_table and policy
            self.q_table[state][action] = (1 - self.learning_rate) * self.q_table[state][action] + self.learning_rate*(reward + self.gamma * max(list(itemgetter(*QLearning.get_legal_actions(state))(self.q_table[state]))) )
            best_action = max(legal_actions, key=lambda a: self.q_table[state][a])
            self.policy[state] = best_action

            episode.append((state, action, reward))
            state = next_state
        return episode

    
    def gen_random_start(self):
        random_start_index = np.random.choice(len(self.starts))
        random_start = self.starts[random_start_index]
        return random_start
    
    def step(self, state, action):
        dvx, dvy = QLearning.actions[action]
        vx, vy = state[2] + dvx, state[3] + dvy
        # Ensure velocity stays within bounds
        vx = max(-self.max_velocity, min(self.max_velocity, vx))
        vy = max(-self.max_velocity, min(self.max_velocity, vy))

        return (state[0] + vx, state[1] + vy) + (vx, vy)
    
    def is_step_legal(self, old_state, new_state):
        old_pos = old_state[:2]
        new_pos = new_state[:2]
        return (0 <= new_pos[0] < self.grid.shape[0] and
                0 <= new_pos[1] < self.grid.shape[1] and
                not self.hits_wall(old_pos, new_pos))

    def hits_wall(self, pos1, pos2):
        """
        Find if the line between pos1 and pos2 hits a wall.
        """
        x1, y1 = pos1
        x2, y2 = pos2
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x = x1
        y = y1
        n = 1 + dx + dy
        x_inc = 1 if x2 > x1 else -1
        y_inc = 1 if y2 > y1 else -1
        error = dx - dy
        dx *= 2
        dy *= 2
        for _ in range(n):
            if self.grid[x, y] == CELL.WALL:
                return True
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return False
    
if __name__ == "__main__":

    grid = np.loadtxt("./maps/map1.txt", dtype=int).T
    grid = np.vectorize(CELL)(grid)
    n_episodes = 10000
    step_size = 200

    ql = QLearning(grid, n_episodes=n_episodes, step_size=step_size, policy_filename="policies/ql_map1.txt")
    ql.train()

    # Display the resulting policy for a subset of states for clarity
    for state in ql.policy.policy.keys():
        if state[2:] == (0, 0):  # Only show policy for zero velocity states for brevity
            print(f"Policy at {state}: {ql.policy[state]}")

    # Display the number of states in the policy
    print(f"There are {len(ql.policy)} states in policy")

    # Verify the policy by running an episodes
    episode = ql.generate_episode(start=(4, 16))
    print("Episode:")
    for s, a, r in episode:
        print(f"State: {s}, Action: {a}, Reward: {r}")

    rewards = np.array(ql.rewards_per_episode)
    
    plt.plot(range(0,n_episodes, step_size), np.abs(np.average(rewards.reshape(-1, step_size), axis=1)))
    plt.yscale("log")
    plt.title("Loss per episode")
    plt.xlabel("Number of episodes")
    plt.ylabel("Negative Reward")
    plt.show()