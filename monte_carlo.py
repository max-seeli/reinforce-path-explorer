import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import warnings

from cell import CELL


class Policy:
    def __init__(self, train_epsilon, test_epsilon):
        self.policy = {}
    
        self.train_epsilon = train_epsilon
        self.test_epsilon = test_epsilon
        self.epsilon = train_epsilon

    def __getitem__(self, state):
        if state not in self.policy:
            self.policy[state] = np.random.choice(MonteCarlo.get_valid_actions(state))
        if np.random.random() < self.epsilon:
            return np.random.choice(MonteCarlo.get_valid_actions(state))
        return self.policy[state]
    
    def __setitem__(self, state, action):
        self.policy[state] = action

    def __len__(self):
        return len(self.policy)
    
    def set_trained(self):
        self.epsilon = self.test_epsilon

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
    
class MonteCarlo:
    actions_memoization = {}
    actions = {'increase_vx': (1, 0), 'decrease_vx': (-1, 0),
           'increase_vy': (0, 1), 'decrease_vy': (0, -1),
           'increase_vx_vy': (1, 1), 'decrease_vx_vy': (-1, -1),
           'increase_vx_decrease_vy': (1, -1), 'decrease_vx_increase_vy': (-1, 1),
           'no_change': (0, 0)}
    max_velocity = 2
    
    def __init__(self, grid, gamma=0.9, num_episodes=10000, train_epsilon=0.9, test_epsilon=0.05, policy_filename=None, load_policy=False):
        self.grid = grid

        self.starts = np.where(self.grid == CELL.START)
        self.starts = list(zip(self.starts[0], self.starts[1]))
        self.goal = np.where(self.grid == CELL.TARGET)
        self.goal = (self.goal[0][0], self.goal[1][0])
        self.obstacles = np.where(self.grid == CELL.WALL)
        self.obstacles = list(zip(self.obstacles[0], self.obstacles[1]))

        self.gamma = gamma
        self.num_episodes = num_episodes
        self.train_epsilon = train_epsilon
        self.test_epsilon = test_epsilon
        
        self.Q = defaultdict(lambda: defaultdict(float))
        self.returns = defaultdict(lambda: defaultdict(list))

        self.policy = Policy(train_epsilon, test_epsilon)
        self.policy_filename = policy_filename
        self.is_trained = False

        if load_policy:
            self.policy.load(policy_filename)
            self.is_trained = True
            self.policy.set_trained()

    @staticmethod
    def get_valid_actions(state):
        valid_actions = []
        
        if state not in MonteCarlo.actions_memoization:
            for action, step in MonteCarlo.actions.items():
                combined = (state[2] + step[0], state[3] + step[1])
                if np.all(np.abs(combined) <= MonteCarlo.max_velocity) and np.any(combined):
                    valid_actions.append(action)


            MonteCarlo.actions_memoization[state] = valid_actions
        return MonteCarlo.actions_memoization[state]
    
    def generate_episode(self, start=None):
        episode = []
        state = start + (0, 0) if start else self.gen_random_start() + (0, 0)
        while state[:2] != self.goal:
            action = self.policy[state]
            next_state = self.step(state, action)

            if not self.is_step_valid(state, next_state):
                reward = -1  # Penalty for hitting boundaries or obstacles
                next_state = self.gen_random_start() + (0, 0)
            elif next_state[:2] == self.goal:
                reward = 1
            else:
                reward = -1  # Small penalty for each move
            
            episode.append((state, action, reward))
            state = next_state
        return episode
    
    def gen_random_start(self):
        random_start_index = np.random.choice(len(self.starts))
        random_start = self.starts[random_start_index]
        return random_start

    def step(self, state, action):
        dvx, dvy = MonteCarlo.actions[action]
        vx, vy = state[2] + dvx, state[3] + dvy
        # Ensure velocity stays within bounds
        vx = max(-self.max_velocity, min(self.max_velocity, vx))
        vy = max(-self.max_velocity, min(self.max_velocity, vy))

        return (state[0] + vx, state[1] + vy) + (vx, vy)
    
    def is_step_valid(self, old_state, new_state):
        old_pos = old_state[:2]
        new_pos = new_state[:2]
        return (0 <= new_pos[0] < self.grid.shape[0] and
                0 <= new_pos[1] < self.grid.shape[1] and
                not self.hits_obstacle(old_pos, new_pos))

    def hits_obstacle(self, pos1, pos2):
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

    def monte_carlo_control(self):
        if self.is_trained:
            warnings.warn("The agent is already trained.")
            return
        
        for _ in tqdm(range(self.num_episodes)):
            episode = self.generate_episode()
            G = 0
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                if not any((state, action) == (e[0], e[1]) for e in episode[:t]):
                    self.returns[state][action].append(G)
                    self.Q[state][action] = np.mean(self.returns[state][action])

                    # Update policy with the action that has the highest value and is legal
                    self.policy[state] = max(MonteCarlo.get_valid_actions(state), key=lambda a: self.Q[state][a])

        self.is_trained = True
        self.policy.set_trained()
        if self.policy_filename:
            self.policy.save(self.policy_filename)

if __name__ == "__main__":
    from map import MapLoader

    map = MapLoader.load_map("./maps/map1.txt")
    mc = MonteCarlo(map, num_episodes=10000, policy_filename="policies/test_map1.txt.policy")
    mc.monte_carlo_control()

    # Display the resulting policy for a subset of states for clarity
    for state in mc.policy.policy.keys():
        if state[2:] == (0, 0):  # Only show policy for zero velocity states for brevity
            print(f"Policy at {state}: {mc.policy[state]}")

    # # Display the number of states in the policy
    # print(f"There are {len(mc.policy)} states in policy")

    # # Verify the policy by running an episodes
    # episode = mc.generate_episode(start=(4, 16))
    # print("Episode:")
    # for s, a, r in episode:
    #     print(f"State: {s}, Action: {a}, Reward: {r}")