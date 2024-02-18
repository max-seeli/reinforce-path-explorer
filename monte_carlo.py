import numpy as np
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
            self.policy[state] = np.random.choice(MonteCarlo.get_legal_actions(state))
        if np.random.random() < self.epsilon:
            return np.random.choice(MonteCarlo.get_legal_actions(state))
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
            self.policy.set_trained()
            self.is_trained = True

    @staticmethod
    def get_legal_actions(state):
        candidates = MonteCarlo.actions.values()
        legal_actions = []
        for c in candidates:
            combined = (state[2] + c[0], state[3] + c[1])
            if np.all(np.abs(combined) <= MonteCarlo.max_velocity) and np.any(combined):
                legal_actions.append(c)

        legal_actions = [list(MonteCarlo.actions.keys())[list(MonteCarlo.actions.values()).index(a)] for a in legal_actions]
        return legal_actions
    
    def generate_episode(self, start=None):
        
        episode = []
        if start is None:
            random_start_index = np.random.choice(len(self.starts))
            random_start = self.starts[random_start_index]
            start = random_start
        state = start + (0, 0)
        while state[:2] != self.goal:
            action = self.policy[state]
            dvx, dvy = MonteCarlo.actions[action]
            vx, vy = state[2] + dvx, state[3] + dvy
            # Ensure velocity stays within bounds
            vx = max(-self.max_velocity, min(self.max_velocity, vx))
            vy = max(-self.max_velocity, min(self.max_velocity, vy))

            next_position = (state[0] + vx, state[1] + vy)
            next_state = next_position + (vx, vy)

            if next_position[0] < 0 or next_position[0] >= self.grid.shape[0] or next_position[1] < 0 or next_position[1] >= self.grid.shape[1] or self.hits_wall(state[:2], next_position):
                reward = -1  # Penalty for hitting boundaries or obstacles
                next_state = start + (0, 0)
            elif next_position == self.goal:
                reward = 1
            else:
                reward = -1  # Small penalty for each move
            
            episode.append((state, action, reward))
            state = next_state
        return episode
    
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

    def monte_carlo_control(self):
        if self.is_trained:
            warnings.warn("The agent is already trained.")
            return
        
        for _ in tqdm(range(self.num_episodes)):
            episode = self.generate_episode()
            G = 0
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                if not (state, action) in [(x[0], x[1]) for x in episode[:t]]:
                    self.returns[state][action].append(G)
                    self.Q[state][action] = np.mean(self.returns[state][action])
                    # Update policy with the action that has the highest value and is legal
                    legal_actions = MonteCarlo.get_legal_actions(state)

                    best_action = max(legal_actions, key=lambda a: self.Q[state][a])
                    self.policy[state] = best_action

        self.is_trained = True
        self.policy.set_trained()
        if self.policy_filename:
            self.policy.save(self.policy_filename)

if __name__ == "__main__":
    from map import MapLoader

    map = MapLoader.load_map("./maps/easy.txt")
    mc = MonteCarlo(map)
    mc.monte_carlo_control()

    # Display the resulting policy for a subset of states for clarity
    for state in mc.policy.policy.keys():
        if state[2:] == (0, 0):  # Only show policy for zero velocity states for brevity
            print(f"Policy at {state}: {mc.policy[state]}")

    # Display the number of states in the policy
    print(f"There are {len(mc.policy)} states in policy")

    # Verify the policy by running an episodes
    episode = mc.generate_episode(start=(4, 16))
    print("Episode:")
    for s, a, r in episode:
        print(f"State: {s}, Action: {a}, Reward: {r}")