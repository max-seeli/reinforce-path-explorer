import numpy as np
from tqdm import tqdm
from collections import defaultdict
import warnings

from cell import CELL


class Policy:
    """
    A wrapper class for the policy dictonary of the `MonteCarlo` agent.
    """

    def __init__(self, train_epsilon, test_epsilon):
        """
        Initializes an instance of the MonteCarlo Reinforcement Learning policy for an agent.

        Parameters
        ----------
        train_epsilon : float
            The exploration rate during training (epsilon-greedy). Higher values encourage more exploration.
        test_epsilon : float
            The exploration rate during testing (epsilon-greedy). Higher values encourage more exploration.
        """

        self.policy = {}
    
        self.train_epsilon = train_epsilon
        self.test_epsilon = test_epsilon
        self.epsilon = train_epsilon

    def __getitem__(self, state):
        """
        Returns the actions for the specified `state`.

        Parameters
        ----------
        state: tuple(int, int, int, int)
            The state for which actions should be returned.
        """

        if state not in self.policy:
            self.policy[state] = np.random.choice(MonteCarlo.get_valid_actions(state))
        if np.random.random() < self.epsilon:
            return np.random.choice(MonteCarlo.get_valid_actions(state))
        return self.policy[state]
    
    def __setitem__(self, state, action):
        """
        Sets the `action` for the specified `state`.

        Parameters
        ----------
        state: tuple(int, int, int, int)
            The state for which `action` should be set.
        action: str
            The action which should be associated to the specified `state`.
        """

        self.policy[state] = action

    def __len__(self):
        """
        Returns the length of the policy.
        """

        return len(self.policy)
    
    def set_trained(self):
        """
        Sets the epsilon to the `test_epsilon`.
        """

        self.epsilon = self.test_epsilon

    def load(self, filename):
        """
        Loads policy based on the specified `filename`.

        Parameters
        ----------
        filename: str
            The filename of the policy to load.
        """

        with open(filename, 'r') as f:
            for line in f:
                state = tuple(map(int, line.split(',')[:4]))
                action = line.split(',')[4].strip()
                self.policy[state] = action

    def save(self, filename):
        """
        Saves policy based on the specified `filename`.

        Parameters
        ----------
        filename: str
            The filename of the policy to save.
        """

        with open(filename, 'w') as f:
            for state, action in self.policy.items():
                f.write(f"{state[0]},{state[1]},{state[2]},{state[3]},{action}\n")
    
class MonteCarlo:
    """
    A reinforcement learning agent based on the monte carlo method, able to learn a simple game.
    """

    valid_actions_memoization = {}
    actions = {'increase_vx': (1, 0), 'decrease_vx': (-1, 0),
           'increase_vy': (0, 1), 'decrease_vy': (0, -1),
           'increase_vx_vy': (1, 1), 'decrease_vx_vy': (-1, -1),
           'increase_vx_decrease_vy': (1, -1), 'decrease_vx_increase_vy': (-1, 1),
           'no_change': (0, 0)}
    max_velocity = 2
    
    def __init__(self, grid, gamma=0.9, num_episodes=10000, train_epsilon=0.9, test_epsilon=0.05, policy_filename=None, load_policy=False):
        """
        Initializes an instance of the MonteCarlo Reinforcement Learning agent.

        Parameters
        ----------
        grid : Grid
            The environment grid for the agent to navigate.
        gamma : float
            The discount factor for future rewards. Defaults to 0.9.
        num_episodes : int
            The number of episodes the agent is trained. Defaults to 10000
        train_epsilon : float
            The exploration rate during training (epsilon-greedy). Higher values encourage more exploration. Defaults to 0.9.
        test_epsilon : float
            The exploration rate during testing (epsilon-greedy). Higher values encourage more exploration. Defaults to 0.05.
        policy_filename : str
            The filename to save or load the learned policy. If specified, the policy will be saved to or loaded from this file. Defaults to None.
        load_policy : bool
            Whether to load a pre-trained policy from the specified policy file. If True, the agent will attempt to load the policy from the file specified by `policy_filename`. Defaults to False.
        """
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
        """
        Returns all valid actions (velocities <= max_velocity) the agent can perform for the specified `state`. 
        Only calculates actions on the first call for a specific state, otherwise uses memoization of previous calculated values.

        Parameters
        ----------
        state : tuple(int, int, int, int)
            The state for which all valid actions should be calculated.
        """

        if state not in MonteCarlo.valid_actions_memoization:
            valid_actions = []
            for action, step in MonteCarlo.actions.items():
                combined = (state[2] + step[0], state[3] + step[1])
                if np.all(np.abs(combined) <= MonteCarlo.max_velocity) and np.any(combined):
                    valid_actions.append(action)


            MonteCarlo.valid_actions_memoization[state] = valid_actions
        return MonteCarlo.valid_actions_memoization[state]
    
    def generate_episode(self, start=None):
        """
        Runs an episode of the game. Episode ends with reaching the target cell and gets resetted to a new 
        random start if we reach an invalid state.

        Parameters
        ----------
        start : tuple(int, int)
            A start position (x, y) for the first try of the episode, if not specified a valid random start point is choosen. Defaults to `None`.
        """

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
        """
        Returns a valid random start point (x, y).
        """

        random_start_index = np.random.choice(len(self.starts))
        random_start = self.starts[random_start_index]
        return random_start

    def step(self, state, action):
        """
        Returns the new state based on current state and action.

        Parameters
        ----------
        state : tuple(int, int, int, int)
            The state for which the action should be performed.
        action: str
            The specified action, which should be performed.
        """

        dvx, dvy = MonteCarlo.actions[action]
        vx, vy = state[2] + dvx, state[3] + dvy
        # Ensure velocity stays within bounds
        vx = max(-self.max_velocity, min(self.max_velocity, vx))
        vy = max(-self.max_velocity, min(self.max_velocity, vy))

        return (state[0] + vx, state[1] + vy) + (vx, vy)
    
    def is_step_valid(self, old_state, new_state):
        """
        Returns wether a new state is valid based on the old one. 

        Parameters
        ----------
        old_state : tuple(int, int, int, int)
            The old state of the game.
        new_state : tuple(int, int, int, int)
            The new state of the game.
        """

        old_pos = old_state[:2]
        new_pos = new_state[:2]
        return (0 <= new_pos[0] < self.grid.shape[0] and
                0 <= new_pos[1] < self.grid.shape[1] and
                not self.hits_obstacle(old_pos, new_pos))

    def hits_obstacle(self, pos1, pos2):
        """
        Returns if the line between pos1 and pos2 hits a wall.

        Parameters
        ----------
        pos1 : tuple(int, int)
            The (x, y) coordinates of the starting position of the line segment.
        pos2 : tuple(int, int)
            The (x, y) coordinates of the ending position of the line segment.
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
        """
        Runs the monte carlo method based on first pass visit.
        """

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