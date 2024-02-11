import numpy as np
from tqdm import tqdm

# Environment parameters
grid_size = (10, 10)
starts = [(0,0), (0,1), (1, 1), (1, 3)]  # Starting position and velocity (x, y, vx, vy)
goal = (9, 9)
obstacles = [(2, 2), (0, 2), (1, 2), (2, 1)]  # Add obstacles as tuples if any
max_velocity = 2  # Maximum velocity in any direction

# Actions are now changes to velocity (dvx, dvy)
actions = {'increase_vx': (1, 0), 'decrease_vx': (-1, 0),
           'increase_vy': (0, 1), 'decrease_vy': (0, -1),
           'increase_vx_vy': (1, 1), 'decrease_vx_vy': (-1, -1),
           'increase_vx_decrease_vy': (1, -1), 'decrease_vx_increase_vy': (-1, 1),
           'no_change': (0, 0)}

def get_legal_actions(state):
    candidates = actions.values()
    legal_actions = []
    for c in candidates:
        combined = (state[2] + c[0], state[3] + c[1])
        if np.all(np.abs(combined) < 3) and np.any(combined):
            legal_actions.append(c)
    
    legal_actions = [list(actions.keys())[list(actions.values()).index(a)] for a in legal_actions]
    return legal_actions

class Policy:
    def __init__(self, epsilon=0.1):
        self.policy = {}
        self.epsilon = epsilon
        self.is_greedy = True

    def __getitem__(self, state):
        if state not in self.policy:
            self.policy[state] = np.random.choice(get_legal_actions(state))
        if np.random.random() < self.epsilon and self.is_greedy:
            return np.random.choice(get_legal_actions(state))
        return self.policy[state]
    
    def __setitem__(self, state, action):
        self.policy[state] = action

    def __len__(self):
        return len(self.policy)

# Initialize state-action values and returns
Q = {} 
returns = {}  # Dictionary to keep track of returns for each state-action pair
policy = Policy(epsilon=0.1)

# Initialize Q and policy for all states (positions and velocities)
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        for vx in range(-max_velocity, max_velocity + 1):
            for vy in range(-max_velocity, max_velocity + 1):
                state = (i, j, vx, vy)
                Q[state] = {a: 0.0 for a in actions}
                returns[state] = {a: [] for a in actions}

                legal_actions = get_legal_actions(state)
                random_index = np.random.choice(len(legal_actions))
                policy[state] = legal_actions[random_index]


def valid_position(state):
    in_bounds = state[0] >= 0 and state[0] < grid_size[0] and state[1] >= 0 and state[1] < grid_size[1]
    not_obstacle = state[:2] not in obstacles
    return in_bounds and not_obstacle

def get_random_valid_state():
    position = (np.random.randint(grid_size[0]), np.random.randint(grid_size[1]))
    while not valid_position(position):
        position = (np.random.randint(grid_size[0]), np.random.randint(grid_size[1]))
    
    velocity = (np.random.randint(-max_velocity, max_velocity + 1), np.random.randint(-max_velocity, max_velocity + 1))
    return position + velocity

def generate_episode(policy, start=None):
    episode = []
    if start is None:
        random_start_index = np.random.choice(len(starts))
        random_start = starts[random_start_index]
        start = random_start + (0, 0)
    state = start
    # max_episode_length = 100
    while state[:2] != goal: # and len(episode) < max_episode_length:
        action = policy[state]
        dvx, dvy = actions[action]
        vx, vy = state[2] + dvx, state[3] + dvy
        # Ensure velocity stays within bounds
        vx = max(-max_velocity, min(max_velocity, vx))
        vy = max(-max_velocity, min(max_velocity, vy))
        next_position = (state[0] + vx, state[1] + vy)
        if next_position[0] < 0 or next_position[0] >= grid_size[0] or next_position[1] < 0 or next_position[1] >= grid_size[1] or next_position in obstacles:
            reward = -1  # Penalty for hitting boundaries or obstacles
            next_state = start  # Remain in the same position
        elif next_position == goal:
            reward = 1
            next_state = next_position + (vx, vy)
        else:
            reward = -1  # Small penalty for each move
            next_state = next_position + (vx, vy)
        episode.append((state, action, reward))
        state = next_state
    return episode

def monte_carlo_control(episodes=100, gamma=0.9):
    for _ in tqdm(range(episodes)):
        episode = generate_episode(policy)
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if not (state, action) in [(x[0], x[1]) for x in episode[:t]]:
                returns[state][action].append(G)
                Q[state][action] = np.mean(returns[state][action])
                # Update policy with the action that has the highest value and is legal
                legal_actions = get_legal_actions(state)
                
                best_action = max(legal_actions, key=lambda a: Q[state][a])
                policy[state] = best_action

                if policy[state] not in get_legal_actions(state):
                    print(policy[state], get_legal_actions(state))
                    print("Action valid:", policy[state] in get_legal_actions(state))

        print("Reward:", sum([x[2] for x in episode]))


monte_carlo_control()

# Display the resulting policy for a subset of states for clarity
for state in policy.policy.keys():
    if state[2:] == (0, 0):  # Only show policy for zero velocity states for brevity
        print(f"Policy at {state}: {policy[state]}")

# Display the number of states in the policy
print(len(policy))



# Verify the policy by running a few episodes
policy.is_greedy = False
episode = generate_episode(policy, start=(0, 0, 0, 0))
print("Episode:")
for s, a, r in episode:
    print(f"State: {s}, Action: {a}, Reward: {r}")
