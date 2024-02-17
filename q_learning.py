import os
from cell import CELL
import numpy as np
import itertools
import random
import matplotlib.pyplot as plt
from datetime import datetime

class QLearningGame:
    """
    Slightly adapted interface for the agent within the Q-learning algorithm to play the game
    """

    def __init__(self, width, height, map_file, easy_target: bool = False):
        """
        Game to play the pathfinding game. The game is played by changing the agents velocity vector. 
        The game is won when the agent reaches the target cell.
        The game is lost when the agent hits a wall. The game is restarted when
        the agent hits a wall or reaches the target cell.

        Parameters
        ----------
        width : int
            Width of the window.
        height : int
            Height of the window.
        map_file : str
            File to load the map.
        easy_target : bool
            whether target can be found by hopping over it or if the agent has to end a move on the target cell.
        """
        self.game_over = False
        self.map_file = map_file
        self.window_dim = np.array([width, height])
        self.easy_target = easy_target

        self.grid = np.loadtxt(self.map_file, dtype=int).T
        self.grid = np.vectorize(CELL)(self.grid)
        self.cell_size = self.window_dim / self.grid.shape

        self.max_velocity = 2

        self.agent = self.find_start_position()

        #velocity vector that the agent can manipulate to decide where to go [horizontal, vertical]
        self.velocity = [0,0]

    def find_start_position(self):
        """
        Find a random start position on a start cell.

        Returns
        -------
        tuple(int, int)
            The start position.
        """

        self.velocity = [0,0]

        start_pos = np.where(self.grid == CELL.START)
        r_idx = np.random.randint(len(start_pos[0]))
        return (start_pos[0][r_idx], start_pos[1][r_idx])  

    def move(self, velocity_changes):
        """
        Move the agent in the direction of the updated velocity vector.

        Parameters
        ----------
        velocity_changes : numpy array
            changes for both velocity components (can each be 0, +1, -1)
        """
        if len(velocity_changes)!=len(self.velocity):
            raise ValueError(
                f"Length of velocity change vector must be {len(self.velocity)} but was {len(velocity_changes)}"
            )
        
        self.update_velocity(velocity_changes)
        new_pos = self.update_position()

        if self.is_valid_move(new_pos):
            if self.is_won(new_pos):
                self.game_over = True
                self.agent = self.find_start_position()
            else:
                self.agent = new_pos
        else:
            self.agent = self.find_start_position()

    def is_valid_move(self, new_pos):
        """
        Check if the move is valid.

        Parameters
        ----------
        new_pos : tuple(int, int)
            The new position of the agent.

        Returns
        -------
        bool
            True if the move is valid, False otherwise.
        """
        return (0 <= new_pos[0] < self.grid.shape[0] and
                0 <= new_pos[1] < self.grid.shape[1] and
                self.grid[new_pos] != CELL.WALL and
                not self.jumped_over_special_field(new_pos, CELL.WALL))
    
    def jumped_over_special_field(self, new_pos, field_type):
        """
        Checks whether or not agent jumped over a special field of the given type in the current move
        
        Parameters
        ----------
        new_pos : tuple(int, int)
            The new position of the agent.
        field_type : Enum
            what type of field should be checked.

        Returns
        -------
        bool
            True if the agent jumped over a wall with its move
        """

        #if abs(self.velocity[0]) == 2:
        #    if abs(self.velocity[1]) == 2 or self.velocity[1] == 0:
        #        middle_pos = (int(np.mean([self.agent[0], new_pos[0]])), int(np.mean([self.agent[1], new_pos[1]])))
        #        return self.grid[middle_pos] == CELL.WALL
        #    else:
        #        middle_pos_1 = (int(np.mean([self.agent[0], new_pos[0]])), int(np.ceil(np.mean([self.agent[1], new_pos[1]]))))
        #        middle_pos_2 = (int(np.mean([self.agent[0], new_pos[0]])), int(np.floor(np.mean([self.agent[1], new_pos[1]]))))
        #        return self.grid[middle_pos_1] == CELL.WALL and self.grid[middle_pos_2] == CELL.WALL
        #elif abs(self.velocity[1]) == 2:
        #    if self.velocity[0] == 0:
        #        middle_pos = (int(np.mean([self.agent[0], new_pos[0]])), int(np.mean([self.agent[1], new_pos[1]])))
        #        return self.grid[middle_pos] == CELL.WALL
        #    else:
        #        middle_pos_1 = (int(np.ceil(np.mean([self.agent[0], new_pos[0]]))), int(np.mean([self.agent[1], new_pos[1]])))
        #        middle_pos_2 = (int(np.floor(np.mean([self.agent[0], new_pos[0]]))), int(np.mean([self.agent[1], new_pos[1]])))
        #        return self.grid[middle_pos_1] == CELL.WALL and self.grid[middle_pos_2] == CELL.WALL
        #else:
        #    return False

        x1, y1 = self.agent
        x2, y2 = new_pos
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
            if self.grid[x, y] == field_type:
                return True
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return False

    def update_velocity(self, changes):
        """
        Update the current velocity vector and check if still valid
        
        Parameters
        ----------
        changes : list
            changes for both velocity components (can each be 0, +1, -1)
        """

        if self.is_valid_action(changes):
            self.velocity[0] += changes[0]
            self.velocity[1] += changes[1]
        else:
            raise ValueError("Given velocity changes are not valid")
        
    def update_position(self):
        """
        Updates the position of the agent based on the velocity vector

        Returns
        -------
        tuple(int, int)
            new position of the agent
        """
        return tuple(self.agent[i] + self.velocity[i] for i in range(len(self.agent)))

    def is_valid_action(self, changes):
        """
        Check if velocity changes are valid based on game description
        
        Parameters
        ----------
        changes : list
            changes for both velocity components (can each be 0, +1, -1)

        Returns
        -------
        bool
            whether or not velocity changes are valid
        """

        validity = (-self.max_velocity <= self.velocity[0] + changes[0] <= self.max_velocity) and (-self.max_velocity <= self.velocity[1] + changes[1] <= self.max_velocity)

        validity = validity and ((self.velocity[0] + changes[0] != 0) or (self.velocity[0] + changes[0] != 0) or self.grid[self.agent] == CELL.START)

        return validity
    
    def is_won(self, new_pos):
        """
        Check if the game is won.
        
        Parameters
        ----------
        new_pos : tuple(int, int)
            The new position of the agent.

        Returns
        -------
        bool
            True if the game is won, False otherwise.
        """
        if self.easy_target:
            return (self.grid[new_pos] == CELL.TARGET or
                    self.jumped_over_special_field(new_pos, CELL.TARGET))
        
        else:
            return self.grid[new_pos] == CELL.TARGET

    def is_game_finished(self):
        """
        Auxiliary function to check from outside if game is done
        
        Returns
        -------
        bool
            whether or not game is over
        """
        return self.game_over
    
    def reset_finished(self):
        """
        Set variable game_over to false
        """
        self.game_over = False

class QLearning:
    
    def __init__(self, n_episodes: int,
                 eps: float = 0.1, 
                 eps_decay_factor: float = 1.0, 
                 min_eps: float = 0.1,
                 gamma: float = 0.99, 
                 lr:float = 0.1, 
                 map_file = os.path.join(os.getcwd(), "maps/map1.txt"),
                 step_size: int = 10,
                 easy_target: bool = False,
                 q_table = None,
                 width: int = 500,
                 height: int = 600):
        
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
        step_size : int
            number of steps in which frequency eps should be decayed
        easy_target : bool
            whether target can be found by hopping over it or if the agent has to end a move on the target cell.
        q_table : ndarray
            pre-trained policy table, default None
        width : int
            Width of the window.
        height : int
            Height of the window.       
        """

        self.game = QLearningGame(width=width, height=height, map_file=map_file, easy_target=easy_target)
        self.n_episodes = n_episodes
        self.eps = max(eps, min_eps)
        self.eps_decay_factor = eps_decay_factor
        self.min_eps = min_eps
        self.gamma = gamma
        self.learning_rate = lr
        self.step_size = step_size

        if q_table is None:
            self.q_table = np.zeros((self.game.grid.shape[0], self.game.grid.shape[1], 25, 9), dtype=float)
        else: 
            self.q_table = q_table

        # set a entries in q table to zero for target states
        #target_pos = np.where(self.game.grid == CELL.TARGET)
        #self.q_table[target_pos[0][0], target_pos[1][0], :, :] = [[0]*9]*25

        self.actions = list(itertools.product([0, 1, -1], [0, 1, -1]))
        self.velocities = [[i,j] for i in range(-2,3) for j in range(-2,3)]

        self.rewards_per_episode = []

    def train(self):
        """
        Implementation of the Q-Learning algorithm to play the path finding game on a definied map
        """

        for episode in range(1, self.n_episodes + 1):

            episode_reward, _ = self.run_episode(training=True)
            self.rewards_per_episode.append(episode_reward)

            if (episode%self.step_size == 0):
                #reduce eps for randomization
                self.eps = max(self.min_eps, self.eps*self.eps_decay_factor)

            if (episode%10 == 0):
                print(f"Episode {episode}: Reward : {episode_reward}")


    def run_episode(self, training = False):
        """
        Runs one complete episode of the game until target is found and then resets the finished variable

        Parameters
        ----------
        training : bool
            whether or not the episode is run during the training process, default = False i.e., no random actions are generated

        Returns
        -------
        int
            reward of this episode
        list(tuple(int, int))
            list of the visited positions in the episode
        """
        episode_reward = 0
        positions = [self.game.agent]

        while not self.game.is_game_finished():

            current_pos = self.game.agent
            current_velocity = self.game.velocity
            velocity_index = self.velocities.index(list(current_velocity))

            if np.random.uniform(0,1) < self.eps:
                action_index = random.randint(0,8)
            else:
                action_index = np.nanargmax(self.q_table[current_pos[0], current_pos[1], velocity_index, :])

            while not self.game.is_valid_action(self.actions[action_index]):
                #if chosen action is not valid set its q-value to a NaN such that cannot be chosen and choose other action

                self.q_table[current_pos[0], current_pos[1], velocity_index, action_index] = np.nan

                if training and np.random.uniform(0,1) < self.eps:
                    action_index = random.randint(0,8)
                else:
                    action_index = np.nanargmax(self.q_table[current_pos[0], current_pos[1], velocity_index, :])

            self.game.move(self.actions[action_index])
            
            new_pos = self.game.agent
            new_velocity = self.game.velocity
            new_velocity_index = self.velocities.index(list(new_velocity))

            if not training:
                print(f"Pos: {current_pos} Action taken: {self.actions[action_index]} landed on {new_pos}")

            # update reward for this episode and q_table entry for previous position 
            episode_reward -= 1
            reward = -1
            if self.game.is_game_finished():
                reward += 2
            self.q_table[current_pos[0], current_pos[1], velocity_index, action_index] = (1 - self.learning_rate) * self.q_table[current_pos[0], current_pos[1], velocity_index, action_index] + self.learning_rate*(reward + self.gamma * max(self.q_table[new_pos[0], new_pos[1], new_velocity_index, :]))

            positions.append(new_pos)

        self.game.reset_finished()

        if not training:
            print(f"Game finished with reward {episode_reward}")

        return episode_reward, positions
    

    def store_qtable(self, file_name = None):
        """
        Store computed Q-Table for reuseability
        """
        #reshape 4d Q-Table to 2d
        q_table_reshaped = self.q_table.reshape(self.q_table.shape[0] * self.q_table.shape[1], self.q_table.shape[2] * self.q_table.shape[3])
        if file_name is None:
            np.savetxt("qtable_run"+datetime.now().strftime("%d%m%Y%H%M%S")+".txt", q_table_reshaped)
        else:
            np.savetxt(file_name, q_table_reshaped)



    def load_q_table_from_text(self, txt_file):
        """
        Loads pretrained Q-Table from a txt file for reusability
        
        Parameters
        ----------
        txt_file : str
            Location of text file to use
        """

        loaded_qtable = np.loadtxt(txt_file)

        reshaped_qtable = loaded_qtable.reshape(self.q_table.shape[0], self.q_table.shape[1], self.q_table.shape[2], self.q_table.shape[3])

        self.q_table = reshaped_qtable



if __name__ == "__main__":

    n_episodes = 100
    step_size = 1
    q_learning = QLearning(n_episodes, eps=1.0, eps_decay_factor=0.9, min_eps=0.1, step_size=step_size)

    q_learning.train()
    
    rewards = np.array(q_learning.rewards_per_episode)
    
    plt.plot(range(0,n_episodes, step_size), np.abs(np.average(rewards.reshape(-1, step_size), axis=1)))
    plt.yscale("log")
    plt.title("Loss per episode")
    plt.xlabel("Number of episodes")
    plt.ylabel("Negative Reward")
    plt.show()