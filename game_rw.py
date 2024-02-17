import os

import numpy as np

from drawing import Container
from cell import CELL
from q_learning import QLearning

import matplotlib.pyplot as plt

class GameVisualization(Container):

    def __init__(self, width, height, map_file, easy_target, q_table):
        """
        Game to play the pathfinding game. This class is used for visualization of the game itself
        and the paths chosen by an agent using Q-Learning

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
        q_table : ndarray
            trained policy for the agent.
        """
        self.window_dim = np.array([width, height])
        self.map_file = map_file
        self.easy_target = easy_target
        self.q_table = q_table


        super().__init__(self.window_dim[0], self.window_dim[1], frame_rate=20)

    def setup(self):  
        """
        Setup the game and the event bindings.
        """      
        self.grid = np.loadtxt(self.map_file, dtype=int).T
        self.grid = np.vectorize(CELL)(self.grid)
        self.cell_size = self.window_dim / self.grid.shape

        self.agent = self.find_start_position()

        self.finder = QLearning(200, q_table = self.q_table, map_file=self.map_file, easy_target=self.easy_target)
        self.agent = self.find_start_position()

        self.bind("<Up>", self.move)
        self.bind("<Down>", self.move)
        self.bind("<Left>", self.move)
        self.bind("<Right>", self.move) 
        self.bind("<Button-1>", self.draw_episode)
        
    def find_start_position(self):
        """
        Find a random start position on a start cell.

        Returns
        -------
        tuple(int, int)
            The start position.
        """
        start_pos = np.where(self.grid == CELL.START)
        r_idx = np.random.randint(len(start_pos[0]))
        return (start_pos[0][r_idx], start_pos[1][r_idx])      

    def draw(self):
        """
        Draw the cells in the grid and the agent on the canvas.
        """
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                self.draw_cell(i, j)

        #self.draw_cell(self.agent[0], self.agent[1], "red")
        self.draw_grid(self.grid.shape[0], self.grid.shape[1], self.cell_size, color="black", linewidth=1.5)
        
    def draw_cell(self, i, j, color=None):
        """
        Draw a cell on the canvas.

        Parameters
        ----------
        i : int
            The x-coordinate of the cell.
        j : int
            The y-coordinate of the cell.
        color : str
            The color of the cell. If None, the color is determined by the cell type.
        """
        if color is None:
            color = self.grid[i][j].get_color()
        self.canvas.create_rectangle(i * self.cell_size[0], j * self.cell_size[1],
                                     (i + 1) * self.cell_size[0], (j + 1) * self.cell_size[1],
                                     fill=color, outline="")

    def move(self, event):
        """
        Move the agent in the direction of the arrow key.

        Parameters
        ----------
        event : tkinter.Event
            The event object.
        """
        if event.keysym == "Up":
            new_pos = (self.agent[0], self.agent[1] - 1)
        elif event.keysym == "Down":
            new_pos = (self.agent[0], self.agent[1] + 1)
        elif event.keysym == "Left":
            new_pos = (self.agent[0] - 1, self.agent[1])
        elif event.keysym == "Right":
            new_pos = (self.agent[0] + 1, self.agent[1])

        if self.is_valid_move(new_pos):
            if self.is_won(new_pos):
                print("You won!")
                self.agent = self.find_start_position()
            else:
                self.agent = new_pos
        else:
            print("You lost!")
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
    
    def draw_episode(self, event):
        coords = (event.x, event.y)
        coords = np.array([event.x, event.y])
        i, j = np.clip(coords // self.cell_size, 0, np.array(self.grid.shape) - 1).astype(int)

        self.finder.game.agent = (i,j)
        _, episode = self.finder.run_episode()

        for state in episode[:-1]:
            self.draw_cell(state[0], state[1], "red")
            self.root.update()
            self.root.after(750)
            self.draw_cell(state[0], state[1], "orange")
            self.root.update()     
    

if __name__ == "__main__":

    n_episodes = 1000000
    step_size = 500
    map_file = os.path.join(os.getcwd(), "maps/map1.txt")
    easy_target = False
    q_learning = QLearning(n_episodes,eps = 1.0, eps_decay_factor=0.75, min_eps=0.1, step_size=step_size, map_file=map_file, easy_target=easy_target)

    q_learning.train()
    
    rewards = np.array(q_learning.rewards_per_episode)
    
    plt.plot(range(0,n_episodes, step_size), np.abs(np.average(rewards.reshape(-1, step_size), axis=1)))
    plt.yscale("log")
    plt.title("Loss per episode")
    plt.xlabel("Number of episodes")
    plt.ylabel("Negative Reward")
    plt.show()

    q_learning.store_qtable("q_tables/map1.txt")

    GameVisualization(width= 500, height=600, map_file=map_file, easy_target=easy_target, q_table=q_learning.q_table)

  


