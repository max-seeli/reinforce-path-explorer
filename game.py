from argparse import ArgumentParser
import os

import numpy as np

from drawing import Container
from cell import CELL
from monte_carlo import MonteCarlo
from map import MapLoader

class Game(Container):

    def __init__(self, width, height, map_file):
        """
        Game to play the pathfinding game. The game is played with the arrow keys
        to move the agent. The game is won when the agent reaches the target cell.
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
        """
        self.window_dim = np.array([width, height])
        self.map_file = map_file

        super().__init__(self.window_dim[0], self.window_dim[1], frame_rate=20)

    def setup(self):  
        """
        Setup the game and the event bindings.
        """      
        self.grid = MapLoader.load_map(self.map_file)
        self.cell_size = self.window_dim / self.grid.shape

        self.finder = MonteCarlo(self.grid, policy_filename=f"./policies/{os.path.basename(self.map_file).split('.')[0]}.policy.txt")
        self.finder.monte_carlo_control()
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

        self.draw_cell(self.agent[0], self.agent[1], "red")
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

    def draw_episode(self, event):
        coords = np.array([event.x, event.y])
        i, j = np.clip(coords // self.cell_size, 0, np.array(self.grid.shape) - 1).astype(int)
        start = (i, j)
        episode = self.finder.generate_episode(start)

        for state, action, reward in episode:
            self.draw_cell(state[0], state[1], "orange")
            self.root.update()
            self.root.after(1000)
            self.draw_cell(state[0], state[1])
            self.root.update()

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
                self.grid[new_pos] != CELL.WALL)
    
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
        return self.grid[new_pos] == CELL.TARGET


if __name__ == "__main__":
    parser = ArgumentParser(description="Game")
    parser.add_argument("--width", type=int, default=500, help="Width of the window")
    parser.add_argument("--height", type=int, default=600, help="Height of the window")
    parser.add_argument("--map_file", type=str, default=os.path.join(os.getcwd(), "maps/map1.txt"), help="File to load the map")
    args = parser.parse_args()

    Game(args.width, args.height, args.map_file)
