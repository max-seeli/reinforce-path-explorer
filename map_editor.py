from argparse import ArgumentParser
import os

import numpy as np

from drawing import Container
from cell import CELL


class MapEditor(Container):

    def __init__(self, width, height, rows, cols, map_file):
        """
        User interface to create or edit a map. The map is saved to a file when
        the window is closed. By clicking on a cell, the cell type is cycled
        through the following values:
        - 0: Empty
        - 1: Target
        - 2: Wall
        - 3: Start

        Parameters
        ----------
        width : int
            Width of the window.
        height : int
            Height of the window.
        rows : int
            Number of rows of the grid.
        cols : int
            Number of columns of the grid.
        map_file : str
            File to save the map. If the file exists, the initial state is 
            loaded from it.
        """
        self.window_dim = np.array([width, height])
        self.map_dim = np.array([cols, rows], dtype=int)
        self.map_file = map_file

        super().__init__(self.window_dim[0], self.window_dim[1], frame_rate=20)

    def setup(self):
        """
        Setup the grid and the event binding.
        """
        if os.path.exists(self.map_file):
            self.grid = np.loadtxt(self.map_file, dtype=int).T
        else:
            self.grid = np.zeros(self.map_dim, dtype=int)

        self.grid = np.vectorize(CELL)(self.grid)
        self.cell_size = self.window_dim / self.grid.shape

        self.bind("<Button-1>", self.cycle_cell)

    def draw(self):
        """
        Draw the grid on the canvas. The color of the cells is determined by the
        cell type.
        """
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                self.canvas.create_rectangle(i * self.cell_size[0], j * self.cell_size[1],
                                     (i + 1) * self.cell_size[0], (j + 1) * self.cell_size[1],
                                     fill=self.grid[i][j].get_color(), outline="")
        self.draw_grid(self.grid.shape[0], self.grid.shape[1], self.cell_size, color="black", linewidth=1.5)

    def shutdown(self):
        """
        Save the map to a file when the window is closed.
        """
        np.savetxt(self.map_file, np.vectorize(lambda x: x.value)(self.grid).T, fmt="%d")
    
    def cycle_cell(self, event):
        """
        Cycle the cell type when a cell is clicked.

        Parameters
        ----------
        event : tkinter.Event
            The event object.
        """
        coords = np.array([event.x, event.y])
        i, j = np.clip(coords // self.cell_size, 0, np.array(self.grid.shape) - 1).astype(int)
        self.grid[i][j] = CELL((self.grid[i][j].value + 1) % 4)


if __name__ == "__main__":
    parser = ArgumentParser(description="Map Editor")
    parser.add_argument("--width", type=int, default=500, help="Width of the window")
    parser.add_argument("--height", type=int, default=500, help="Height of the window")
    parser.add_argument("--cols", type=int, default=10, help="Number of columns")
    parser.add_argument("--rows", type=int, default=10, help="Number of rows")
    parser.add_argument("--map_file", type=str, default=os.path.join(os.getcwd(), "maps/map.txt"), help="File to save the map (or load if it exists)")
    args = parser.parse_args()

    MapEditor(args.width, args.height, args.rows, args.cols, args.map_file)
