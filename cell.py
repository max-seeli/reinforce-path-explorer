from enum import Enum

class CELL(Enum):
    """
    Enumeration of the cell types.
    """
    EMPTY = 0
    TARGET = 1
    WALL = 2
    START = 3

    def get_color(self):
        """
        Return the color of the cell type.

        Returns
        -------
        str
            The color of the cell type.
        """
        if self == CELL.EMPTY:
            return "white"
        elif self == CELL.TARGET:
            return "green"
        elif self == CELL.WALL:
            return "black"
        elif self == CELL.START:
            return "blue" 
