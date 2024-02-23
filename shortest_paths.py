import numpy as np
import networkx as nx
from map import MapLoader
from cell import CELL
from monte_carlo import MonteCarlo
import os
import json


class ShortestPaths:

    def __init__(self, map_file, json_file="maps/opt_path.json", verbose=False):
        """
        Class to compute shortest paths from all staring points to a target field on a given map.

        Parameters
        ----------
        map_file : str
            File to load the map.
        json_file : str
            File where results can be stored later.
        verbose : bool
            if there should be intermediate output.
        """
        self.map_file = os.path.join(os.getcwd(), map_file)
        self.grid = MapLoader.load_map(map_file=map_file)
        self.mc = MonteCarlo(self.grid)

        self.json_file = json_file
        self.verbose = verbose

        self.velocities = [(i, j) for i in range(-2, 3) for j in range(-2, 3)]

        self.start_nodes = []
        self.target = None

        self.results = {}

    def store_all_paths(self, map_files):
        """
        Computes and stores all shortest paths for the given maps as a JSON file.

        Parameters
        ----------
        map_files : list(str)
            list of map files
        """

        for map_file in map_files:
            self.map_file = os.path.join(os.getcwd(), map_file)
            self.grid = MapLoader.load_map(map_file=map_file)
            self.mc = MonteCarlo(self.grid)

            self.calc_shortest_paths()

        with open(self.json_file, "w") as outfile:
            json.dump(self.results, outfile, indent=4)

    def calc_shortest_paths(self):
        """
        Computes all shortest paths and stores information about them in a JSON file.
        """
        G = self.build_graph()

        map_name = self.map_file.split("/")[-1]

        results = dict(
            file="maps/" + map_name,
            optimal_path_lengths=[],
            start_positions=[],
            shortest_paths=[],
        )

        for start in self.start_nodes:

            path = nx.shortest_path(G, start, self.target)

            if self.verbose:
                print([(G.nodes[i]["x_coord"], G.nodes[i]["y_coord"]) for i in path])

            results["start_positions"].append(
                [G.nodes[start]["x_coord"], G.nodes[start]["y_coord"]]
            )
            results["optimal_path_lengths"].append(len(path) - 1)
            results["shortest_paths"].append(
                [(G.nodes[i]["x_coord"], G.nodes[i]["y_coord"]) for i in path]
            )

        self.results[map_name[:-4]] = results

    def build_graph(self):
        """
        Build a NetworkX graph representing the map and valid moves on it.

        Returns
        -------
        networkx.Graph representing the game
        """
        G = nx.Graph()

        self.start_nodes = []
        self.target = None

        # add nodes to graph
        for (x, y), value in np.ndenumerate(self.grid):

            if value == CELL.WALL:
                continue

            if value == CELL.TARGET:
                node_index = self.enumerate_node(x, y, 0)
                self.target = node_index
                G.add_node(node_index, x_coord=x, y_coord=y, vx=0, vy=0, z=0)
            else:
                for z in range(len(self.velocities)):
                    node_index = self.enumerate_node(x, y, z)
                    G.add_node(
                        node_index,
                        x_coord=x,
                        y_coord=y,
                        vx=self.velocities[z][0],
                        vy=self.velocities[z][1],
                        z=z,
                    )

                    if value == CELL.START and self.velocities[z] == (0, 0):
                        self.start_nodes.append(node_index)

        # add edges
        for node_index in G.nodes:
            node = G.nodes[node_index]
            neighbors = self.calc_neighbor_indices(
                (node["x_coord"], node["y_coord"]), node["z"]
            )

            for neighbor in neighbors:
                if self.grid[neighbor[:2]] == CELL.TARGET:
                    z = 0
                else:
                    z = self.velocities.index(neighbor[-2:])

                G.add_edge(node_index, self.enumerate_node(neighbor[0], neighbor[1], z))

        return G

    def calc_neighbor_indices(self, coords, velocity_index):
        """
        Computes valid neighbor indices for a given node in the graph.

        Parameters
        ----------
        coords : tuple(int, int)
            current position on the map
        velocity_index : int
            current index corresponding to a velocity vector in self.velocities

        Returns
        -------
        list(int)
            list of neighbor enumeration values
        """
        state = coords + self.velocities[velocity_index]
        valid_actions = MonteCarlo.get_valid_actions(state)

        neighbors = []
        for action in valid_actions:
            neighbor = self.mc.step(state, action)
            if self.mc.is_step_valid(state, neighbor):
                neighbors.append(neighbor)

        return neighbors

    def enumerate_node(self, x, y, z):
        """
        Calculates the enumeration value of a node based on its state on the map

        Parameters
        ----------
        x : int
            first coordinate of the current position
        y : int
            second coordinate of the current position
        z : int
            current index corresponding to a velocity vector in self.velocities

        Returns
        -------
        int
            enumeration value for the current node
        """
        return (
            x * self.grid.shape[1] * len(self.velocities)
            + y * len(self.velocities)
            + z
            + 1
        )
