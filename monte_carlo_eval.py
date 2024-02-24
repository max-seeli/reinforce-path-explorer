from map import MapLoader
import json
import numpy as np
import os
from monte_carlo import MonteCarlo
import warnings
import time

with open("maps/opt_path.json", "r") as file:
    maps = json.load(file)

def eval(map_name, epsilon, num_episodes=10000, checkpoint=200):
    """
    Evaluate the Monte Carlo control algorithm on a map.
    
    Parameters
    ----------
    map_name : str
        The name of the map to evaluate.
    epsilon : float
        The epsilon value for the Monte Carlo control algorithm.
    num_episodes : int, optional
        The number of episodes to run. Default is 10000.
    checkpoint : int, optional
        The interval at which to save the policy. Default is 200.

    Returns
    -------
    float
        The total time taken to run the algorithm.
    list of float
        The time taken for each epoch.
    float
        The mean absolute error of the policy.
    list of float
        The mean absolute error of the policy at each checkpoint.
    """
    
    map_data = maps[map_name]
    grid = MapLoader.load_map(map_data["file"])
    policy_filename = f"./policies/{map_name}.policy.txt"
    finder = MonteCarlo(grid, 
                        num_episodes=num_episodes,
                        train_epsilon=epsilon,
                        policy_filename=policy_filename,
                        load_policy=False,
                        checkpoint=checkpoint)
    
    start = time.time()
    epoch_times = finder.monte_carlo_control()
    end = time.time()
    total_time = end - start
    
    overall_effectiveness = calculate_mae(map_name, policy_filename)
    checkpoint_effectiveness = [calculate_mae(map_name, f"./policies/{map_name}_{i}.policy.txt") for i in range(0, num_episodes, checkpoint)]

    return total_time, epoch_times, overall_effectiveness, checkpoint_effectiveness

    
    

def calculate_mae(map_name, policy_file):
    """
    Calculate the mean absolute error of the policy on a map.

    Parameters
    ----------
    map_name : str
        The name of the map to evaluate.
    policy_file : str
        The file to load the policy from.

    Returns
    -------
    float
        The mean absolute error of the policy.
    """
    
    map_data = maps[map_name]
    grid = MapLoader.load_map(map_data["file"])
    finder = MonteCarlo(grid, 
                        test_epsilon=0,
                        policy_filename=policy_file,
                        load_policy=True)
    
    sum_absolutes = 0
    for i, start in enumerate(map_data["start_positions"]):
        start = tuple(start)
        episode = finder.generate_episode(start, max_steps=1000)
        error = len(episode) - map_data["optimal_path_lengths"][i]
        if error < 0:
            warnings.warn(f"Error is negative: Found path of length {len(episode)} for optimal length {map_data['optimal_lengths'][i]} at start position {start}.")
        sum_absolutes += abs(error)
    
    mae = sum_absolutes / len(map_data["start_positions"])
    return mae
