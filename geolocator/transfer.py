"""
Functions for generating transfer probability matrices.
"""
import os
from multiprocessing import Pool

import numpy as np
from scipy.stats import norm
from scipy import integrate
from tqdm import tqdm
from functools import partial

# Transfer Distributions

class NormalDistanceDistribution:
    def __init__(self, sigma):
        self.sigma = sigma
        self.distribution = norm(loc=0, scale=sigma)

        self.ID = f'NormalDistanceDistribution_{sigma}'

    def likelihood(self, center_y, center_x, y, x):
        distance = np.sqrt((center_x - x)**2 + (center_y - y)**2)
        return self.distribution.pdf(distance)
    
# Caches 

class MemoryCache:
    def __init__(self):
        self.cache = {}
    
    def get(self, key):
        return self.cache.get(key, None)
    
    def set(self, key, value):
        self.cache[key] = value

    def clear(self):
        self.cache = {}

    def save(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)
        for key, value in self.cache.items():
            np.save(f'{dir}/{key}.npy', value, allow_pickle=False)

    def load(self, dir):
        self.cache = {}
        for file in os.listdir(dir):
            value = np.load(f'{dir}/{file}')
            key = '.'.join(file.split('.')[:-1])
            self.cache[key] = value

# Functions

def get_transfer_matrix(transfer_dist, world, cache, processes=1, min_probability=10**-10):
    """
    Inputs:
    - transfer_dist (obj): a transfer distribution object
    - world (obj): a world object
    - cache (obj): a caching object
    - processes (int): number of processes to use
    - min_probability (float): minimum probability to keep

    Outputs:
    - transfer_matrix (np.array): a transfer probability matrix
        columns are the starting locations and rows are the ending locations
    """
    assert '+' not in transfer_dist.ID
    assert '+' not in world.ID

    cache_key = f'{transfer_dist.ID}+{world.ID}'
    transfer_matrix = cache.get(cache_key)
    if transfer_matrix is None:
        transfer_matrix = build_transfer_matrix(transfer_dist, world, processes, min_probability)
        cache.set(cache_key, transfer_matrix)
    return transfer_matrix


def integrate_likelihood(transfer_likelihood, world, source_x, source_y, dest_x, dest_y):
    """
    Inputs:
    - transfer_likelihood (func): transfer likelihood
    - world (obj): a world object
    - source_x (float): x coordinate of source location
    - source_y (float): y coordinate of source location
    - dest_x (float): x coordinate of destination location
    - dest_y (float): y coordinate of destination location

    Outputs:
    - probability (float): the probability of transfer from source to dest cells
    """

    likelihood = partial(transfer_likelihood, source_y, source_x)

    resolution = world.resolution
    x_min = dest_x - resolution / 2
    x_max = dest_x + resolution / 2
    y_min = dest_y - resolution / 2
    y_max = dest_y + resolution / 2

    probability, _ = integrate.dblquad(likelihood, x_min, x_max, lambda x: y_min, lambda x: y_max)
    return probability


def build_transfer_matrix(transfer_dist, world, processes=1, min_probability=10**-10):
    """
    Inputs:
    - transfer_dist (obj): a transfer distribution object
    - world (obj): a world object
    - processes (int): number of processes to use
    - min_probability (float): minimum probability to keep

    Outputs:
    - transfer_matrix (np.array): a transfer probability matrix
        columns are the starting locations and rows are the ending locations
    """
    x, y = world.world[:2]
    transfer_matrix = np.zeros((len(x), len(x)))
    iterator = list(enumerate(zip(x, y)))
    for i, (source_x, source_y) in tqdm(iterator):
        integrate = partial(
            integrate_likelihood,
            transfer_dist.likelihood, world, 
            source_x, source_y
        )
        with Pool(processes) as pool:
            column = np.array(pool.starmap(integrate, zip(x, y)))
            column[column < min_probability] = 0
            column /= np.sum(column)
        transfer_matrix[:, i] = column
    return transfer_matrix
