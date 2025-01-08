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
        self.ID = f'NormalDistanceDistribution-{sigma}'
        self.distribution = norm(loc=0, scale=sigma)

    def likelihood(self, center_x, center_y, x, y):
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
        for key, value in self.cache.items():
            np.save(f'{dir}/{key}.npy', value, allow_pickle=False)

    def load(self, dir):
        self.cache = {}
        for file in os.listdir(dir):
            key = file.split('.')[0]
            value = np.load(f'{dir}/{file}')
            self.cache[key] = value

# Functions

def get_transfer_matrix(transfer_dist, cache, x, y, resolution, processes=1, min_probability=10**-10):
    """
    Inputs:
    - transfer_dist (obj): a transfer distribution object
    - cache (obj): a caching object
    - x (np.array): array of x coordinates
    - y (np.array): array of y coordinates
    - resolution (float): width of a cell
    - processes (int): number of processes to use
    - min_probability (float): minimum probability to keep

    Outputs:
    - transfer_matrix (np.array): a transfer probability matrix
        columns are the starting locations and rows are the ending locations
    """
    transfer_matrix = cache.get(transfer_dist.ID)
    if transfer_matrix is None:
        transfer_matrix = build_transfer_matrix(transfer_dist, x, y, resolution, processes, min_probability)
        cache.set(transfer_dist.ID, transfer_matrix)
    return transfer_matrix


def integrate_likelihood(transfer_likelihood, source_x, source_y, resolution, dest_x, dest_y):
    """
    Inputs:
    - transfer_likelihood (func): transfer likelihood
    - source_x (float): x coordinate of source location
    - source_y (float): y coordinate of source location
    - dest_x (float): x coordinate of destination location
    - dest_y (float): y coordinate of destination location
    - resolution (float): width of the cell

    Outputs:
    - probability (float): the probability of transfer from source to dest cells
    """

    likelihood = partial(transfer_likelihood, source_x, source_y)

    x_min = dest_x - resolution / 2
    x_max = dest_x + resolution / 2
    y_min = dest_y - resolution / 2
    y_max = dest_y + resolution / 2

    probability, _ = integrate.dblquad(likelihood, x_min, x_max, lambda x: y_min, lambda x: y_max)
    return probability


def build_transfer_matrix(transfer_dist, x, y, resolution, processes=1, min_probability=10**-10):
    """
    Inputs:
    - transfer_dist (obj): a transfer distribution object
    - x (np.array): array of x coordinates
    - y (np.array): array of y coordinates
    - resolution (float): width of a cell
    - processes (int): number of processes to use
    - min_probability (float): minimum probability to keep

    Outputs:
    - transfer_matrix (np.array): a transfer probability matrix
        columns are the starting locations and rows are the ending locations
    """
    transfer_matrix = np.zeros((len(x), len(x)))
    for i, (source_x, source_y) in tqdm(enumerate(zip(x, y))):
        integrate = partial(
            integrate_likelihood, 
            transfer_dist.likelihood, source_x, source_y, resolution
        )
        with Pool(processes) as pool:
            column = np.array(pool.starmap(integrate, zip(x, y)))
            column[column < min_probability] = 0
            column /= np.sum(column)
        transfer_matrix[:, i] = column
    return transfer_matrix
