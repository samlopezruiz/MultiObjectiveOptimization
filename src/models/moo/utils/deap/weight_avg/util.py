import numpy as np


def weighted_avg(ind, weights, function):
    return (np.array(function(ind)).dot(weights), )
