import numpy as np

import torch
from torch.nn import functional

from scipy.spatial.distance import cdist

def euclidean_torch(x, y):
    """
    Computes the Euclidean distance between two tensors.
    """
    return torch.norm(x - y, dim = -1)

def euclidean_numpy(x, y):
    """
    Computes the Euclidean distance between two numpy arrays.
    """
    return np.linalg.norm(x - y)


def euclidean_torch_separate(x, y, channel = 8):
    N = len(x)
    x = x.view(N, -1, channel)
    y = y.view(N, -1, channel)
    return torch.norm(x - y, dim = -1)


def euclidean_torch_separate_sum(x, y, channel = 8):
    N = len(x)
    x = x.view(N, -1, channel)
    y = y.view(N, -1, channel)

    return torch.norm(x - y, dim = -1).sum(dim=1)





def l2_dist(q: np.ndarray, x: np.ndarray):
    assert len(q.shape) == 2
    assert len(x.shape) == 2
    assert q.shape[1] == q.shape[1]

    x = x.T
    sqr_q = np.sum(q ** 2, axis=1, keepdims=True)
    sqr_x = np.sum(x ** 2, axis=0, keepdims=True)
    l2 = sqr_q + sqr_x - 2 * q @ x
    l2[ np.nonzero(l2 < 0) ] = 0.0
    return np.sqrt(l2)
