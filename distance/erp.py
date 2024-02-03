import numpy as np
from scipy.spatial.distance import cdist

def eucl_dist(x, y):
    """
    Usage
    -----
    L2-norm between point x and y

    Parameters
    ----------
    param x : numpy_array
    param y : numpy_array

    Returns
    -------
    dist : float
           L2-norm between x and y
    """
    dist = np.linalg.norm(x - y)
    return dist


def eucl_dist_traj(t1, t2):
    """
    Usage
    -----
    Pairwise L2-norm between point of trajectories t1 and t2

    Parameters
    ----------
    param t1 : len(t1)x2 numpy_array
    param t2 : len(t1)x2 numpy_array

    Returns
    -------
    dist : float
           L2-norm between x and y
    """
    mdist = cdist(t1, t2, 'euclidean')
    return mdist

def erp(t0, t1, g):
    """
    Usage
    -----
    The Edit distance with Real Penalty between trajectory t0 and t1.

    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array

    Returns
    -------
    erp : float
          The Edit distance with Real Penalty between trajectory t0 and t1. 
    """

    n0 = len(t0)
    n1 = len(t1)
    C = np.zeros((n0 + 1, n1 + 1))

    gt0_dist = [abs(eucl_dist(g, x)) for x in t0]
    gt1_dist = [abs(eucl_dist(g, x)) for x in t1]
    mdist = eucl_dist_traj(t0, t1)

    C[1:, 0] = sum(gt0_dist)
    C[0, 1:] = sum(gt1_dist)
    for i in np.arange(n0) + 1:
        for j in np.arange(n1) + 1:
            derp0 = C[i - 1, j] + gt0_dist[i-1]
            derp1 = C[i, j - 1] + gt1_dist[j-1]
            derp01 = C[i - 1, j - 1] + mdist[i-1, j-1]
            C[i, j] = min(derp0, derp1, derp01)
    erp = C[n0, n1]
    return erp
