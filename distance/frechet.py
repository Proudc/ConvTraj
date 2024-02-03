from random import randint
import numpy as np
import time
import matplotlib.pyplot as plt


__all__ = ['frdist']

import sys   
sys.setrecursionlimit(100000)


def _c(ca, i, j, p, q):

    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = np.linalg.norm(p[i]-q[j])
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(ca, i-1, 0, p, q), np.linalg.norm(p[i]-q[j]))
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(ca, 0, j-1, p, q), np.linalg.norm(p[i]-q[j]))
    elif i > 0 and j > 0:
        ca[i, j] = max(
            min(
                _c(ca, i-1, j, p, q),
                _c(ca, i-1, j-1, p, q),
                _c(ca, i, j-1, p, q)
            ),
            np.linalg.norm(p[i]-q[j])
            )
    else:
        ca[i, j] = float('inf')

    return ca[i, j]


def frdist(p, q):
    """
    Computes the discrete Fréchet distance between
    two curves. The Fréchet distance between two curves in a
    metric space is a measure of the similarity between the curves.
    The discrete Fréchet distance may be used for approximately computing
    the Fréchet distance between two arbitrary curves,
    as an alternative to using the exact Fréchet distance between a polygonal
    approximation of the curves or an approximation of this value.
    This is a Python 3.* implementation of the algorithm produced
    in Eiter, T. and Mannila, H., 1994. Computing discrete Fréchet distance.
    Tech. Report CD-TR 94/64, Information Systems Department, Technical
    University of Vienna.
    http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
    Function dF(P, Q): real;
        input: polygonal curves P = (u1, . . . , up) and Q = (v1, . . . , vq).
        return: δdF (P, Q)
        ca : array [1..p, 1..q] of real;
        function c(i, j): real;
            begin
                if ca(i, j) > −1 then return ca(i, j)
                elsif i = 1 and j = 1 then ca(i, j) := d(u1, v1)
                elsif i > 1 and j = 1 then ca(i, j) := max{ c(i − 1, 1), d(ui, v1) }
                elsif i = 1 and j > 1 then ca(i, j) := max{ c(1, j − 1), d(u1, vj) }
                elsif i > 1 and j > 1 then ca(i, j) :=
                max{ min(c(i − 1, j), c(i − 1, j − 1), c(i, j − 1)), d(ui, vj ) }
                else ca(i, j) = ∞
                return ca(i, j);
            end; /* function c */
        begin
            for i = 1 to p do for j = 1 to q do ca(i, j) := −1.0;
            return c(p, q);
        end.
    Parameters
    ----------
    P : Input curve - two dimensional array of points
    Q : Input curve - two dimensional array of points
    Returns
    -------
    dist: float64
        The discrete Fréchet distance between curves `P` and `Q`.
    Examples
    --------
    >>> from frechetdist import frdist
    >>> P=[[1,1], [2,1], [2,2]]
    >>> Q=[[2,2], [0,1], [2,4]]
    >>> frdist(P,Q)
    >>> 2.0
    >>> P=[[1,1], [2,1], [2,2]]
    >>> Q=[[1,1], [2,1], [2,2]]
    >>> frdist(P,Q)
    >>> 0
    """
    p = np.array(p, np.float64)
    q = np.array(q, np.float64)

    len_p = len(p)
    len_q = len(q)

    if len_p == 0 or len_q == 0:
        raise ValueError('Input curves are empty.')

    

    ca = (np.ones((len_p, len_q), dtype=np.float64) * -1)

    dist = _c(ca, len_p-1, len_q-1, p, q)
    return dist


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))


def discrete_frechet_distance(P, Q):
    n = len(P)
    m = len(Q)
    
    dp = np.array([[-1 for _ in range(m)] for _ in range(n)])
    
    def dfs(i, j):
        if dp[i][j] != -1:
            return dp[i][j]
        
        if i == 0 and j == 0:
            dp[i][j] = euclidean_distance(P[0], Q[0])
        elif i > 0 and j == 0:
            dp[i][j] = max(dfs(i-1, 0), euclidean_distance(P[i], Q[0]))
        elif i == 0 and j > 0:
            dp[i][j] = max(dfs(0, j-1), euclidean_distance(P[0], Q[j]))
        elif i > 0 and j > 0:
            dp[i][j] = max(min(dfs(i-1, j), dfs(i-1, j-1), dfs(i, j-1)), euclidean_distance(P[i], Q[j]))
        
        return dp[i][j]
    value = dfs(n-1, m-1)
    return value

def mine_frchet(P, Q):
    
    n = len(P)
    m = len(Q)
    dp = [[-1 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                dp[i][j] = euclidean_distance(P[0], Q[0])
                # print(P[0], Q[0], dp[i][j])
            elif i > 0 and j == 0:
                dp[i][j] = max(dp[i-1][0], euclidean_distance(P[i], Q[0]))
            elif i == 0 and j > 0:
                dp[i][j] = max(dp[0][j-1], euclidean_distance(P[0], Q[j]))
            else:
                dp[i][j] = max(min(dp[i][j-1], dp[i-1][j-1], dp[i-1][j]), euclidean_distance(P[i], Q[j]))

    return dp, dp[n-1][m-1]



if __name__ == "__main__":
    pass