import numpy as np
import sys

def point_dis(a, b):
    dis = np.linalg.norm(a - b)
    return dis

def edr_dis(seq1, seq2):  
    seq1 = np.array(seq1)
    seq2 = np.array(seq2)
    sigma = 0.002
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    #draw matrix
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y
    #compute
    for x in range(1, size_x):
        for y in range(1, size_y):
            if point_dis(seq1[x-1], seq2[y-1]) <= sigma:
                edit = 0                    
            else:
                edit = 1
            x1y = matrix[x-1,y] + 1         #[x - 1, y]
            x1y1 = matrix[x-1,y-1] + edit   #[x - 1, y - 1]
            xy1 = matrix[x,y-1] + 1         #[x, y - 1]
            matrix [x,y] = min(x1y, x1y1, xy1)
    # print(matrix)
    return matrix[1:, 1:], matrix[size_x - 1, size_y - 1]
        




if __name__ == '__main__':
    C = np.array([[1, 1], [2, 1]])
    Q = np.array([[1, 1], [2, 1], [2, 1]])
    C = [[1, 1], [2, 1]]
    Q = [[1, 1], [2, 1], [2, 1]]
    matrix, dist = edr_dis(C,Q)
    print('Levenshtein distance: ', dist)
    assert matrix.shape==(len(C), len(Q)), "Matrix Size Must Be (size_traj1, size_traj2)"
    print(matrix.shape)
    