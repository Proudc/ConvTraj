from distance.dtw     import dtw_dis
from distance.edr     import edr_dis
from distance.frechet import mine_frchet
from distance.haus    import hausdorff

import numpy as np


def cal_dist_between_traj(traj1, traj2, dist_type, matrix_flag = True):

    if dist_type == "edr":
        matrix, dist = edr_dis(traj1, traj2)
    elif dist_type == "dtw":
        matrix, dist = dtw_dis(traj1, traj2)
    elif dist_type == "frechet":
        matrix, dist = mine_frchet(traj1, traj2)
    elif dist_type == "haus":
        size_x = len(traj1) + 1
        size_y = len(traj2) + 1
        if matrix_flag == True:    
            matrix = np.ones((size_x, size_y))
            coords = np.array(np.nonzero(matrix)).T            
            r = 20
            selected_coords = coords[np.random.choice(coords.shape[0], r, replace=False)]

            matrix = [[0 for _ in range(size_y)] for _ in range(size_x)]

            for x in range(size_x):
                matrix[x][0] = x
            for y in range(size_y):
                matrix[0][y] = y

            for i, j in selected_coords:
                matrix[i][j] = hausdorff(traj1[:i], traj2[:j])
            matrix = np.array(matrix)[1:, 1:]
        else:
            matrix = [[0 for _ in range(size_y)] for _ in range(size_x)]

        dist = hausdorff(traj1, traj2)

    else:
        raise ValueError('Unknown dist type.')
    
    matrix = np.array(matrix)
    # print(matrix)
    # assert matrix.shape==(len(traj1), len(traj2)), "Matrix Size Must Be (size_traj1, size_traj2)"
    
    if matrix_flag == True:
        return matrix, dist
    else:
        return dist