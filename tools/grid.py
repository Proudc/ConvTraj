import random
import numpy as np
import matplotlib.pyplot as plt

GRID_SIZE = 0.002


def _equal_grid(one_dim_list):
    min_value_list = [np.min(value) for value in one_dim_list]
    max_value_list = [np.max(value) for value in one_dim_list]
    
    min_value = np.min(min_value_list)
    max_value = np.max(max_value_list)
    
    grid_size = GRID_SIZE

    grid_id_list = []
    test_max_grid = 0
    for traj in one_dim_list:
        single_list = []
        for value in traj:
            grid_id = int((value - min_value) / grid_size)
            if grid_id > test_max_grid:
                test_max_grid = grid_id
            single_list.append(grid_id)
        grid_id_list.append(single_list)
    
    max_grid_id = int((max_value - min_value) / grid_size)
    print(test_max_grid, max_grid_id)
    return grid_id_list, max_grid_id, min_value, max_value

def split_traj_into_equal_grid(traj_list):
    lon_list = []
    lat_list = []
    for traj in traj_list:
        tem_lon = []
        tem_lat = []
        for point in traj:
            tem_lon.append(point[0])
            tem_lat.append(point[1])
        lon_list.append(tem_lon)
        lat_list.append(tem_lat)
    
    lon_grid_id_list, lon_max_grid_id, lon_min_value, lon_max_value = _equal_grid(lon_list)
    lat_grid_id_list, lat_max_grid_id, lat_min_value, lat_max_value = _equal_grid(lat_list)

    lon_list = []
    lat_list = []
    for traj in traj_list:
        tem_lon = []
        tem_lat = []
        for point in traj:
            tem_lon.append([(point[0] - (lon_min_value)) * 100])
            tem_lat.append([(point[1] - (lat_min_value)) * 100])
        lon_list.append(tem_lon)
        lat_list.append(tem_lat)

    return lon_grid_id_list, lat_grid_id_list, lon_max_grid_id + 1, lat_max_grid_id + 1, lon_list, lat_list

    



def generate_random_trajectory(num_points, lon_range, lat_range):
    trajectory = []
    for _ in range(num_points):
        lon = random.uniform(lon_range[0], lon_range[1])
        lat = random.uniform(lat_range[0], lat_range[1])
        trajectory.append((lon, lat))
    return trajectory

if __name__ == "__main__":
    pass