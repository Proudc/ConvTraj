import pickle

from distance.distance_compution import traj_dist_batch
from distance.distance_compution import traj_dist_combain

def cal_dist_matrix(traj_path, root_write_path, dist_type):
    traj_list = pickle.load(open(traj_path, 'rb'))
    traj_num = len(traj_list)
    print("-------------------------------------")
    print("TRAJ NUM: ", traj_num)
    print("-------------------------------------")
    traj_dist_batch(traj_list, root_write_path, dist_type, processors = 50)
    traj_dist_combain(traj_num, dist_type, root_write_path)



if __name__ == "__main__":
    root_write_path = "/mnt/data_hdd1/czh/Neutraj/0_porto_all/"
    
    traj_path       = root_write_path + "split_traj_list/train_list"

    dist_type_list = ["edr","frechet","haus", "dtw"]
    

    for dist_type in dist_type_list:
        cal_dist_matrix(traj_path, root_write_path, dist_type)
        