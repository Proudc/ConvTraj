import os
import pickle
import multiprocessing
import numpy as np


from distance.traj_dist import cal_dist_between_traj

def cdist(traj_list1, traj_list2, dist_type, matrix_flag):
    matrix = np.zeros((len(traj_list1), len(traj_list2)))
    dp_result = []
    for i, traj1 in enumerate(traj_list1):
        tem_dp_result = []
        for j, traj2 in enumerate(traj_list2):
            if matrix_flag == True:
                dp_matrix, similarity = cal_dist_between_traj(traj1, traj2, dist_type, matrix_flag = True)
            
                # print(similarity)
                matrix[i, j] = similarity
                coords = np.array(np.nonzero(dp_matrix)).T
                r = 10
                selected_coords = coords[np.random.choice(coords.shape[0], r, replace=False)]

                selected_values = np.array([[dp_matrix[x, y]] for x, y in selected_coords])
                # print(selected_values)

                result = [selected_coords, selected_values]
                tem_dp_result.append(result)
            else:
                similarity = cal_dist_between_traj(traj1, traj2, dist_type, matrix_flag = False)
                matrix[i, j] = similarity
        dp_result.append(tem_dp_result)
    return dp_result, matrix



def traj_all_dist(i, traj_list1, traj_list2, dist_type, root_write_path, matrix_flag, j = 0):
    print("Begin Compute Dist of Traj {} & {}".format(i, j))    
    dp_result_matrix, similarity_matrix = cdist(traj_list1, traj_list2, dist_type, matrix_flag)
    if matrix_flag == True:
        pickle.dump(similarity_matrix, open(root_write_path + "/" + str(dist_type) + "_dist_matrix/matrix_{}".format(i), 'wb'))
        pickle.dump(dp_result_matrix, open(root_write_path + "/" + str(dist_type) + "_dist_matrix/dp_matrix_{}".format(i), 'wb'))
    else:
        pickle.dump(similarity_matrix, open(root_write_path + "/" + str(dist_type) + "_dist_matrix/matrix_{}_{}".format(i, j), 'wb'))
    print("End Compute Dist of Traj {} & {}".format(i, j))

def traj_dist_batch(traj_list, root_write_path, dist_type, processors = 96):
    pool = multiprocessing.Pool(processes = processors)
    if not os.path.exists(root_write_path + "/" + str(dist_type) + "_dist_matrix"):
        os.makedirs(root_write_path + "/" + str(dist_type) + "_dist_matrix")
    for i, traj in enumerate(traj_list):
        if i < 3000:
            candidate_seq_list = traj_list[:3000]
            matrix_flag = True
            if not os.path.exists(root_write_path + "/" + str(dist_type) + "_dist_matrix/matrix_{}".format(i)):       
                pool.apply_async(traj_all_dist, (i, [traj], candidate_seq_list, dist_type, root_write_path, matrix_flag))
        if i >= 3000:
            break
    pool.close()
    pool.join()



def traj_dist_combain(traj_num, dist_type, root_write_path):
    row, column = 3000, 3000
    similarity_matrix_list = []
    dp_result_list = []
    for i in range(row):
        tem_matrix_list = pickle.load(open(root_write_path + "/" + str(dist_type) + "_dist_matrix/matrix_{}".format(i), 'rb'))
        tem_dp_list = pickle.load(open(root_write_path + "/" + str(dist_type) + "_dist_matrix/dp_matrix_{}".format(i), 'rb'))
        for j in range(len(tem_matrix_list)):
            similarity_matrix_list.append(tem_matrix_list[j])
            dp_result_list.append(tem_dp_list[j])
    
    similarity_matrix = np.array(similarity_matrix_list)    
    dp_matrix = dp_result_list    
    pickle.dump(similarity_matrix, open(root_write_path + "/" + str(dist_type) + "_train_distance_matrix_result", 'wb'))
    pickle.dump(dp_matrix, open(root_write_path + "/" + str(dist_type) + "_train_dp_matrix_result", 'wb'))
    print("The shape of train dist_matrix is {}".format(similarity_matrix.shape))

    # row, column = 1000, traj_num - 4000
    # similarity_matrix_list = []
    # for i in range(row):
    #     tem_matrix_list = pickle.load(open(root_write_path + "/" + str(dist_type) + "_dist_matrix/matrix_{}".format(i + 3000), 'rb'))
    #     for j in range(len(tem_matrix_list)):
    #         similarity_matrix_list.append(tem_matrix_list[j])

    # similarity_matrix = np.array(similarity_matrix_list)
    # pickle.dump(similarity_matrix, open(root_write_path + "/" + str(dist_type) + "_test_distance_matrix_result", 'wb'))
    # print("The shape of text dist_matrix is {}".format(similarity_matrix.shape))
        