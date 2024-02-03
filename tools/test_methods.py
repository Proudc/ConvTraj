import os
import pickle
import numpy as np

from tools import feature_distance


def test_all_log(total_embeddings, my_config, test_flag, train_distance_matrix, test_distance_matrix, epoch = -1):
    root_write_path  = my_config.my_dict["root_write_path"]
    train_flag       = my_config.my_dict["train_flag"]
    train_embeddings = total_embeddings[:my_config.my_dict["train_set"]]
    query_embeddings = total_embeddings[my_config.my_dict["train_set"]:my_config.my_dict["train_set"] + my_config.my_dict["query_set"]]
    base_embeddings  = total_embeddings[my_config.my_dict["train_set"] + my_config.my_dict["query_set"]:]

    if not os.path.exists(root_write_path + "/feature_distance_dir/"):
        os.mkdir(root_write_path + "/feature_distance_dir/")

    if not os.path.exists(root_write_path + "/feature_dir/"):
        os.mkdir(root_write_path + "/feature_dir/")

    if epoch == -1:
        pickle.dump(np.array(train_distance_matrix), open(root_write_path + "/feature_distance_dir/train_" + my_config.my_dict["dist_type"] + "_" + test_flag + "_" + train_flag, "wb"))
        pickle.dump(np.array(test_distance_matrix), open(root_write_path + "/feature_distance_dir/test_" + my_config.my_dict["dist_type"] + "_" + test_flag + "_" + train_flag, "wb"))
    else:
        pickle.dump(np.array(train_distance_matrix), open(root_write_path + "/feature_distance_dir/train_" + my_config.my_dict["dist_type"] + "_" + test_flag + "_" + train_flag + "_" + str(epoch), "wb"))
        pickle.dump(np.array(test_distance_matrix), open(root_write_path + "/feature_distance_dir/test_" + my_config.my_dict["dist_type"] + "_" + test_flag + "_" + train_flag + "_" + str(epoch), "wb"))

    if epoch == -1:
        pickle.dump(np.array(train_embeddings), open(root_write_path + "/feature_dir/train_" + my_config.my_dict["dist_type"] + "_" + test_flag + "_" + train_flag, "wb"))
        pickle.dump(np.array(query_embeddings), open(root_write_path + "/feature_dir/query_" + my_config.my_dict["dist_type"] + "_" + test_flag + "_" + train_flag, "wb"))
        pickle.dump(np.array(base_embeddings), open(root_write_path + "/feature_dir/base_" + my_config.my_dict["dist_type"] + "_" + test_flag + "_" + train_flag, "wb"), protocol=4)
        pickle.dump(np.array(total_embeddings), open(root_write_path + "/feature_dir/" + my_config.my_dict["dist_type"] + "_" + test_flag + "_" + train_flag, "wb"), protocol=4)
    else:
        pickle.dump(np.array(train_embeddings), open(root_write_path + "/feature_dir/train_" + my_config.my_dict["dist_type"] + "_" + test_flag + "_" + train_flag + "_" + str(epoch), "wb"))
        pickle.dump(np.array(query_embeddings), open(root_write_path + "/feature_dir/query_" + my_config.my_dict["dist_type"] + "_" + test_flag + "_" + train_flag + "_" + str(epoch), "wb"))
        pickle.dump(np.array(base_embeddings), open(root_write_path + "/feature_dir/base_" + my_config.my_dict["dist_type"] + "_" + test_flag + "_" + train_flag + "_" + str(epoch), "wb"), protocol=4)
        pickle.dump(np.array(total_embeddings), open(root_write_path + "/feature_dir/" + my_config.my_dict["dist_type"] + "_" + test_flag + "_" + train_flag + "_" + str(epoch), "wb"), protocol=4)
    
def topk_recall(query_id_list, base_id_list, distance_matrix, test_distance_matrix):
    result = []
    s_test_distance_dict = {}
    s_true_distance_dict = {}
    for query_id in query_id_list:
        test_distance = []
        true_distance = []
        for base_id in base_id_list:
            test_distance.append((base_id, test_distance_matrix[query_id][base_id]))
            true_distance.append((base_id, distance_matrix[query_id][base_id]))
        s_test_distance    = sorted(test_distance, key = lambda a: a[1], reverse = False)
        s_true_distance    = sorted(true_distance, key = lambda a: a[1], reverse = False)
        s_test_distance_dict[query_id] = s_test_distance
        s_true_distance_dict[query_id] = s_true_distance

    top_test_num = [1]
    top_true_num = [1]
    
    for tem_test_num in top_test_num:
        top_count_list = [0 for i in range(len(top_true_num))]
        for query_id in query_id_list:
            s_test_distance = s_test_distance_dict[query_id]
            s_true_distance = s_true_distance_dict[query_id]
            for pos in range(len(top_true_num)):
                tem_true_num = top_true_num[pos]
                tem_top_list = [l[0] for l in s_test_distance[:tem_test_num] if l[0] in [j[0] for j in s_true_distance[:tem_true_num]]]
                if tem_test_num == 1 and tem_true_num == 1 and s_test_distance[0][0] != s_true_distance[0][0]:
                    top11_true_list = [s_true_distance[0][0]]
                    for id, value in s_true_distance:
                        if value == s_true_distance[0][1]:
                            top11_true_list.append(id)
                    if s_test_distance[0][0] in top11_true_list:
                        tem_top_list.append(s_test_distance[0][0])
                top_count_list[pos] += len(tem_top_list)
        for pos in range(len(top_true_num)):
            tem_recall = top_count_list[pos] / (len(query_id_list) * top_true_num[pos])
            result.append(tem_recall)
    return result

def test_all_print(query_id_list, base_id_list, distance_matrix, test_distance_matrix):
    recall_list  = topk_recall(query_id_list, base_id_list, distance_matrix, test_distance_matrix)
    return recall_list

def intersect_sizes(true_list, test_list):
    return np.array([len(np.intersect1d(true_value, list(test_value))) for true_value, test_value in zip(true_list, test_list)])


def test_all_print_new(query_id_list, base_id_list, true_knn, test_distance_matrix):
    
    test_knn = np.argsort(test_distance_matrix)

    top_test_dict = {5:[5], 10:[10], 50:[50, 10], 100:[100]}

    for tem_test_num in top_test_dict.keys():
        top_true_list  = top_test_dict[tem_test_num]
        test_top_id    = test_knn[:, :tem_test_num]
        intersect_list = [intersect_sizes(true_knn[:, :tem_true_num], test_top_id) / float(tem_true_num) for tem_true_num in top_true_list]
        recall_list    = [np.mean(tem_list) for tem_list in intersect_list]
        for pos in range(len(top_true_list)):
            if tem_test_num == 5 and top_true_list[pos] == 5:
                top5_recall = recall_list[pos]
            if tem_test_num == 10 and top_true_list[pos] == 10:
                top10_recall = recall_list[pos]
            if tem_test_num == 50 and top_true_list[pos] == 50:
                top50_recall = recall_list[pos]
            if tem_test_num == 100 and top_true_list[pos] == 100:
                top100_recall = recall_list[pos]
            if tem_test_num == 50 and top_true_list[pos] == 10:
                top10_50_recall = recall_list[pos]
            
    total_list = []
    recall_list = [top5_recall, top10_recall, top50_recall, top100_recall, top10_50_recall]
    total_list.extend(recall_list)
    return total_list


def get_feature_distance(query_embeddings, base_embeddings, test_flag, my_config):
    cnn_feature_distance_type      = my_config.my_dict["cnn_feature_distance_type"]
    cnntotal_feature_distance_type = my_config.my_dict["cnntotal_feature_distance_type"]
    all_feature_distance_type      = my_config.my_dict["all_feature_distance_type"]
    if test_flag == "cnnfeature" and "euclidean" in cnn_feature_distance_type:
        test_distance_matrix = feature_distance.l2_dist_separate(query_embeddings, base_embeddings)
    elif test_flag == "cnnfeature" and "manhattan" in cnn_feature_distance_type:
        test_distance_matrix = feature_distance.manhattan_dist_separate(query_embeddings, base_embeddings)
    elif test_flag == "cnnfeature" and "hyperbolic" in cnn_feature_distance_type:
        test_distance_matrix = feature_distance.hyperbolic_dist_separate(query_embeddings, base_embeddings)
    elif test_flag == "cnntotalfeature" and cnntotal_feature_distance_type == "euclidean":
        test_distance_matrix = feature_distance.l2_dist(query_embeddings, base_embeddings)
    elif test_flag == "cnntotalfeature" and cnntotal_feature_distance_type == "hyperbolic":
        test_distance_matrix = feature_distance.hyperbolic_dist(query_embeddings, base_embeddings)
    elif test_flag == "allfeature" and all_feature_distance_type == "euclidean":
        test_distance_matrix = feature_distance.l2_dist(query_embeddings, base_embeddings)
    elif test_flag == "allfeature" and all_feature_distance_type == "manhattan":
        test_distance_matrix = feature_distance.manhattan_dist(query_embeddings, base_embeddings)
    elif test_flag == "allfeature" and all_feature_distance_type == "hyperbolic":
        test_distance_matrix = feature_distance.hyperbolic_dist(query_embeddings, base_embeddings)
    else:
        raise ValueError('Unsupported Test Flag: {}'.format(test_flag))
    return test_distance_matrix
    