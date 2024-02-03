import torch
import argparse
import pickle

from tools import function
from tools import grid
from tools import pre_rep

from config.new_config import ConfigClass
from mynn.Traj_KNN import NeuTrajTrainer

import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="HyperParameters for Traj Embedding")

    parser.add_argument("--network_type",                   type=str,   default="TJCNN", help="network type")
    parser.add_argument("--loss_type",                      type=str,   default="triplet", help="loss type")
    parser.add_argument("--cnn_feature_distance_type",      type=str,   default="euclidean_sep", help="cnn feature distance type")
    parser.add_argument("--cnntotal_feature_distance_type", type=str,   default="euclidean", help="@Deprecated")
    parser.add_argument("--all_feature_distance_type",      type=str,   default="euclidean", help="all feature distance type")
    parser.add_argument("--sampling_type",                  type=str,   default="distance_sampling1", help="sampling type")
    parser.add_argument("--train_flag",                     type=str,   default="test", help="train flag")
    parser.add_argument("--head_num",                       type=str,   default=1, help="mlp head num")
    parser.add_argument("--area_path",                      type=str,   default="", help="selection area path")
    
    
    parser.add_argument("--target_size",                    type=int,   default=128, help="mlp target size")
    parser.add_argument("--channel",                        type=int,   default=8, help="channel num")
    parser.add_argument("--sampling_num",                   type=int,   default=1, help="sampling num for each sampling type")
    parser.add_argument("--epoch_num",                      type=int,   default=1000, help="epoch num")
    parser.add_argument("--device",                         type=str,   default="cuda:0", help="device")
    
    parser.add_argument("--learning_rate",                  type=float, default=0.001, help="learning rate")
    parser.add_argument("--train_ratio",                    type=float, default=1, help="train ratio")
    parser.add_argument("--batch_size",                     type=int,   default=128, help="batch size")
    parser.add_argument("--random_seed",                    type=int,   default=666, help="random seed")
    parser.add_argument("--mode",                           type=str,   default="train-directly", help="mode")
    parser.add_argument("--test_epoch",                     type=int,   default=5, help="test epoch")
    parser.add_argument("--print_epoch",                    type=int,   default=1, help="print epoch")
    parser.add_argument("--save_model",                     type=bool,  default=False, help="save model")
    parser.add_argument("--save_model_epoch",               type=int,   default=5, help="save model epoch")

    parser.add_argument("--root_write_path",                type=str,   default="/mnt/data_hdd1/czh/Neutraj/0_porto_10000", help="root write path")
    parser.add_argument("--root_read_path",                 type=str,   default="/mnt/data_hdd1/czh/Neutraj/0_porto_10000", help="root read path")
    parser.add_argument("--dist_type",                      type=str,   default="dtw", help="distance type")

    args = parser.parse_args()

    return args
    

if __name__ == "__main__":
    
    if not torch.cuda.is_available():
        print("GPU is Not Available.")
        exit()

    args   = get_args()
    device = torch.device(args.device)
    print("Device is:", device)

    function.setup_seed(args.random_seed)

    train_set, query_set, base_set = function.set_dataset(args.root_read_path)

    traj_list_path         = args.root_read_path + '/traj_list'
    train_dist_matrix_path = args.root_read_path + '/' + args.dist_type + '_train_distance_matrix_result'
    test_dist_matrix_path  = args.root_read_path + '/' + args.dist_type + '_test_distance_matrix_result'
    save_model_path        = args.root_write_path + "/model"

    traj_list = pickle.load(open(traj_list_path, "rb"))


    print("Time to meshing...")
    lon_grid_id_list, lat_grid_id_list, lon_input_size, lat_input_size, lon_list, lat_list = grid.split_traj_into_equal_grid(traj_list)


    my_config = ConfigClass(lon_input_size                 = lon_input_size,
                            lat_input_size                 = lat_input_size,
                            target_size                    = args.target_size,
                            batch_size                     = args.batch_size,
                            sampling_num                   = args.sampling_num,
                            learning_rate                  = args.learning_rate,
                            epoch_num                      = args.epoch_num,
                            network_type                   = args.network_type,
                            channel                        = args.channel,
                            loss_type                      = args.loss_type,
                            cnn_feature_distance_type      = args.cnn_feature_distance_type,
                            cnntotal_feature_distance_type = args.cnntotal_feature_distance_type,
                            all_feature_distance_type      = args.all_feature_distance_type,
                            sampling_type                  = args.sampling_type,
                            root_write_path                = args.root_write_path,
                            root_read_path                 = args.root_read_path,
                            train_ratio                    = args.train_ratio,
                            mode                           = args.mode,
                            test_epoch                     = args.test_epoch,
                            print_epoch                    = args.print_epoch,
                            save_model                     = args.save_model,
                            save_model_path                = save_model_path,
                            dist_type                      = args.dist_type,
                            device                         = device,
                            LDS                            = False,
                            FDS                            = False,
                            train_flag                     = args.train_flag,
                            head_num                       = int(args.head_num),
                            area_path                      = args.area_path,
                            train_set                      = train_set,
                            query_set                      = query_set,
                            base_set                       = base_set)
    
    if my_config.my_dict["network_type"] == "TJCNN":
        lon_onehot = lon_list
        lat_onehot = lat_list
    else:
        new_lon_grid_id_list = []
        for traj in lon_grid_id_list:
            tem_lon_list = []
            for value in traj:
                tem_lon_list.append([value])
            new_lon_grid_id_list.append(tem_lon_list)
        new_lat_grid_id_list = []
        for traj in lat_grid_id_list:
            tem_lat_list = []
            for value in traj:
                tem_lat_list.append([value])
            new_lat_grid_id_list.append(tem_lat_list)
        lon_onehot = new_lon_grid_id_list
        lat_onehot = new_lat_grid_id_list



    traj_network = NeuTrajTrainer(my_config)

    traj_network.data_prepare(traj_list,
                              train_dist_matrix_path,
                              test_dist_matrix_path,
                              lon_onehot = lon_onehot,
                              lat_onehot = lat_onehot,
                              lon_grid_id_list = lon_grid_id_list,
                              lat_grid_id_list = lat_grid_id_list)
    
    mode = my_config.my_dict["mode"]
    if mode == "test":
        traj_network.extract_feature()
    elif mode == "train-directly":
        traj_network.train()
        traj_network.extract_feature_from_path(lon_grid_id_list = lon_grid_id_list,
                                               lat_grid_id_list = lat_grid_id_list)
    else:
        raise ValueError("Train Mode Value Error!")



