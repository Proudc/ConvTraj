import torch
import random
import numpy as np

from mynn.transformer import Transformer



from mynn.shortcut import ShortCutCNN as ConvTraj
from mynn.simplest_cnn import SimpleCNN




from loss.triplet import TripletLoss


def set_dataset(root_read_path):
    if "0_geolife" in root_read_path:
        train_set, query_set, base_set = 3000, 1000, 13386 - 4000
    elif "0_porto_all" in root_read_path:
        train_set, query_set, base_set = 3000, 1000, 1601579 - 4000
    else:
        raise Exception("root_read_path error")
    return train_set, query_set, base_set


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def pad_traj_list(seq_list, max_length, pad_value = 0.0):
    value = [1.0 * pad_value for i in range(len(seq_list[0][0]))]
    final_pad_seq_list = []
    for seq in seq_list:
        assert len(seq) <= max_length, "Sequence length {} is larger than max_length {}".format(len(seq), max_length)

        for j in range(max_length - len(seq)):
            seq.append(value)
        final_pad_seq_list.append(seq)
    return final_pad_seq_list



def initialize_loss(my_dict):
    batch_size   = my_dict["batch_size"]
    sampling_num = my_dict["sampling_num"]
    epoch_num    = my_dict["epoch_num"]
    if my_dict["loss_type"] == "triplet":
        my_loss = TripletLoss(epoch_num)
    else:
        raise ValueError("Loss Type Error")
    print("Init {} Loss Done !!!".format(my_dict["loss_type"]))
    return my_loss

def initialize_model(my_dict, max_traj_length):

    lon_input_size = my_dict["lon_input_size"]
    lat_input_size = my_dict["lat_input_size"]
    target_size    = my_dict["target_size"]
    batch_size     = my_dict["batch_size"]
    sampling_num   = my_dict["sampling_num"]
    device         = my_dict["device"]
    channel        = my_dict["channel"]
    head_num       = my_dict["head_num"]
    
    if ("CNN" in my_dict["network_type"]) and ("0_porto_all" in my_dict["root_read_path"]):
        my_net = ConvTraj(lon_input_size, lat_input_size, target_size, batch_size, sampling_num, max_traj_length, channel, device, head_num)
    elif ("CNN" in my_dict["network_type"]) and ("0_geolife" in my_dict["root_read_path"]):
        my_net = ConvTraj(lon_input_size, lat_input_size, target_size, batch_size, sampling_num, max_traj_length, channel, device, head_num)
    elif ("CNN" in my_dict["network_type"]) and ("Test_porto" in my_dict["root_read_path"]):
        my_net = SimpleCNN(lon_input_size, lat_input_size, target_size, batch_size, sampling_num, max_traj_length, channel, device, head_num)
    elif my_dict["network_type"] == "Global_T":
        my_net = Transformer(lon_input_size, lat_input_size, target_size, max_traj_length, device)
    elif "Local_T" in my_dict["network_type"]:
        windows_size = int(my_dict["network_type"][7:])
        my_net = Transformer(lon_input_size, lat_input_size, target_size, max_traj_length, device, mask = "local", k = windows_size)
    else:
        raise ValueError("Network Type Error")
    
    print("Init {} Model Done !!!".format(my_dict["network_type"]))
    return my_net

if __name__ == '__main__':
    pass