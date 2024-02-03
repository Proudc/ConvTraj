import torch
import torch.nn as nn

import math

POOL = nn.MaxPool1d

class SimpleCNN(nn.Module):
    def __init__(self,
                 lon_input_size,
                 lat_input_size,
                 target_size,
                 batch_size,
                 sampling_num,
                 max_seq_length,
                 channel,
                 device,
                 head_num):
        super(SimpleCNN, self).__init__()
        self.lon_input_size = lon_input_size
        self.lat_input_size = lat_input_size
        self.target_size    = target_size
        self.max_seq_length = max_seq_length
        self.device         = device
        self.channel        = channel
        self.head_num       = int(head_num)
        
        total_layers = int(math.log2(self.max_seq_length))

        
        self.channel_1d = self.channel * 10
        self.conv_new = nn.Sequential(
            nn.Conv1d(2, self.channel_1d, 3, 1, padding=1, bias=False),
            nn.ReLU(),
            POOL(2),
        )
        for i in range(total_layers - 1):
            self.conv_new.add_module("conv{}".format(i + 1), nn.Conv1d(self.channel_1d, self.channel_1d, 3, 1, padding=1, bias=False)),
            nn.ReLU(),
            self.conv_new.add_module("pool{}".format(i + 1), POOL(2))
        

        self.flat_size = self.channel_1d
        
        
        self.total_size = self.flat_size
        
        self.fc1 = nn.Linear(self.total_size, self.total_size)
        self.fc2 = nn.Linear(self.total_size, self.target_size)

        

    def forward(self, inputs_lon_array, inputs_lat_array, inputs_lon_lat_image_array):
        
        anchor_lon_list,           positive_lon_list,           negative_lon_list           = [], [], []
        anchor_lat_list,           positive_lat_list,           negative_lat_list           = [], [], []
        anchor_lon_lat_image_list, positive_lon_lat_image_list, negative_lon_lat_image_list = [], [], []


        anchor_lon_list   = torch.stack(inputs_lon_array[0]).to(self.device)
        positive_lon_list = torch.stack(inputs_lon_array[1]).to(self.device)
        negative_lon_list = torch.stack(inputs_lon_array[2]).to(self.device)
        
        
        anchor_lat_list   = torch.stack(inputs_lat_array[0]).to(self.device)
        positive_lat_list = torch.stack(inputs_lat_array[1]).to(self.device)
        negative_lat_list = torch.stack(inputs_lat_array[2]).to(self.device)
        
        
        anchor_lon_lat_image_list   = torch.stack(inputs_lon_lat_image_array[0]).to(self.device)
        positive_lon_lat_image_list = torch.stack(inputs_lon_lat_image_array[1]).to(self.device)
        negative_lon_lat_image_list = torch.stack(inputs_lon_lat_image_array[2]).to(self.device)
        
        anchor_result   = self.encode(anchor_lon_list, anchor_lat_list, anchor_lon_lat_image_list)
        positive_result = self.encode(positive_lon_list, positive_lat_list, positive_lon_lat_image_list)
        negative_result = self.encode(negative_lon_list, negative_lat_list, negative_lon_lat_image_list)

        return anchor_result, positive_result, negative_result
    

    def encode(self, x, y, xy):
        seq_num = len(x)
        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)
        
        x = torch.cat((x, y), dim = 1)
        x = self.conv_new(x)
        x = x.view(seq_num, self.flat_size)


        cnn_feature = x


        cnn_feature = self.fc1(cnn_feature)
        cnn_feature = torch.relu(cnn_feature)
        cnn_feature = self.fc2(cnn_feature)
        
        return cnn_feature


    def inference(self, x, y, xy):
        seq_num = len(x)
        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)
        
        x = torch.cat((x, y), dim = 1)
        x = self.conv_new(x)
        x = x.view(seq_num, self.flat_size)

        cnn_feature = x


        cnn_feature = self.fc1(cnn_feature)
        cnn_feature = torch.relu(cnn_feature)
        cnn_feature = self.fc2(cnn_feature)
        
        return cnn_feature

    
