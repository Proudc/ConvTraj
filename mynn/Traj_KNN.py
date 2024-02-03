import os
import time
import pickle
import numpy as np

import torch

from tools import sampling_methods
from tools import test_methods
from tools import function
from tools import torch_feature_distance
from tools import pre_rep

class NeuTrajTrainer(object):
    def __init__(self, my_config):
        self.my_config = my_config

    def data_prepare(self,
                     traj_list,
                     train_dist_matrix_path,
                     test_dist_matrix_path,
                     lon_onehot = None,
                     lat_onehot = None,
                     lon_grid_id_list = None,
                     lat_grid_id_list = None):
        print("Start Data Prepare...")
        
        self.traj_list        = traj_list
        self.traj_length_list = [len(traj) for traj in self.traj_list]
        self.total_traj_num   = len(self.traj_list)
        self.max_traj_length  = max(self.traj_length_list)
        self.min_traj_length  = min(self.traj_length_list)

        print("Total Traj Number: {}".format(self.total_traj_num))
        print("Max Traj Length : {}".format(self.max_traj_length))
        print("Min Traj Length : {}".format(self.min_traj_length))

        self.lon_onehot = lon_onehot
        self.lat_onehot = lat_onehot


        self.lon_lat_image = pre_rep.image_encode(lon_grid_id_list[:self.my_config.my_dict["train_set"]],
                                                  lat_grid_id_list[:self.my_config.my_dict["train_set"]],
                                                  self.my_config.my_dict["lon_input_size"],
                                                  self.my_config.my_dict["lat_input_size"])
        
        
        self.pad_total_lon_onehot    = function.pad_traj_list(self.lon_onehot[:self.my_config.my_dict["train_set"]], self.max_traj_length, pad_value = 0.0)  
        self.pad_total_lat_onehot    = function.pad_traj_list(self.lat_onehot[:self.my_config.my_dict["train_set"]], self.max_traj_length, pad_value = 0.0)  
        self.pad_total_lon_lat_image = self.lon_lat_image[:self.my_config.my_dict["train_set"]]


        self.pad_total_lon_onehot    = torch.tensor(self.pad_total_lon_onehot, dtype = torch.float32)
        self.pad_total_lat_onehot    = torch.tensor(self.pad_total_lat_onehot, dtype = torch.float32)
        self.pad_total_lon_lat_image = torch.tensor(self.pad_total_lon_lat_image, dtype = torch.float32)
        
        print("The Size Of Total Seq Pad List Is: ", self.pad_total_lon_onehot.size())

        self.total_train_lon_onehot = self.lon_onehot[:self.my_config.my_dict["train_set"]]
        self.total_train_lat_onehot = self.lat_onehot[:self.my_config.my_dict["train_set"]]
        
        self.pad_total_train_lon_onehot    = self.pad_total_lon_onehot[:self.my_config.my_dict["train_set"]]
        self.pad_total_train_lat_onehot    = self.pad_total_lat_onehot[:self.my_config.my_dict["train_set"]]
        self.pad_total_train_lon_lat_image = self.pad_total_lon_lat_image[:self.my_config.my_dict["train_set"]]

        self.train_distance_matrix = pickle.load(open(train_dist_matrix_path, 'rb'))
        

        if self.my_config.my_dict["dist_type"] == "dtw" or self.my_config.my_dict["dist_type"] == "edr":
            self.train_distance_matrix = self.train_distance_matrix / np.max(self.train_distance_matrix)
            
        
        self.avg_distance = np.mean(self.train_distance_matrix)
        self.max_distance = np.max(self.train_distance_matrix)
        print("Train Matrix size : {}".format(self.train_distance_matrix.shape))
        # print("Test  Matrix size : {}".format(self.test_distance_matrix.shape))
        print("Train Avg Distance: {}".format(self.avg_distance))
        print("Train Max Distance: {}".format(self.max_distance))

        self.final_traj_train_num = int(self.my_config.my_dict["train_set"] * self.my_config.my_dict["train_ratio"])
        print("Num Of Train Traj Is: {}".format(self.final_traj_train_num))
        print("Generating Train Traj Length List...")
        self.final_train_length_list = []
        for i in range(self.final_traj_train_num):
            self.final_train_length_list.append(self.traj_length_list[:self.final_traj_train_num])

        self.train_knn = np.empty(dtype=np.int32, shape=(self.final_traj_train_num, self.final_traj_train_num))
        for i in range(self.final_traj_train_num):
            self.train_knn[i] = np.argsort(self.train_distance_matrix[i][:self.final_traj_train_num])

        if ("0_porto_all" not in self.my_config.my_dict["root_read_path"]):
            self.test_distance_matrix  = pickle.load(open(test_dist_matrix_path, 'rb'))
            if self.my_config.my_dict["dist_type"] == "dtw" or self.my_config.my_dict["dist_type"] == "edr":
                self.test_distance_matrix  = self.test_distance_matrix / np.max(self.train_distance_matrix)
            self.test_knn = np.empty(dtype=np.int32, shape=(self.my_config.my_dict["query_set"], self.my_config.my_dict["base_set"]))
            for i in range(self.my_config.my_dict["query_set"]):
                self.test_knn[i] = np.argsort(self.test_distance_matrix[i])

        

        print("End Data Prepare !!!")
          
    def generate_train_data(self,
                            final_traj_train_num,
                            final_train_length_list,
                            batch_size,
                            train_distance_matrix,
                            sampling_num,
                            sampling_type,
                            test_knn,
                            lon_onehot,
                            lat_onehot,
                            lon_lat_image,
                            epoch):

        new_list = [[i, final_train_length_list[i]] for i in range(final_traj_train_num)]
        new_list = [x[0] for x in new_list]
        total_result = []
        for i in range(0, final_traj_train_num, batch_size):
            anchor_lon,         positive_lon,         negative_lon       = [], [], []
            anchor_lat,         positive_lat,         negative_lat       = [], [], []
            anchor_image,       positive_image,       negative_image     = [], [], []
            anchor_length,      positive_length,      negative_length    = [], [], []
            positive_distance,  negative_distance,    cross_distance     = [], [], []
            for j in range(batch_size):
                if i + j >= final_traj_train_num:
                    break
                anchor_pos = new_list[(i + j)]
                
                positive_sampling_index_list, negative_sampling_index_list = sampling_methods.main_triplet_selection(sampling_type, sampling_num, test_knn[anchor_pos], train_distance_matrix[anchor_pos], anchor_pos, final_train_length_list[anchor_pos], final_train_length_list, epoch)

                # cross distance
                for k in range(len(positive_sampling_index_list)):
                    cross_distance.append(train_distance_matrix[positive_sampling_index_list[k]][negative_sampling_index_list[k]])

                # positive distance
                for positive_index in positive_sampling_index_list:
                    anchor_lon.append(lon_onehot[anchor_pos])
                    anchor_lat.append(lat_onehot[anchor_pos])
                    anchor_image.append(lon_lat_image[anchor_pos])
                    
                    positive_lon.append(lon_onehot[positive_index])
                    positive_lat.append(lat_onehot[positive_index])
                    positive_image.append(lon_lat_image[positive_index])

                    anchor_length.append(self.final_train_length_list[anchor_pos])
                    positive_length.append(self.final_train_length_list[anchor_pos][positive_index])

                    positive_distance.append(train_distance_matrix[anchor_pos][positive_index])
                
                # negative distance
                for negative_index in negative_sampling_index_list:
                    negative_lon.append(lon_onehot[negative_index])
                    negative_lat.append(lat_onehot[negative_index])
                    negative_image.append(lon_lat_image[negative_index])
                    
                    negative_length.append(self.final_train_length_list[anchor_pos][negative_index])

                    negative_distance.append(train_distance_matrix[anchor_pos][negative_index])
            
            tem_batch = ([anchor_lon,        positive_lon,     negative_lon], 
                         [anchor_lat,        positive_lat,     negative_lat], 
                         [anchor_image,      positive_image,     negative_image], 
                         [anchor_length,     positive_length,    negative_length], 
                         [positive_distance, negative_distance,  cross_distance])
            total_result.append(tem_batch)
        return total_result


    def get_embeddings(self, my_net, test_batch, total_traj_num):
        my_net.eval()
        embedding_list = []
        start_time = time.time()
        with torch.no_grad():
            for i in range(0, total_traj_num, test_batch):
                input_lon_onehot_tensor    = self.pad_total_lon_onehot[i : i + test_batch]
                input_lat_onehot_tensor    = self.pad_total_lat_onehot[i : i + test_batch]
                input_lon_lat_image_tensor = self.pad_total_lon_lat_image[i : i + test_batch]

                out_feature = my_net.inference(input_lon_onehot_tensor, input_lat_onehot_tensor, input_lon_lat_image_tensor)
                embedding_list.append(out_feature.data)
        end_time = time.time()
        embedding_list = torch.cat(embedding_list, dim = 0)
        
        print("Embedding time: {:.4f}, Size of Embedding list: {}".format(end_time - start_time, embedding_list.size()))
        return embedding_list.cpu().numpy()

    def train(self):
        my_net    = function.initialize_model(self.my_config.my_dict, self.max_traj_length).to(self.my_config.my_dict["device"])
        my_loss   = function.initialize_loss(self.my_config.my_dict).to(self.my_config.my_dict["device"])
        optimizer = torch.optim.Adam(my_net.parameters(), lr = self.my_config.my_dict["learning_rate"])

        for name,parameters in my_net.named_parameters():
            print(name,':',parameters.size())
        trainable_num = sum(p.numel() for p in my_net.parameters() if p.requires_grad)
        print("Total Trainable Parameter Num:", trainable_num)

        learning_rate = self.my_config.my_dict["learning_rate"]

        for epoch in range(self.my_config.my_dict["epoch_num"]):
            start_time = time.time()

            my_net.train()
            my_loss.init_loss(epoch)
            
            train_data = self.generate_train_data(self.final_traj_train_num,
                                                  self.final_train_length_list,
                                                  self.my_config.my_dict["batch_size"],
                                                  self.train_distance_matrix,
                                                  self.my_config.my_dict["sampling_num"],
                                                  self.my_config.my_dict["sampling_type"],
                                                  self.train_knn,
                                                  self.pad_total_train_lon_onehot,
                                                  self.pad_total_train_lat_onehot,
                                                  self.pad_total_train_lon_lat_image,
                                                  epoch)
            
            for i, batch in enumerate(train_data):
                inputs_lon_array, inputs_lat_array, inputs_lon_lat_image_array, inputs_length_array, distance_array = batch[0], batch[1], batch[2], batch[3], batch[4]
                
                anchor_embedding, positive_embedding, negative_embedding = my_net(inputs_lon_array, inputs_lat_array, inputs_lon_lat_image_array)

                positive_distance_target = torch.tensor(distance_array[0]).to(self.my_config.my_dict["device"])
                negative_distance_target = torch.tensor(distance_array[1]).to(self.my_config.my_dict["device"])
                cross_distance_target    = torch.tensor(distance_array[2]).to(self.my_config.my_dict["device"])

                positive_learning_distance, \
                negative_learning_distance, \
                cross_learning_distance = torch_feature_distance.all_feature_distance(self.my_config.my_dict["all_feature_distance_type"],
                                                                                      anchor_embedding,
                                                                                      positive_embedding,
                                                                                      negative_embedding,
                                                                                      self.my_config.my_dict["channel"])
                
                if self.my_config.my_dict["loss_type"] == "triplet":
                    rank_loss, mse_loss, loss = my_loss(self.my_config,
                                                        epoch,
                                                        positive_learning_distance,
                                                        positive_distance_target,
                                                        negative_learning_distance,
                                                        negative_distance_target,
                                                        cross_learning_distance,
                                                        cross_distance_target)
                else:
                    raise ValueError("Loss Type Error")
            
                optimizer.zero_grad()                                
                loss.backward()
                optimizer.step()
            end_time = time.time()


            if (epoch + 1) % self.my_config.my_dict["print_epoch"] == 0:
                epoch_num = self.my_config.my_dict["epoch_num"]
                if self.my_config.my_dict["loss_type"] == "triplet":
                    print('Print Epoch: [{:3d}/{:3d}], Rank Loss: {:.4f}, Mse Loss: {:.4f}, Total Loss: {:.4f}, Time: {:.4f}'.format(epoch, epoch_num, rank_loss.item(), mse_loss.item(), loss.item(), (end_time - start_time)))
                else:
                    raise ValueError("Loss Type Error")
                

            if (epoch + 1) % self.my_config.my_dict["test_epoch"] == 0:  
                if not os.path.exists(self.my_config.my_dict["save_model_path"]):
                    os.mkdir(self.my_config.my_dict["save_model_path"])
                save_model_name = self.my_config.my_dict["save_model_path"] + "/" + self.my_config.my_dict["train_flag"]
                torch.save(my_net.state_dict(), save_model_name)

                
  


    def extract_feature_from_path(self,
                                  lon_grid_id_list = None,
                                  lat_grid_id_list = None):
        model_path = self.my_config.my_dict["save_model_path"] + "/" + self.my_config.my_dict["train_flag"]
        my_net = function.initialize_model(self.my_config.my_dict, self.max_traj_length).to(self.my_config.my_dict["device"])
        my_net.load_state_dict(torch.load(model_path))
        total_embeddings_list = []
        begin_pos, end_pos = 0, 10240
        total_time = 0
        while True:
            print(begin_pos, end_pos)
            self.pad_total_lon_onehot    = function.pad_traj_list(self.lon_onehot[begin_pos:end_pos], self.max_traj_length, pad_value = 0.0)  
            self.pad_total_lat_onehot    = function.pad_traj_list(self.lat_onehot[begin_pos:end_pos], self.max_traj_length, pad_value = 0.0)  
            self.pad_total_lon_lat_image = pre_rep.image_encode(lon_grid_id_list[begin_pos:end_pos],
                                                                lat_grid_id_list[begin_pos:end_pos],
                                                                self.my_config.my_dict["lon_input_size"],
                                                                self.my_config.my_dict["lat_input_size"])

            self.pad_total_lon_onehot    = torch.tensor(self.pad_total_lon_onehot, dtype = torch.float32).to(self.my_config.my_dict["device"])
            self.pad_total_lat_onehot    = torch.tensor(self.pad_total_lat_onehot, dtype = torch.float32).to(self.my_config.my_dict["device"])
            self.pad_total_lon_lat_image = torch.tensor(self.pad_total_lon_lat_image, dtype = torch.float32).to(self.my_config.my_dict["device"])

            time1 = time.time()
            embeddings_list = self.get_embeddings(my_net, 1024, end_pos - begin_pos)
            total_embeddings_list.extend(embeddings_list)
            time2 = time.time()
            total_time += time2 - time1
            if end_pos == len(self.traj_list):
                break
            begin_pos = end_pos
            end_pos += 10240
            if end_pos > len(self.traj_list):
                end_pos = len(self.traj_list)

        time1 = time.time()
        total_embeddings_list = np.array(total_embeddings_list)
        time2 = time.time()
        total_time += time2 - time1
        print(total_embeddings_list.shape)
        print("Total inference time: ", total_time)

        test_methods.test_all_log(total_embeddings_list, self.my_config, "feature", [], [], epoch = -1)