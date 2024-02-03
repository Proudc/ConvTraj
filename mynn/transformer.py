import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def generate_square_previous_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_local_mask(sz, k=3):
    mask = torch.eye(sz)
    for i in range(1, k + 1):
        mask += torch.cat((torch.zeros(i, sz), torch.eye(sz)[:-i]), dim=0)
        mask += torch.cat((torch.zeros(sz, i), torch.eye(sz)[:, :-i]), dim=1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_empty_mask(sz):
    mask = torch.zeros(sz, sz)
    return mask


class Transformer(nn.Module):

    def __init__(self, lon_size, lat_size, d_out, max_seq_length, device, mask='empty', k = 3):
        super(Transformer, self).__init__()
        self.lon_size = lon_size
        self.lat_size = lat_size
        n_token       = lon_size * lat_size + 1
        self.device   = device
        if mask == "empty":
            self.mask = generate_empty_mask(max_seq_length).to(device)
        elif mask == "no_prev":
            self.mask = generate_square_previous_mask(max_seq_length).to(device)
        elif mask == "local":
            self.mask = generate_local_mask(max_seq_length, k=k).to(device)

        self.T_encoder = TransformerEncoderModel(n_token = n_token, d_out = d_out, max_len = max_seq_length)


    def forward(self, inputs_lon_array, inputs_lat_array, inputs_lon_lat_image_array):
        anchor_lon_list   = torch.stack(inputs_lon_array[0]).to(self.device)
        positive_lon_list = torch.stack(inputs_lon_array[1]).to(self.device)
        negative_lon_list = torch.stack(inputs_lon_array[2]).to(self.device)
        
        
        anchor_lat_list   = torch.stack(inputs_lat_array[0]).to(self.device)
        positive_lat_list = torch.stack(inputs_lat_array[1]).to(self.device)
        negative_lat_list = torch.stack(inputs_lat_array[2]).to(self.device)


        anchor_array   = anchor_lon_list * self.lat_size + anchor_lat_list + 1.0
        positive_array = positive_lon_list * self.lat_size + positive_lat_list + 1.0
        negative_array = negative_lon_list * self.lat_size + negative_lat_list + 1.0
        

        anchor_result   = self.T_encoder(anchor_array.to(torch.int32), self.mask)
        positive_result = self.T_encoder(positive_array.to(torch.int32), self.mask)
        negative_result = self.T_encoder(negative_array.to(torch.int32), self.mask)
        
        return anchor_result, positive_result, negative_result

    def inference(self, lon, lat, lon_lat):
        input = lon * self.lat_size + lat + 1.0
        result = self.T_encoder(input.to(torch.int32), self.mask)
        return result

class TransformerEncoderModel(nn.Module):

    def __init__(self, n_token, d_out, max_len, d_model = 128, nhead = 4, nhidden = 2048, nlayers = 3, dropout=0.0):
        super(TransformerEncoderModel, self).__init__()
        self.n_token       = n_token
        self.embedding     = nn.Embedding(n_token, d_model)
        self.pos_encoder   = PositionalEncoding(d_model, dropout, max_len=max_len)
        
        encoder_norm = nn.LayerNorm(d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, nhidden, dropout)
        self.encoder       = nn.TransformerEncoder(self.encoder_layer, nlayers, norm=encoder_norm)
        self.linear2       = nn.Linear(d_model, d_out)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear2.bias.data.zero_()
        self.linear2.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        batch_size, seq_length, _ = src.size()
        src = src.view(batch_size, seq_length)

        src    = self.embedding(src)
        src = src.permute(1, 0, 2)

        src    = self.pos_encoder(src)
        output = self.encoder(src, mask=src_mask)
        

        output = torch.mean(output, dim=0)
        return output
