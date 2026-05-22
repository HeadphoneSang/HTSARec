import math

import torch
from torch import nn


class AttentionScoreEncoder(nn.Module):
    def __init__(self, hidden_size, inspire_act_name):
        super().__init__()
        self.hidden_size = hidden_size
        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.t_K = nn.Linear(hidden_size, hidden_size)
        self.sqrt_d = math.sqrt(hidden_size)
        self.act = self.get_inspire_activation(inspire_act_name)

    def get_inspire_activation(self, inspire_act_name):
        inspire_act_name = inspire_act_name.lower()
        act_dict = {
            'sigmod': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'softplus': nn.Softplus()
        }
        return act_dict.get(inspire_act_name)

    def forward(self, x_tensors, item_seq_mat, time_intervals_mat, masked_attention=None):

        item_seq_mat = item_seq_mat.unsqueeze(1)
        x_query = self.Q(x_tensors)
        x_key = self.K(item_seq_mat)
        time_key = self.t_K(time_intervals_mat)
        total_key = x_key + time_key  #(batch,masked_len,seq_len,hidden_size)
        attn_score = (torch.matmul(x_query, total_key.transpose(2, 3)) / self.sqrt_d)
        #(batch,masked_len,sample_len,seq_len)
        if masked_attention is not None:
            attn_score = masked_attention + attn_score
        if self.act is not None:
            attn_score = self.act(attn_score)
        return attn_score
