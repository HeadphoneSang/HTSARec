import math

import torch
from torch import nn


class AttentionScoreEncoder(nn.Module):
    """
        AttentionScoreEncoder computes the inspiration coefficients between target items and historical
        items in Hawkes-process modeling.

        Based on the attention mechanism, this module performs weighted scoring on inputs that fuse
        semantic information and time-interval features, and outputs the attention weights (inspiration
        intensity) between each target item and historical interaction items, which are used to model
        the conditional intensity function of events in a recommender system.

        Features:
        - Inputs include target-item embeddings, historical-interaction embeddings, and time-interval embeddings
        - Uses a dot-product attention mechanism to compute the inspiration coefficients
        - Output shape is (batch_size, sample_len, seq_len), representing the attention score of each sampled
          target item against each historical position
    """

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
        """
        Compute the attention weights of the samples in x_tensors (batch, sample_len, hidden_size) towards
        historical interactions.
        Args:
            x_tensors: (batch, masked_num, sample_len, hidden_size) the initial representation matrix of the
                samples for which weights are computed.
            item_seq_mat: (batch, seq_len, hidden_size) the list of hidden representations for each interaction sequence.
            time_intervals_mat: (batch, masked_num, seq_len, hidden_size) the discretized time intervals to other interactions.
            masked_attention: (batch, masked_num, sample_len, seq_len) a mask marking padding positions and all
                masked positions, so that the model avoids attending to itself or to the future.
        Returns:
            the attention-coefficient matrix of shape (batch, masked_num, sample, seq_len).
        """
        item_seq_mat = item_seq_mat.unsqueeze(1)
        x_query = self.Q(x_tensors)
        x_key = self.K(item_seq_mat)
        time_key = self.t_K(time_intervals_mat)
        total_key = x_key + time_key  # (batch, masked_len, seq_len, hidden_size) key matrix fusing interaction time-interval information and interaction semantic information;
        # each row represents the key vector containing the interval and the current-row semantics for the target interaction versus the interaction represented by that row of the current sequence.
        attn_score = (torch.matmul(x_query, total_key.transpose(2, 3)) / self.sqrt_d)
        # (batch, masked_len, sample_len, seq_len)
        # the preliminary attention coefficients
        if masked_attention is not None:
            attn_score = masked_attention + attn_score
        if self.act is not None:
            attn_score = self.act(attn_score)  # apply sigmoid to each attention coefficient; no softmax is used because these are not attention weights,
        # but the inspiration coefficients of historical events in the Hawkes process, measuring the importance of historical events on the occurrence of the current event
        return attn_score
