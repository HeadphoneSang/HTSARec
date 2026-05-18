import math

import torch
from torch import nn


class AttentionScoreEncoder(nn.Module):
    """
        AttentionScoreEncoder 用于在霍克斯过程建模中，计算目标项目与历史项目之间的激励系数。

        本模块基于注意力机制，对融合了语义信息和时间间隔特征的输入进行加权评分，输出每个目标项
        与历史交互项之间的注意力权重（激励强度），用于建模推荐系统中事件的条件强度函数。

        特点：
        - 输入包括目标项目嵌入、历史交互嵌入和时间间隔嵌入
        - 利用点积注意力机制计算激励系数
        - 输出形状为 (batch_size, sample_len, seq_len)，表示每个采样目标项对历史每个位置的注意力得分
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
        计算x_tensors(batch,sample_len,hidden_size) 中对应的样本对历史交互的注意力权重
        Args:
            x_tensors: (batch,masked_num,sample_len,hidden_size) 要计算权重的样本初始表示矩阵
            item_seq_mat: (batch,seq_len,hidden_size)  每个交互序列的隐藏表示列表
            time_intervals_mat: (batch,masked_num,seq_len,hidden_size)  与其他交互的时间间隔离散化
            masked_attention: (batch,masked_num,sample_len,seq_len)  用来标记出填充的位置和所有掩码的位置避免看到自己可以看到未来
        Returns:
            返回(batch,masked_num,sample,seq_len)的注意力系数矩阵
        """
        item_seq_mat = item_seq_mat.unsqueeze(1)
        x_query = self.Q(x_tensors)
        x_key = self.K(item_seq_mat)
        time_key = self.t_K(time_intervals_mat)
        total_key = x_key + time_key  #(batch,masked_len,seq_len,hidden_size)融合了交互时间间隔信息和交互的语义信息的key矩阵，
        # 每一行表示当前交互序列的目标交互和当前行表示的交互的包含了间隔和当前行语义的key向量.
        attn_score = (torch.matmul(x_query, total_key.transpose(2, 3)) / self.sqrt_d)
        #(batch,masked_len,sample_len,seq_len)
        #得到的初步的注意力系数
        if masked_attention is not None:
            attn_score = masked_attention + attn_score
        if self.act is not None:
            attn_score = self.act(attn_score)  #对每一个注意力系数进行sigmoid计算，这里不进行softmax因为不是注意力系数，
        #这里计算的是霍克斯过程中历史事件的激励系数，衡量历史事件对当前事件发生的重要程度
        return attn_score
