import time

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, TimeAwareEncoder
from recbole.model.loss import BPRLoss
from recbole.model.customized_layers import AttentionScoreEncoder
from recbole.utils.operation.augments import HawkesInsertAugmentation, CropAugmentation, NoDataAugmentation, \
    MaskAugmentation, ReOrderAugmentation, InsertAugmentation, TiInsertAugmentation
from recbole.utils.utils import plot_diff_heatmap, plot_heatmap, plot_pca_pointcloud
from recbole.utils.utils import plot_two_heatmaps_with_diff


def get_data_augment(augment_name):
    augment_dict = {
        'hawkes_insert': HawkesInsertAugmentation,
        "ti_insert": TiInsertAugmentation,
        "insert": InsertAugmentation,
        "reorder": ReOrderAugmentation,
        "crop": CropAugmentation,
        "mask": MaskAugmentation,
        "none": NoDataAugmentation,
    }
    return augment_dict.get(augment_name.lower())


class AHRec(SequentialRecommender):

    def __init__(self, config, dataset):
        super(AHRec, self).__init__(config, dataset)
        self.max_time_gap = dataset.max_time_gap
        self.min_time_gap = dataset.min_time_gap
        self.eps = 1e-12
        self.base = config['base']
        self.TIME_SEQ = "timestamp_list"
        self.TAR_TIME = "timestamp"
        self.bukkit_num = int(self.calculate_bukkit(self.max_time_gap).item())
        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.input_dropout_prob = config["input_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)  #后面加一位是为了生成预测交互的位置向量
        self.time_interval_embedding = nn.Embedding(self.bukkit_num + 1, self.hidden_size,
                                                    padding_idx=self.bukkit_num)  # 多出来一个用来放未来交互的填充位
        self.trm_encoder = TimeAwareEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        #初始化双端编码器
        self.g_n_layers = config["generator_args"]["n_layers"]
        self.g_n_heads = config["generator_args"]["n_heads"]
        self.g_inner_size = config["generator_args"]["inner_size"]
        self.g_hidden_dropout_prob = config["generator_args"]["hidden_dropout_prob"]
        self.g_attn_dropout_prob = config["generator_args"]["attn_dropout_prob"]
        self.g_hidden_act = config["generator_args"]["hidden_act"]
        self.g_layer_norm_eps = config["generator_args"]["layer_norm_eps"]
        self.g_loss_type = config["generator_args"]["loss_type"]
        if self.g_loss_type == "BPR":
            self.g_loss_fct = self.bpr_masked_seq
        elif self.g_loss_type == "CE":
            self.g_loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'g-loss_type' in ['BPR', 'CE']!")
        self.normal_trm_encoder = TransformerEncoder(
            n_layers=self.g_n_layers,
            n_heads=self.g_n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.g_inner_size,
            hidden_dropout_prob=self.g_hidden_dropout_prob,
            attn_dropout_prob=self.g_attn_dropout_prob,
            hidden_act=self.g_hidden_act,
            layer_norm_eps=self.g_layer_norm_eps,
        )  #用于双端编码的transformer
        self.mask_ratio = config["mask_ratio"]
        self.MASK_ITEM_SEQ = config["MASK_ITEM_SEQ"]
        self.POS_ITEMS = config["POS_ITEMS"]
        self.NEG_ITEMS = config["NEG_ITEMS"]
        self.MASK_INDEX = config["MASK_INDEX"]

        #用于对输入编码层的原始交互嵌入进行处理的层
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.input_dropout_prob)

        #确认损失函数类型
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

        #用于快速负采样评估
        self.NEG_FIELD = config['eval_args']['neg_field']

        #用于计算激励系数
        self.score_act = config['generator_args']['score_act']
        self.score_act = self.get_act_score(self.score_act)
        inspire_act_name = config['generator_args']['inspire_act']
        self.score_attn_layer = AttentionScoreEncoder(self.hidden_size, inspire_act_name)

        #均匀化选择性数据增强的参数:
        self.uniform_thr = config['generator_args']['uniform_thr']
        asc_std_slist = dataset.asc_std_slist
        no_aug_num = int(len(asc_std_slist) * self.uniform_thr)
        self.no_aug_seq = set(asc_std_slist[: no_aug_num])  #用于记录哪些序列被认为是均匀的，不需要数据增强的
        self.SEQ_ID_LIST = "sid"

        #初始化数据增强操作
        augment_name = config['data_augment']
        data_augment = get_data_augment(augment_name)
        self.data_augment = data_augment(config, self)

    def get_shared_layers(self):
        return self.item_embedding.weight

    def get_act_score(self, act_name):
        """
        获得对应名字的激活函数，如果没有，报错，None返回NOne
        Args:
            act_name: 激活函数的名称，不区分大小写

        Returns: 返回激活函数

        """
        act_name = act_name.lower()
        act_dict = {
            'softplus': nn.Softplus(),
            'sigmoid': nn.Sigmoid()
        }
        try:
            act_fn = act_dict[act_name]
        except:
            return None
        return act_fn

    def calculate_bukkit(self, intervals):
        """
        计算时间间隔的离散值
        Args:
            intervals: 时间间隔的张量

        Returns: 离散后的结果

        """
        return torch.floor(
            torch.log((intervals / self.min_time_gap + self.eps) + 1) / torch.log(torch.tensor(self.base))).long()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def calculate_bukkit_mat(self, time_seq):
        """
        将交互序列的时间戳序列，转化为用户交互的时间间隔矩阵，每一行代表某个交互和其他对应位置交互的时间间隔，对角线上是自己和自己的时间间隔，后面是和未来的时间间隔
        Args:
            time_seq:  交互序列对应的时间戳序列矩阵 (batch,seq_len)

        Returns: 返回时间差分矩阵经过离散化的结果矩阵t_bukkit_mat (batch,seq_len,seq_len)
        进行了上三角处理，将上三角的未来交互的离散结果替换为了不可能存在的时间间隔离散值bukkit_num,用来标记未来交互
        """
        # time_seq (batch,len)
        ori_time = time_seq.unsqueeze(1).expand(-1, time_seq.size(-1), -1)  # 原始的时间矩阵，(len,len) 每一行都是一样的，都为len次交互的时间戳
        exp_time = time_seq.unsqueeze(-1).expand(-1, -1, time_seq.size(-1))
        time_diff_mat = exp_time - ori_time  # (batch,len,len)
        masked_mat = torch.triu(torch.ones_like(time_diff_mat),
                                diagonal=1).bool()  #上三角矩阵，生成一个和时间差值矩阵对应的掩码矩阵，将对角线以上的部分设为True，从而标记处每个交互和他未来交互的差值
        t_bukkit_mat = self.calculate_bukkit(time_diff_mat)
        t_bukkit_mat = t_bukkit_mat.masked_fill(masked_mat, self.bukkit_num)
        return t_bukkit_mat

    def forward(self, item_seq, time_seq):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)
        #这里的mask[batch,1,len,len]应该是为了多头注意力准备的多一个维度，如果有h个头，那么就有h个掩码矩阵，就将当前矩阵广播到[batch,h,len,len]，然后与注意力系数矩阵相加
        time_bukkit_mat = self.calculate_bukkit_mat(time_seq)
        time_bukkit_mat = self.time_interval_embedding(time_bukkit_mat)
        # time_bukkit_mat = self.LayerNorm(time_bukkit_mat)
        # time_bukkit_mat = self.dropout(time_bukkit_mat)
        trm_output = self.trm_encoder(
            input_emb, time_bukkit_mat, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        hidden_idx = torch.tensor(item_seq.size()[1], dtype=torch.long, device=item_seq.device).expand(
            item_seq.size()[0])
        origin_output = output
        output = self.gather_indexes(output, hidden_idx - 1)  #取得序列的最后一位的隐藏状态作为预测的隐藏状态
        return output, origin_output  # [B H]

    def bert_forward(self, item_seq):
        """
        根据传入的掩码交互序列，通过双端编码器生成交互序列的隐藏表示
        Args:
            item_seq:  掩码后的交互序列 (batch,seq_len)

        Returns:  返回掩码序列的隐藏表示 (batch,seq_len,hidden_size)

        """
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)  # 不同的地方就是掩码是双向掩码
        trm_output = self.normal_trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )  # 普通的transformer编码
        output = trm_output[-1]
        return output  # [B L H]

    def bpr_masked_seq(self, pos_scores, neg_scores, seq_mask):
        """
        为掩码序列计算掩码预测的bpr损失计算函数
        Args:
            pos_scores: (batch,masked_len,1)正样本得分，就是每个序列的每个位置的正样本的得分
            neg_scores: (batch,masked_len,neg_num)每个序列掩码位置的每个负样本的得分
            seq_mask: (batch,masked_len)掩码序列的掩码矩阵，bool，标记哪个掩码是真实的掩码，标记为true，padding为false

        Returns: 返回bpr损失标量

        """
        neg_num = neg_scores.size(2)
        pos_scores = pos_scores.expand(-1, -1, neg_num)
        loss_per = -torch.log(1e-14 + torch.sigmoid(pos_scores - neg_scores))
        seq_mask = seq_mask.unsqueeze(-1).float()
        loss_per = loss_per * seq_mask
        loss = loss_per.sum() / (seq_mask.sum() * neg_num + 1e-14)
        return loss

    def calculate_native_scores(self, masked_ids, bert_seq_output, masked_item_seq, sample_num):
        """
        计算bert的输出隐藏吧表示和序列里面其他位置的点积运算结果，包含了位置嵌入
        Args:
            masked_ids: 记录masked_seq里面哪些位置被掩码了，下标索引(batch,masked_len)
            bert_seq_output: bert输出的隐藏表示张量(batch,seq_len,hidden_Size)
            masked_item_seq: 被掩码的序列(batch,seq_len)
            sample_num: 每个掩码位置的待计算得分的样本的数目

        Returns: (batch,masked_len,sample_num,seq_len)每个掩码位置对序列中其他交互的相似程度，扩展成了和激励系数相同形状

        """
        masked_ids_expanded = masked_ids.unsqueeze(-1)
        # col_ids = masked_ids[seq_mask]  #(tuple_len) 真实掩码位置的列索引列表
        masked_inter_embs = torch.gather(bert_seq_output, dim=1,
                                         index=masked_ids_expanded.expand(-1, -1, bert_seq_output.size(-1)))
        # 取出来了掩码位置的bert隐藏表示(batch_size,masked_len,hidden_size)前面可能包含了padding的0位置对应的隐藏表示
        # 0padding的隐藏表示不影响，后面计算loss的时候会用掩码过滤掉
        position_ids = torch.arange(
            masked_item_seq.size(1), dtype=torch.long, device=masked_item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(masked_item_seq)
        position_embedding = self.position_embedding(position_ids)
        masked_seq_emb = self.item_embedding(masked_item_seq)
        masked_seq_emb = masked_seq_emb + position_embedding  # (batch_size,seq_len,hidden_size)
        native_scores = torch.matmul(masked_inter_embs, masked_seq_emb.transpose(1, 2))  # (batch,masked_len,seq_len)
        # (batch,masked_len,seq_len)得到朴素的每个掩码位置对序列交互的兴趣打分
        native_scores = native_scores.unsqueeze(2).expand(-1, -1, sample_num, -1)
        # (batch,masked_len,sample_num,seq_len)
        if self.score_act is None:
            return native_scores
        native_scores = self.score_act(native_scores)
        return native_scores

    def calculate_generator_loss(self, interaction):
        """
         计算均匀化插入生成器的损失，优化生成器的参数，返回加权后的生成器损失
        Args:
            interaction: 用户的交互数据

        Returns (torch.Tensor): (1,)标量，加权后生成器损失

        """
        masked_item_seq = interaction[self.MASK_ITEM_SEQ]
        bert_seq_output = self.bert_forward(masked_item_seq)  # 返回的是掩码序列的隐藏更表示(batch,seq_len,hidden_size)
        #bert就是一个普通的注意力，时间复杂度为O(L^2*d + L*d^2)
        pos_ids = interaction[self.POS_ITEMS].unsqueeze(2)  # (batch_num,masked_num,1)
        neg_ids = interaction[self.NEG_ITEMS]  # (batch_num,masked_num,neg_num)
        all_tar_ids = torch.cat([pos_ids, neg_ids], dim=2)  # (batch_num,masked_num,neg_num+1)
        x_tensors = self.item_embedding(all_tar_ids)  # 要计算激励系数的目标项目的嵌入表示(batch,masked_num,neg_num+1,hidden_size)
        time_seq = interaction[self.TIME_SEQ]
        masked_ids = interaction[self.MASK_INDEX]
        sample_num = neg_ids.size(-1) + pos_ids.size(-1)
        #上面的数据转换操作认为时间复杂度为O(1)
        inspire_scores, seq_mask = self.calculate_all_inspire_score(x_tensors, masked_item_seq, time_seq, masked_ids,
                                                                    pos_ids.squeeze(2), sample_num)
        # 返回的是每个序列每个掩码位置正负样本对序列中其他交互的激励系数矩阵(batch,masked_len,sample_num,seq_len)
        # 还有掩码序列中那些位置是真实的掩码位置，哪些是padding(batch,masked_len)

        """
        通过掩码序列的隐藏表示和霍克斯过程对掩码位置预测得分
        """
        native_scores = self.calculate_native_scores(masked_ids, bert_seq_output, masked_item_seq, sample_num)
        # (batch,masked_len,sample_num,seq_len)
        hawkes_scores = (native_scores * inspire_scores).sum(dim=-1)  # 逐点相乘，求和，得到(batch,masked_len,sample_num)
        hawkes_pos_scores = hawkes_scores[:, :, :1]  # 所有序列的所有掩码位置的正样本的得分，形(batch_size，masked_len,1)
        hawkes_neg_scores = hawkes_scores[:, :, 1:]  # 所有序列的所有掩码位置的的负样本们的得分形(batch_size，masked_len,neg_size)
        """
        通过得分计算生成器的损失
        """
        uni_loss = 0
        if self.g_loss_type == 'BPR':
            uni_loss = self.bpr_masked_seq(hawkes_pos_scores, hawkes_neg_scores, seq_mask)
        elif self.g_loss_type == 'CE':
            uni_loss = self.bpr_masked_seq(hawkes_pos_scores, hawkes_neg_scores, seq_mask)
        return uni_loss

    def calculate_all_inspire_score(self, x_tensors, masked_item_seq, time_seq, masked_ids, valid_mat, sample_num):
        """
        计算目标项目在目标掩码位置对序列里面其他交互的激励系数
        Args:
            x_tensors: 要计算的每个交互序列的每个目标掩码位置的所有待计算项目的隐藏表示(不包含位置信息)(batch,masked_len,sample_num,hidden_size)
            masked_item_seq: 要进行掩码填充的掩码序列集合(batch,seq_len)
            time_seq: 对应掩码序列的掩码序列的时间序列(batch,seq_len)
            masked_ids: 对应掩码序列的索引序列，标记每个序列里面哪些位置被掩码了，前面是0填充(batch,masked_len)
            valid_mat: 用来验证掩码索引矩阵哪些位置是非填充位(batch,masked_len)
            sample_num: 每个掩码位置有几个采样，标量

        Returns: scores, seq_mask 目标掩码的目标项目对其他交互的激励系数、标记哪些位置是真实的掩码位置(batch,masked_len)

        """
        batch_size = masked_ids.size(0)
        masked_len = masked_ids.size(-1)

        # 利用广播机制，构造了所有目标掩码位置要进行预测霍克斯系数的张量 x_tensors = self.item_embedding.weight.view(1, 1, -1,
        # self.hidden_size.size(-1))  # 要计算激励系数的目标项目的嵌入表示(1,1,n_items,hidden_size)
        x_position = masked_ids  # (batch_num,masked_num)
        x_position_embedding = self.position_embedding(x_position).unsqueeze(2)
        # (batch_num,masked_num,n_items,hidden_size)
        x_tensors = x_tensors + x_position_embedding  # 加位置嵌入

        # 计算掩码序列的隐藏表示
        masked_seq_emb = self.item_embedding(masked_item_seq)
        position_ids = torch.arange(
            self.max_seq_length, dtype=torch.long, device=masked_item_seq.device
        )
        position_ids = position_ids.unsqueeze(0)
        position_embs = self.position_embedding(position_ids)  #(1,seq_len,hidden_size)
        masked_seq_emb = masked_seq_emb + position_embs  # 掩码序列的隐藏表示

        # 计算每个掩码位置和其他位置的时间间隔
        masked_ids = x_position
        masked_time_seq = torch.gather(time_seq, index=masked_ids, dim=1)  # (batch,masked_len)掩码位置的时间戳
        masked_time_seq = masked_time_seq.unsqueeze(2).expand(-1, -1, time_seq.size(-1))  # (batch,masked_len,seq_len)
        time_seq = time_seq.unsqueeze(1)  # (batch,1,seq_len)
        time_intervals_mat = torch.abs(masked_time_seq - time_seq)  # 时间间隔矩阵，每一行表示每个掩码位置和其他交互的时间间隔
        # (batch,masked_len,seq_len) 这里加了绝对值，是想要双向霍克斯建模
        time_intervals_mat = self.calculate_bukkit(time_intervals_mat)
        time_intervals_embs = self.time_interval_embedding(time_intervals_mat)
        # 时间间隔矩阵(batch,masked_len,seq_len,hidden_size)

        # 获取每个masked_seq的mask掩码矩阵，用于计算注意力系数

        seq_mask = valid_mat != 0  # 找到掩码序列中，真实的掩码索引 (batch_size,masked_len)
        score_mask = torch.zeros(batch_size, self.max_seq_length, dtype=torch.float64,
                                 device=self.device)  # 用来标记每个序列哪些位置被掩码
        row_ids = torch.arange(batch_size, dtype=torch.long, device=self.device).unsqueeze(1).expand(-1, masked_len)[
            seq_mask]
        col_ids = masked_ids[seq_mask]
        score_mask[row_ids, col_ids] = float('-inf')  # 将掩码索引指定的位置的掩码设为负无穷
        padding_mask = masked_item_seq == 0
        score_mask[padding_mask] = float('-inf')  # 将padding位置也设为-inf
        score_mask = score_mask.unsqueeze(1).unsqueeze(1).expand(-1, masked_len, sample_num, -1)
        # (batch,masked_len,sample_num,seq_len)最后得到的激励系数的掩码

        # 送入注意力系数层计算激励系数，记得LN和DP
        x_tensors = self.LayerNorm(x_tensors)
        x_tensors = self.dropout(x_tensors)
        masked_seq_emb = self.LayerNorm(masked_seq_emb)
        masked_seq_emb = self.dropout(masked_seq_emb)
        scores = self.score_attn_layer(x_tensors, masked_seq_emb, time_intervals_embs, score_mask)
        return scores, seq_mask

    def calculate_loss(self, interaction):
        uni_loss = None
        if isinstance(self.data_augment, HawkesInsertAugmentation):
            uni_loss = self.calculate_generator_loss(interaction)
        """
        对原始序列进行随机插入。先插入插入占位符n_items
        """
        raw_item_seq = interaction[self.ITEM_SEQ]
        raw_time_seq = interaction[self.TIME_SEQ]
        new_item_seq, new_time_seq = self.data_augment(interaction)
        """
        将增强后的用户交互序列送进时间感知注意力层
        """
        aug_seq_output, origin_aug_seq = self.forward(new_item_seq, new_time_seq)
        raw_seq_output, origin_raw_seq = self.forward(raw_item_seq, raw_time_seq)
        check_padding_mask = raw_item_seq != 0
        check_padding_mask = check_padding_mask.all(dim=1)  #(batch)
        batch_size = raw_item_seq.size(0)
        no_padding_ids = torch.arange(0, batch_size, device="cuda")[check_padding_mask]
        select_id = 4  #4
        plot_heatmap(origin_raw_seq, no_padding_ids, select_id, save_path="raw_seq_heatmap.pdf")
        plot_heatmap(origin_aug_seq, no_padding_ids, select_id, save_path="aug_seq_heatmap.pdf")
        plot_diff_heatmap(origin_raw_seq, origin_aug_seq, no_padding_ids, select_id, save_path="diff_heatmap.pdf")
        plot_pca_pointcloud(origin_raw_seq, origin_aug_seq, no_padding_ids, select_id, save_path="same_heatmap.pdf")
        pos_items = interaction[self.POS_ITEM_ID]

        """
        下面是计算最终预测的损失，通过时间感知注意力生成的时间步隐藏表示
        """
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(aug_seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(aug_seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            all_outputs = torch.cat([raw_seq_output, aug_seq_output], dim=0)
            all_logits = torch.matmul(all_outputs, test_item_emb.transpose(0, 1))  #(batch,item_num)
            batch_size = raw_seq_output.size(0)
            raw_logits = all_logits[:batch_size, :]
            aug_logits = all_logits[batch_size:, :]
            raw_loss = self.loss_fct(raw_logits, pos_items)
            aug_loss = self.loss_fct(aug_logits, pos_items)
            loss = raw_loss + aug_loss
        if uni_loss is not None:
            return loss, uni_loss
        else:
            return loss

    def predict(self, interaction):
        """
        为了泛用性，recbole在评估和测试阶段，都会调用此方法去对一个用户序列的正样本项目和负样本集合进行打分
        因为这里是对每一个序列的一个目标样本进行评分，所以为了给一个序列的所有正负样本同时打分，他对一个序列的
        正样本和负样本都单独构建了序列。增加了成倍的计算开销，原本一个序列计算正负样本的得分，只需要一次forward
        现在假如进行100的负采样，就要额外进行100次forward的计算。
        这里不需要担心dropout等随机操作影响forward的输出。因为该方法只会在评估和测试阶段调用。在评估和测试
        之前，trainer会将model设为评估模式，那么模型下面的dropout层等随即操作层都会失效。从而固定了forward的
        结果输出。所以为正负样本构建多个交互序列不会影响最后的得分的计算
        Args:
            interaction: 为多个用户的交互序列构建目标项分别为正样本和单个负样本的用户交互，每个负样本一个inter

        Returns (torch.Tensor): 返回(batch，)每个交互序列样本的得分

        """
        item_seq = interaction[self.ITEM_SEQ]
        time_seq = interaction[self.TIME_SEQ]
        test_item = interaction[self.ITEM_ID]
        seq_output, _ = self.forward(item_seq, time_seq)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def fast_predict(self, interaction):
        """

        Args:
            interaction: (batch_size,) 一个batch大小的交互数据，包含了每个序列的负采样序列

        Returns:
            返回一个(batch_size,(neg_num+1))

        """
        item_seq = interaction[self.ITEM_SEQ]
        time_seq = interaction[self.TIME_SEQ]
        pos_item_ids = interaction[self.ITEM_ID]  #(batch,)
        neg_item_ids = interaction[self.NEG_FIELD]  #(batch_size,neg_num)
        seq_output, _ = self.forward(item_seq, time_seq)  #(batch,hidden_size)
        pos_item_embeds = self.item_embedding(pos_item_ids)  #(batch_size,hidden_size)
        neg_item_embeds = self.item_embedding(neg_item_ids)  #(batch_size,neg_num,hidden_size)
        pos_scores = torch.mul(seq_output, pos_item_embeds).sum(dim=1)
        neg_output = seq_output.unsqueeze(1).transpose(1, 2)  #(batch_size,hidden_size,1)
        neg_scores = torch.matmul(neg_item_embeds, neg_output).squeeze(2)  #(batch,neg_num)
        pos_scores = pos_scores.unsqueeze(1)  #(batch_size,1)
        scores = torch.cat([pos_scores, neg_scores], dim=1)  #将正负样本的得分按列组合(batch,(neg_num+1))
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        time_seq = interaction[self.TIME_SEQ]
        seq_output, _ = self.forward(item_seq, time_seq)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
