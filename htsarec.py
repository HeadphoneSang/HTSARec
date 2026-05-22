import time

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, TimeAwareEncoder
from recbole.model.loss import BPRLoss
from recbole.model.customized_layers import AttentionScoreEncoder
from recbole.utils.operation.augments import HawkesInsertAugmentation, CropAugmentation, NoDataAugmentation, \
    MaskAugmentation, ReOrderAugmentation, InsertAugmentation, TiInsertAugmentation


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
        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)
        self.time_interval_embedding = nn.Embedding(self.bukkit_num + 1, self.hidden_size,
                                                    padding_idx=self.bukkit_num)
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

        # init BERT encoder
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
        )  # transformer for BERT encoder
        self.mask_ratio = config["mask_ratio"]
        self.MASK_ITEM_SEQ = config["MASK_ITEM_SEQ"]
        self.POS_ITEMS = config["POS_ITEMS"]
        self.NEG_ITEMS = config["NEG_ITEMS"]
        self.MASK_INDEX = config["MASK_INDEX"]

        # FFN layer
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.input_dropout_prob)
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

        # FAST EVALUATION
        self.NEG_FIELD = config['eval_args']['neg_field']

        # inspire scores parameters
        self.score_act = config['generator_args']['score_act']
        self.score_act = self.get_act_score(self.score_act)
        inspire_act_name = config['generator_args']['inspire_act']
        self.score_attn_layer = AttentionScoreEncoder(self.hidden_size, inspire_act_name)

        # uniform data_augments parameters
        self.uniform_thr = config['generator_args']['uniform_thr']
        asc_std_slist = dataset.asc_std_slist
        no_aug_num = int(len(asc_std_slist) * self.uniform_thr)
        self.no_aug_seq = set(asc_std_slist[: no_aug_num])
        self.SEQ_ID_LIST = "sid"
        augment_name = config['data_augment']
        data_augment = get_data_augment(augment_name)
        self.data_augment = data_augment(config, self)

    def get_shared_layers(self):
        return self.item_embedding.weight

    def get_act_score(self, act_name):
        """
        Get activation function by name. Return None if not found.
        Args:
            act_name: Name of activation function, case-insensitive

        Returns: Activation function

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
        Calculate discrete values of time intervals
        Args:
            intervals: Tensor of time intervals

        Returns: Discreted results

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
        Convert timestamp sequence of interaction sequence into time interval matrix of user interactions.
        Each row represents time intervals between a certain interaction and interactions at corresponding positions.
        The diagonal contains self-self intervals, and the rest contains intervals with future interactions.
        Args:
            time_seq: Timestamp sequence matrix of interaction sequence (batch, seq_len)

        Returns: Discretized time difference matrix result t_bukkit_mat (batch, seq_len, seq_len)
        Processed upper triangle: replaced discretized results of future interactions in upper triangle
        with impossible time interval discrete value bukkit_num to mark future interactions
        """
        # time_seq (batch,len)
        ori_time = time_seq.unsqueeze(1).expand(-1, time_seq.size(-1), -1)  # (len,len)
        exp_time = time_seq.unsqueeze(-1).expand(-1, -1, time_seq.size(-1))
        time_diff_mat = exp_time - ori_time  # (batch,len,len)
        masked_mat = torch.triu(torch.ones_like(time_diff_mat),
                                diagonal=1).bool()
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
        # [batch,1,len,len]
        time_bukkit_mat = self.calculate_bukkit_mat(time_seq)
        time_bukkit_mat = self.time_interval_embedding(time_bukkit_mat)
        trm_output = self.trm_encoder(
            input_emb, time_bukkit_mat, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        hidden_idx = torch.tensor(item_seq.size()[1], dtype=torch.long, device=item_seq.device).expand(
            item_seq.size()[0])
        origin_output = output
        output = self.gather_indexes(output, hidden_idx - 1)
        return output, origin_output  # [B H]

    def bert_forward(self, item_seq):
        """
        Generate hidden representations of interaction sequence through dual-end encoder based on passed masked interaction sequence
        Args:
            item_seq: Masked interaction sequence (batch, seq_len)

        Returns: Hidden representations of masked sequence (batch, seq_len, hidden_size)

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
        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
        trm_output = self.normal_trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        return output  # [B L H]

    def bpr_masked_seq(self, pos_scores, neg_scores, seq_mask):
        """
        Calculate BPR loss for masked prediction of masked sequence
        Args:
            pos_scores: (batch, masked_len, 1) Positive sample scores, i.e., scores of positive samples at each position of each sequence
            neg_scores: (batch, masked_len, neg_num) Scores of each negative sample at masked positions of each sequence
            seq_mask: (batch, masked_len) Mask matrix of masked sequence, bool, marks which masks are real masks (True), padding is False

        Returns: BPR loss scalar

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
        Calculate dot product results between BERT output hidden representations and other positions in sequence, including position embeddings
        Args:
            masked_ids: Record which positions in masked_seq are masked, index indices (batch, masked_len)
            bert_seq_output: BERT output hidden representation tensor (batch, seq_len, hidden_size)
            masked_item_seq: Masked sequence (batch, seq_len)
            sample_num: Number of samples to calculate scores for each masked position

        Returns: (batch, masked_len, sample_num, seq_len) Similarity degree of each masked position to other interactions in sequence,
        expanded to same shape as incentive coefficients

        """
        masked_ids_expanded = masked_ids.unsqueeze(-1)
        masked_inter_embs = torch.gather(bert_seq_output, dim=1,
                                         index=masked_ids_expanded.expand(-1, -1, bert_seq_output.size(-1)))
        position_ids = torch.arange(
            masked_item_seq.size(1), dtype=torch.long, device=masked_item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(masked_item_seq)
        position_embedding = self.position_embedding(position_ids)
        masked_seq_emb = self.item_embedding(masked_item_seq)
        masked_seq_emb = masked_seq_emb + position_embedding  # (batch_size,seq_len,hidden_size)
        native_scores = torch.matmul(masked_inter_embs, masked_seq_emb.transpose(1, 2))  # (batch,masked_len,seq_len)
        # (batch,masked_len,seq_len)
        native_scores = native_scores.unsqueeze(2).expand(-1, -1, sample_num, -1)
        # (batch,masked_len,sample_num,seq_len)
        if self.score_act is None:
            return native_scores
        native_scores = self.score_act(native_scores)
        return native_scores

    def calculate_generator_loss(self, interaction):
        """
        Calculate loss of uniform insertion generator, optimize generator parameters, return weighted generator loss
        Args:
            interaction: User interaction data

        Returns (torch.Tensor): (1), Scalar, weighted generator loss

        """
        masked_item_seq = interaction[self.MASK_ITEM_SEQ]
        bert_seq_output = self.bert_forward(masked_item_seq)  # (batch,seq_len,hidden_size)
        # O(L^2*d + L*d^2)
        pos_ids = interaction[self.POS_ITEMS].unsqueeze(2)  # (batch_num,masked_num,1)
        neg_ids = interaction[self.NEG_ITEMS]  # (batch_num,masked_num,neg_num)
        all_tar_ids = torch.cat([pos_ids, neg_ids], dim=2)  # (batch_num,masked_num,neg_num+1)
        x_tensors = self.item_embedding(all_tar_ids)  # (batch,masked_num,neg_num+1,hidden_size)
        time_seq = interaction[self.TIME_SEQ]
        masked_ids = interaction[self.MASK_INDEX]
        sample_num = neg_ids.size(-1) + pos_ids.size(-1)
        inspire_scores, seq_mask = self.calculate_all_inspire_score(x_tensors, masked_item_seq, time_seq, masked_ids,
                                                                    pos_ids.squeeze(2), sample_num)
        native_scores = self.calculate_native_scores(masked_ids, bert_seq_output, masked_item_seq, sample_num)
        # (batch,masked_len,sample_num,seq_len)
        hawkes_scores = (native_scores * inspire_scores).sum(dim=-1)  # (batch,masked_len,sample_num)
        hawkes_pos_scores = hawkes_scores[:, :, :1]  # (batch_size，masked_len,1)
        hawkes_neg_scores = hawkes_scores[:, :, 1:]  # (batch_size，masked_len,neg_size)
        uni_loss = 0
        if self.g_loss_type == 'BPR':
            uni_loss = self.bpr_masked_seq(hawkes_pos_scores, hawkes_neg_scores, seq_mask)
        elif self.g_loss_type == 'CE':
            uni_loss = self.bpr_masked_seq(hawkes_pos_scores, hawkes_neg_scores, seq_mask)
        return uni_loss

    def calculate_all_inspire_score(self, x_tensors, masked_item_seq, time_seq, masked_ids, valid_mat, sample_num):
        """
        Calculate incentive coefficients of target items at target masked positions for other interactions in sequence
        Args:
            x_tensors: Hidden representations (excluding position information) of all items to be calculated at each target masked position
            of each interaction sequence (batch, masked_len, sample_num, hidden_size)
            masked_item_seq: Masked sequence collection for mask filling (batch, seq_len)
            time_seq: Timestamp sequence of corresponding masked sequences (batch, seq_len)
            masked_ids: Index sequence of corresponding masked sequences, marking which positions in each sequence are masked,
            with 0 padding at front (batch, masked_len)
            valid_mat: Used to verify which positions in mask index matrix are non-padding positions (batch, masked_len)
            sample_num: Number of samples at each masked position, scalar

        Returns: scores, seq_mask - Incentive coefficients of target items at target masked positions for other interactions,
        mark which positions are real mask positions (batch, masked_len)

        """
        batch_size = masked_ids.size(0)
        masked_len = masked_ids.size(-1)

        # Using broadcasting mechanism, construct tensors of Hawkes coefficients to be predicted for all target
        # masked positions x_tensors = self.item_embedding.weight.view(1, 1, -1, self.hidden_size.size(-1)) Embedding
        # representations of target items to calculate incentive coefficients (1, 1, n_items, hidden_size)
        x_position = masked_ids  # (batch_num,masked_num)
        x_position_embedding = self.position_embedding(x_position).unsqueeze(2)
        # (batch_num,masked_num,n_items,hidden_size)
        x_tensors = x_tensors + x_position_embedding

        masked_seq_emb = self.item_embedding(masked_item_seq)
        position_ids = torch.arange(
            self.max_seq_length, dtype=torch.long, device=masked_item_seq.device
        )
        position_ids = position_ids.unsqueeze(0)
        position_embs = self.position_embedding(position_ids)  #(1,seq_len,hidden_size)
        masked_seq_emb = masked_seq_emb + position_embs
        masked_ids = x_position
        masked_time_seq = torch.gather(time_seq, index=masked_ids, dim=1)  # (batch,masked_len)
        masked_time_seq = masked_time_seq.unsqueeze(2).expand(-1, -1, time_seq.size(-1))  # (batch,masked_len,seq_len)
        time_seq = time_seq.unsqueeze(1)  # (batch,1,seq_len)
        time_intervals_mat = torch.abs(masked_time_seq - time_seq)
        # (batch,masked_len,seq_len)
        time_intervals_mat = self.calculate_bukkit(time_intervals_mat)
        time_intervals_embs = self.time_interval_embedding(time_intervals_mat)
        # (batch,masked_len,seq_len,hidden_size)
        seq_mask = valid_mat != 0  # (batch_size,masked_len)
        score_mask = torch.zeros(batch_size, self.max_seq_length, dtype=torch.float64,
                                 device=self.device)
        row_ids = torch.arange(batch_size, dtype=torch.long, device=self.device).unsqueeze(1).expand(-1, masked_len)[
            seq_mask]
        col_ids = masked_ids[seq_mask]
        score_mask[row_ids, col_ids] = float('-inf')
        padding_mask = masked_item_seq == 0
        score_mask[padding_mask] = float('-inf')
        score_mask = score_mask.unsqueeze(1).unsqueeze(1).expand(-1, masked_len, sample_num, -1)
        # (batch,masked_len,sample_num,seq_len)
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
        raw_item_seq = interaction[self.ITEM_SEQ]
        raw_time_seq = interaction[self.TIME_SEQ]
        new_item_seq, new_time_seq = self.data_augment(interaction)
        aug_seq_output, origin_aug_seq = self.forward(new_item_seq, new_time_seq)
        raw_seq_output, origin_raw_seq = self.forward(raw_item_seq, raw_time_seq)
        pos_items = interaction[self.POS_ITEM_ID]
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
            all_logits = torch.matmul(all_outputs, test_item_emb.transpose(0, 1))  # (batch,item_num)
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
        item_seq = interaction[self.ITEM_SEQ]
        time_seq = interaction[self.TIME_SEQ]
        test_item = interaction[self.ITEM_ID]
        seq_output, _ = self.forward(item_seq, time_seq)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def fast_predict(self, interaction):
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
        scores = torch.cat([pos_scores, neg_scores], dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        time_seq = interaction[self.TIME_SEQ]
        seq_output, _ = self.forward(item_seq, time_seq)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
