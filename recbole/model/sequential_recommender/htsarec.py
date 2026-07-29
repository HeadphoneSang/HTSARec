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


class HTSARec(SequentialRecommender):

    def __init__(self, config, dataset):
        super(HTSARec, self).__init__(config, dataset)
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
        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)  # the extra position is used to generate the position embedding for the predicted interaction
        self.time_interval_embedding = nn.Embedding(self.bukkit_num + 1, self.hidden_size,
                                                    padding_idx=self.bukkit_num)  # an extra bin reserved as the padding slot for future interactions
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

        # Initialize the bidirectional (BERT-style) encoder
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
        )  # transformer used for bidirectional encoding
        self.mask_ratio = config["mask_ratio"]
        self.MASK_ITEM_SEQ = config["MASK_ITEM_SEQ"]
        self.POS_ITEMS = config["POS_ITEMS"]
        self.NEG_ITEMS = config["NEG_ITEMS"]
        self.MASK_INDEX = config["MASK_INDEX"]

        # Layers for processing the raw interaction embeddings fed into the encoder
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.input_dropout_prob)

        # Determine the loss function type
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

        # Used for fast negative-sampling evaluation
        self.NEG_FIELD = config['eval_args']['neg_field']

        # Used to compute the inspiration (excitation) scores
        self.score_act = config['generator_args']['score_act']
        self.score_act = self.get_act_score(self.score_act)
        inspire_act_name = config['generator_args']['inspire_act']
        self.score_attn_layer = AttentionScoreEncoder(self.hidden_size, inspire_act_name)

        # Parameters for uniformity-aware selective data augmentation:
        self.uniform_thr = config['generator_args']['uniform_thr']
        asc_std_slist = dataset.asc_std_slist
        no_aug_num = int(len(asc_std_slist) * self.uniform_thr)
        self.no_aug_seq = set(asc_std_slist[: no_aug_num])  # records which sequences are considered uniform and thus skipped for augmentation
        self.SEQ_ID_LIST = "sid"

        # Initialize the data augmentation operation
        augment_name = config['data_augment']
        data_augment = get_data_augment(augment_name)
        self.data_augment = data_augment(config, self)

    def get_shared_layers(self):
        return self.item_embedding.weight

    def get_act_score(self, act_name):
        """
        Return the activation function corresponding to the given name; return None if the name is not found.
        Args:
            act_name: the name of the activation function (case-insensitive).

        Returns: the activation function.

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
        Discretize the time intervals.
        Args:
            intervals: the tensor of time intervals.

        Returns: the discretized result.

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
        Convert the timestamp sequence of an interaction sequence into a time-interval matrix. Each row represents
        the time intervals between a given interaction and the interactions at the other positions; the diagonal holds
        the interval of an interaction with itself, and the entries after it are the intervals with future interactions.
        Args:
            time_seq: the timestamp sequence corresponding to the interaction sequence, shape (batch, seq_len).

        Returns: the discretized time-difference matrix t_bukkit_mat, shape (batch, seq_len, seq_len).
        The upper triangle is processed so that the discretized intervals of future interactions are replaced with
        bukkit_num (an impossible bin value), which marks future interactions.
        """
        # time_seq (batch, len)
        ori_time = time_seq.unsqueeze(1).expand(-1, time_seq.size(-1), -1)  # the original time matrix (len, len); every row is identical, holding the timestamps of the len interactions
        exp_time = time_seq.unsqueeze(-1).expand(-1, -1, time_seq.size(-1))
        time_diff_mat = exp_time - ori_time  # (batch, len, len)
        masked_mat = torch.triu(torch.ones_like(time_diff_mat),
                                diagonal=1).bool()  # upper-triangular mask matching the time-difference matrix; sets the entries above the diagonal to True, marking each interaction's difference with its future interactions
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
        # The mask [batch, 1, len, len] adds an extra dimension for multi-head attention: with h heads there are h mask matrices, so this matrix is broadcast to [batch, h, len, len] and added to the attention score matrix.
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
        output = self.gather_indexes(output, hidden_idx - 1)  # take the hidden state of the last position as the prediction hidden state
        return output, origin_output  # [B H]

    def bert_forward(self, item_seq):
        """
        Generate the hidden representation of the interaction sequence through the bidirectional encoder,
        given the masked interaction sequence.
        Args:
            item_seq: the masked interaction sequence, shape (batch, seq_len).

        Returns: the hidden representation of the masked sequence, shape (batch, seq_len, hidden_size).

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
        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)  # the difference is that the mask here is bidirectional
        trm_output = self.normal_trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )  # standard (non-time-aware) transformer encoding
        output = trm_output[-1]
        return output  # [B L H]

    def bpr_masked_seq(self, pos_scores, neg_scores, seq_mask):
        """
        Compute the BPR loss for masked-sequence prediction.
        Args:
            pos_scores: (batch, masked_len, 1) the positive-item score at each masked position of each sequence.
            neg_scores: (batch, masked_len, neg_num) the score of each negative sample at each masked position.
            seq_mask: (batch, masked_len) the boolean mask of the masked sequence, marking real masks as True and padding as False.

        Returns: the scalar BPR loss.

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
        Compute the dot-product scores between the BERT output hidden representation and the embeddings at the
        other positions of the sequence (position embeddings included).
        Args:
            masked_ids: the indices recording which positions of masked_seq are masked, shape (batch, masked_len).
            bert_seq_output: the hidden representation output by BERT, shape (batch, seq_len, hidden_size).
            masked_item_seq: the masked sequence, shape (batch, seq_len).
            sample_num: the number of candidate items scored at each masked position.

        Returns: (batch, masked_len, sample_num, seq_len) the similarity of each masked position to the other
        interactions in the sequence, expanded to the same shape as the inspiration scores.

        """
        masked_ids_expanded = masked_ids.unsqueeze(-1)
        # col_ids = masked_ids[seq_mask]  # (tuple_len) the column-index list of the real masked positions
        masked_inter_embs = torch.gather(bert_seq_output, dim=1,
                                         index=masked_ids_expanded.expand(-1, -1, bert_seq_output.size(-1)))
        # The BERT hidden representations at the masked positions (batch_size, masked_len, hidden_size); the leading entries may include the hidden representations of padding (0) positions.
        # The hidden representations of 0-padding do not matter; they are filtered out by the mask when computing the loss later.
        position_ids = torch.arange(
            masked_item_seq.size(1), dtype=torch.long, device=masked_item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(masked_item_seq)
        position_embedding = self.position_embedding(position_ids)
        masked_seq_emb = self.item_embedding(masked_item_seq)
        masked_seq_emb = masked_seq_emb + position_embedding  # (batch_size,seq_len,hidden_size)
        native_scores = torch.matmul(masked_inter_embs, masked_seq_emb.transpose(1, 2))  # (batch, masked_len, seq_len)
        # (batch, masked_len, seq_len) the native interest score of each masked position towards the sequence interactions.
        native_scores = native_scores.unsqueeze(2).expand(-1, -1, sample_num, -1)
        # (batch,masked_len,sample_num,seq_len)
        if self.score_act is None:
            return native_scores
        native_scores = self.score_act(native_scores)
        return native_scores

    def calculate_generator_loss(self, interaction):
        """
         Compute the loss of the uniformity-aware insertion generator, optimize the generator's parameters,
        and return the weighted generator loss.
        Args:
            interaction: the user interaction data.

        Returns (torch.Tensor): a scalar tensor of shape (1,), the weighted generator loss.

        """
        masked_item_seq = interaction[self.MASK_ITEM_SEQ]
        bert_seq_output = self.bert_forward(masked_item_seq)  # the hidden representation of the masked sequence (batch, seq_len, hidden_size)
        # BERT is just standard attention, with time complexity O(L^2*d + L*d^2).
        pos_ids = interaction[self.POS_ITEMS].unsqueeze(2)  # (batch_num, masked_num, 1)
        neg_ids = interaction[self.NEG_ITEMS]  # (batch_num, masked_num, neg_num)
        all_tar_ids = torch.cat([pos_ids, neg_ids], dim=2)  # (batch_num, masked_num, neg_num+1)
        x_tensors = self.item_embedding(all_tar_ids)  # the embeddings of the target items for which inspiration scores are computed (batch, masked_num, neg_num+1, hidden_size)
        time_seq = interaction[self.TIME_SEQ]
        masked_ids = interaction[self.MASK_INDEX]
        sample_num = neg_ids.size(-1) + pos_ids.size(-1)
        # The data-transform operations above are considered O(1) in time complexity.
        inspire_scores, seq_mask = self.calculate_all_inspire_score(x_tensors, masked_item_seq, time_seq, masked_ids,
                                                                    pos_ids.squeeze(2), sample_num)
        # Returns the inspiration-score matrix of each positive/negative sample at each masked position of each
        # sequence towards the other interactions (batch, masked_len, sample_num, seq_len),
        # and which positions of the masked sequence are real masks vs. padding (batch, masked_len).

        """
        Predict the scores at the masked positions using the hidden representation of the masked sequence and the Hawkes process.
        """
        native_scores = self.calculate_native_scores(masked_ids, bert_seq_output, masked_item_seq, sample_num)
        # (batch, masked_len, sample_num, seq_len)
        hawkes_scores = (native_scores * inspire_scores).sum(dim=-1)  # element-wise multiply then sum, yielding (batch, masked_len, sample_num)
        hawkes_pos_scores = hawkes_scores[:, :, :1]  # the positive-item scores at all masked positions of all sequences, shape (batch_size, masked_len, 1)
        hawkes_neg_scores = hawkes_scores[:, :, 1:]  # the negative-sample scores at all masked positions of all sequences, shape (batch_size, masked_len, neg_size)
        """
        Compute the generator loss from the scores.
        """
        uni_loss = 0
        if self.g_loss_type == 'BPR':
            uni_loss = self.bpr_masked_seq(hawkes_pos_scores, hawkes_neg_scores, seq_mask)
        elif self.g_loss_type == 'CE':
            uni_loss = self.bpr_masked_seq(hawkes_pos_scores, hawkes_neg_scores, seq_mask)
        return uni_loss

    def calculate_all_inspire_score(self, x_tensors, masked_item_seq, time_seq, masked_ids, valid_mat, sample_num):
        """
        Compute the inspiration scores of the target items at the target masked positions towards the other
        interactions in the sequence.
        Args:
            x_tensors: the hidden representations (without position info) of all candidate items at each target
            masked position of each sequence, shape (batch, masked_len, sample_num, hidden_size).
            masked_item_seq: the set of masked sequences to be mask-filled, shape (batch, seq_len).
            time_seq: the timestamp sequence corresponding to the masked sequence, shape (batch, seq_len).
            masked_ids: the index sequence of the masked sequence, marking which positions are masked in each
            sequence; leading entries are 0-padding, shape (batch, masked_len).
            valid_mat: used to verify which positions of the masked-index matrix are non-padding, shape (batch, masked_len).
            sample_num: the number of samples at each masked position (scalar).

        Returns: scores, seq_mask — the inspiration scores of the target items at the target masked positions
        towards the other interactions, and a mask marking which positions are real masks, shape (batch, masked_len).

        """
        batch_size = masked_ids.size(0)
        masked_len = masked_ids.size(-1)

        # Use broadcasting to build the tensor of Hawkes coefficients to be predicted at all target masked positions:
        # x_tensors = self.item_embedding.weight.view(1, 1, -1, self.hidden_size.size(-1))
        # the embeddings of the target items for which inspiration scores are computed (1, 1, n_items, hidden_size)
        x_position = masked_ids  # (batch_num, masked_num)
        x_position_embedding = self.position_embedding(x_position).unsqueeze(2)
        # (batch_num, masked_num, n_items, hidden_size)
        x_tensors = x_tensors + x_position_embedding  # add position embeddings

        # Compute the hidden representation of the masked sequence.
        masked_seq_emb = self.item_embedding(masked_item_seq)
        position_ids = torch.arange(
            self.max_seq_length, dtype=torch.long, device=masked_item_seq.device
        )
        position_ids = position_ids.unsqueeze(0)
        position_embs = self.position_embedding(position_ids)  # (1, seq_len, hidden_size)
        masked_seq_emb = masked_seq_emb + position_embs  # the hidden representation of the masked sequence

        # Compute the time interval between each masked position and the other positions.
        masked_ids = x_position
        masked_time_seq = torch.gather(time_seq, index=masked_ids, dim=1)  # (batch, masked_len) the timestamps of the masked positions
        masked_time_seq = masked_time_seq.unsqueeze(2).expand(-1, -1, time_seq.size(-1))  # (batch, masked_len, seq_len)
        time_seq = time_seq.unsqueeze(1)  # (batch, 1, seq_len)
        time_intervals_mat = torch.abs(masked_time_seq - time_seq)  # the time-interval matrix; each row gives the intervals between a masked position and the other interactions
        # (batch, masked_len, seq_len) the absolute value is taken here to enable bidirectional Hawkes modeling.
        time_intervals_mat = self.calculate_bukkit(time_intervals_mat)
        time_intervals_embs = self.time_interval_embedding(time_intervals_mat)
        # the time-interval embedding matrix (batch, masked_len, seq_len, hidden_size)

        # Obtain the mask matrix of each masked_seq, used to compute the attention scores.

        seq_mask = valid_mat != 0  # find the real mask indices in the masked sequence (batch_size, masked_len)
        score_mask = torch.zeros(batch_size, self.max_seq_length, dtype=torch.float64,
                                 device=self.device)  # marks which positions of each sequence are masked
        row_ids = torch.arange(batch_size, dtype=torch.long, device=self.device).unsqueeze(1).expand(-1, masked_len)[
            seq_mask]
        col_ids = masked_ids[seq_mask]
        score_mask[row_ids, col_ids] = float('-inf')  # set the mask value at the masked-index positions to negative infinity
        padding_mask = masked_item_seq == 0
        score_mask[padding_mask] = float('-inf')  # set the padding positions to -inf as well
        score_mask = score_mask.unsqueeze(1).unsqueeze(1).expand(-1, masked_len, sample_num, -1)
        # (batch, masked_len, sample_num, seq_len) the final mask for the inspiration scores.

        # Feed into the attention-score layer to compute the inspiration scores; remember LayerNorm and Dropout.
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
        Perform random insertion on the original sequence. First insert the placeholder n_items.
        """
        raw_item_seq = interaction[self.ITEM_SEQ]
        raw_time_seq = interaction[self.TIME_SEQ]
        new_item_seq, new_time_seq = self.data_augment(interaction)
        """
        Feed the augmented user interaction sequence into the time-aware attention layer.
        """
        aug_seq_output, origin_aug_seq = self.forward(new_item_seq, new_time_seq)
        raw_seq_output, origin_raw_seq = self.forward(raw_item_seq, raw_time_seq)
        pos_items = interaction[self.POS_ITEM_ID]

        """
        Below computes the final prediction loss, using the time-step hidden representation produced by the time-aware attention.
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
            all_logits = torch.matmul(all_outputs, test_item_emb.transpose(0, 1))  # (batch, item_num)
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
        For generality, RecBoLE calls this method during the evaluation and testing phases to score each user
        sequence's positive item and its negative-sample set. Since scoring is performed per target item per
        sequence, all positive and negative samples of a sequence are scored simultaneously by constructing
        separate sequences for the positive and each negative sample. This multiplies the computational
        overhead: originally, scoring the positive and negative samples of one sequence required only a single
        forward pass, but with, say, 100 negative samples, 100 additional forward passes are now needed.
        There is no need to worry about dropout or other stochastic operations affecting the forward output,
        because this method is called only during evaluation and testing. Before evaluation/testing, the
        trainer sets the model to eval mode, disabling dropout and other stochastic layers, which fixes the
        forward output. Therefore, constructing multiple interaction sequences for positive and negative samples
        does not affect the final score calculation.
        Args:
            interaction: user interactions built for multiple users' sequences, with the target item being
            either a positive sample or a single negative sample for each sequence; one interaction per negative sample.

        Returns (torch.Tensor): scores of shape (batch,) for each interaction-sequence sample.

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
            interaction: (batch_size,) a batch of interaction data containing the negative-sampling sequence for each sequence.

        Returns:
            a tensor of shape (batch_size, (neg_num+1)).

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
        scores = torch.cat([pos_scores, neg_scores], dim=1)  # concatenate positive and negative sample scores by column (batch, (neg_num+1))
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        time_seq = interaction[self.TIME_SEQ]
        seq_output, _ = self.forward(item_seq, time_seq)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
