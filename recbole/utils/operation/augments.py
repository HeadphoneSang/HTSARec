import torch


class DataAugmentation():
    def __init__(self, config, model):
        self.data_augment = config.data_augment
        self.model = model

    def augment(self, interaction):
        """
        Takes in a RecBoLE batch data type, augments all data in this batch, and returns the augmented
        item sequence and time sequence.
        Args:
            interaction: all user interaction data under one batch

        Returns:
        new_item_seq(batch, max_seq_len) augmented user interaction sequence,
        new_time_seq(batch, max_seq_len) augmented user time sequence

        """
        raise NotImplementedError("You have not specified a concrete data augmentation method")

    def __call__(self, interaction):
        return self.augment(interaction)


class TiInsertAugmentation(DataAugmentation):
    def __init__(self, config, model):
        super(TiInsertAugmentation, self).__init__(config, model)
        self.insert_ratio = config['data_augments']['insert_ratio']

    def augment(self, interaction):
        origin_item_seq = interaction[self.model.ITEM_SEQ]  # (batch_size, max_len)
        origin_time_seq = interaction[self.model.TIME_SEQ]
        seq_lens = interaction[self.model.ITEM_SEQ_LEN]  # (batch,)

        batch_size, max_len = origin_item_seq.shape
        dev = origin_item_seq.device

        new_item_seq = torch.zeros_like(origin_item_seq, device=dev)
        new_time_seq = torch.zeros_like(origin_time_seq, device=dev)
        pad_token = self.model.n_items
        max_insert_len = int(self.insert_ratio * max_len)
        insert_idx = torch.full((batch_size, max_insert_len), max_len - 1, device=dev)
        new_insert_positions = []
        for i in range(batch_size):
            seq_len = seq_lens[i].item()
            # Compute the theoretical number of insertions
            num_insert = int(seq_len * self.insert_ratio)
            if num_insert < 1:
                new_insert_positions.append([0] * max_insert_len)
                new_item_seq[i] = origin_item_seq[i]
                new_time_seq[i] = origin_time_seq[i]
                continue

            # Cannot exceed max_len
            num_insert = min(num_insert, max_len - seq_len)
            cur_idx = max_len - (num_insert + seq_len)  # the starting index of the new sequence
            if cur_idx > 0:
                new_time_seq[i, 0:cur_idx] = origin_time_seq[i, 0]
            # Obtain the time-interval sequence and compute the position with the largest interval
            time_intervals_seq = origin_time_seq[i, max_len-seq_len+1:] - origin_time_seq[i, max_len-seq_len:-1]  # (max_seq_len) the time interval between i and i+1
            _, insert_positions = torch.sort(time_intervals_seq, dim=-1, descending=True)
            insert_positions = insert_positions + (max_len-seq_len)
            insert_positions = insert_positions[:num_insert]
            # Sort from small to large so that the rightward shift during insertion does not disrupt subsequent insertions
            insert_positions, _ = torch.sort(insert_positions)  # the indices in the original sequence after which elements should be inserted
            insert_idx[i, max_insert_len - num_insert:] = insert_positions
            pre_insert_pos = max_len - seq_len - 1
            # Iterate over insertion positions (note that after each insertion, seq_len must be updated +1)
            new_insert_pos = []
            for insert_pos in insert_positions:
                # Construct the new sequence
                insert_pos = insert_pos.item()
                insert_seq = origin_item_seq[i, pre_insert_pos + 1:insert_pos + 1]
                insert_time_seq = origin_time_seq[i, pre_insert_pos + 1:insert_pos + 1]
                new_item_seq[i, cur_idx:cur_idx + len(insert_seq)] = insert_seq
                new_time_seq[i, cur_idx:cur_idx + len(insert_time_seq)] = insert_time_seq
                cur_idx += len(insert_seq)
                new_item_seq[i, cur_idx] = pad_token
                new_time_seq[i, cur_idx] = origin_time_seq[i, insert_pos]
                new_insert_pos.append(cur_idx)
                cur_idx += 1
                pre_insert_pos = insert_pos
            new_item_seq[i, cur_idx:] = origin_item_seq[i, pre_insert_pos + 1:]
            new_time_seq[i, cur_idx:] = origin_time_seq[i, pre_insert_pos + 1:]
            new_insert_pos = [0] * (max_insert_len - len(new_insert_pos)) + new_insert_pos
            new_insert_positions.append(new_insert_pos)

        new_insert_positions = torch.tensor(new_insert_positions, device=dev)
        real_pos_mask = insert_idx != (max_len - 1)  # (batch, max_insert_len) True indicates real insertion positions
        insert_ner_ids = torch.gather(origin_item_seq, dim=1, index=insert_idx)  # (batch, max_insert_len) the ID of the preceding neighbor at the insertion positions
        insert_ner_embeds = self.model.item_embedding(insert_ner_ids)  # (batch,max_insert_len,hidden_size)
        similar_scores = torch.matmul(insert_ner_embeds,
                                      self.model.item_embedding.weight.unsqueeze(0).transpose(-2, -1))
        # (batch, max_insert_len, item_n) scores of all items at each insertion position
        sim_rows = torch.arange(batch_size, device=dev).unsqueeze(1).expand(-1,
                                                                            max_insert_len)  # (batch,max_insert_len)
        pos_idx = torch.arange(max_insert_len, device=dev).unsqueeze(0).expand(batch_size, -1)
        # (batch,max_insert_len,item_n)
        similar_scores[sim_rows, pos_idx, insert_ner_ids] = float('-inf')
        # Select the highest-scoring item at each position, yielding (batch, max_insert_len) storing the items to insert
        _, indices = torch.max(similar_scores, dim=-1)  # (batch,max_insert_len)
        insert_values = indices[real_pos_mask]  # select the values to insert
        insert_row_ids = torch.arange(batch_size, device=dev).unsqueeze(1).expand(-1, max_insert_len)
        insert_row_ids = insert_row_ids[real_pos_mask]
        insert_col_ids = new_insert_positions[real_pos_mask]
        new_item_seq[insert_row_ids, insert_col_ids] = insert_values
        return new_item_seq, new_time_seq


class InsertAugmentation(DataAugmentation):
    def __init__(self, config, model):
        super(InsertAugmentation, self).__init__(config, model)
        self.insert_ratio = config['data_augments']['insert_ratio']

    def augment(self, interaction):
        origin_item_seq = interaction[self.model.ITEM_SEQ]  # (batch_size, max_len)
        origin_time_seq = interaction[self.model.TIME_SEQ]
        seq_lens = interaction[self.model.ITEM_SEQ_LEN]  # (batch,)

        batch_size, max_len = origin_item_seq.shape
        dev = origin_item_seq.device

        new_item_seq = torch.zeros_like(origin_item_seq, device=dev)
        new_time_seq = torch.zeros_like(origin_time_seq, device=dev)
        pad_token = self.model.n_items
        max_insert_len = int(self.insert_ratio * max_len)
        insert_idx = torch.full((batch_size, max_insert_len), max_len - 1, device=dev)
        new_insert_positions = []
        for i in range(batch_size):
            seq_len = seq_lens[i].item()
            # Compute the theoretical number of insertions
            num_insert = int(seq_len * self.insert_ratio)
            if num_insert < 1:
                new_insert_positions.append([0] * max_insert_len)
                new_item_seq[i] = origin_item_seq[i]
                new_time_seq[i] = origin_time_seq[i]
                continue

            # Cannot exceed max_len
            num_insert = min(num_insert, max_len - seq_len)
            cur_idx = max_len - (num_insert + seq_len)  # the starting index of the new sequence
            if cur_idx > 0:
                new_time_seq[i, 0:cur_idx] = origin_time_seq[i, 0]
            insert_positions = torch.randint(low=max_len - seq_len, high=max_len - 1, size=(num_insert,), device=dev)
            # Sort from small to large so that the rightward shift during insertion does not disrupt subsequent insertions
            insert_positions, _ = torch.sort(insert_positions)  # the indices in the original sequence after which elements should be inserted
            insert_idx[i, max_insert_len - num_insert:] = insert_positions
            pre_insert_pos = max_len - seq_len - 1
            # Iterate over insertion positions (note that after each insertion, seq_len must be updated +1)
            new_insert_pos = []
            for insert_pos in insert_positions:
                # Construct the new sequence
                insert_pos = insert_pos.item()
                insert_seq = origin_item_seq[i, pre_insert_pos + 1:insert_pos + 1]
                insert_time_seq = origin_time_seq[i, pre_insert_pos + 1:insert_pos + 1]
                new_item_seq[i, cur_idx:cur_idx + len(insert_seq)] = insert_seq
                new_time_seq[i, cur_idx:cur_idx + len(insert_time_seq)] = insert_time_seq
                cur_idx += len(insert_seq)
                new_item_seq[i, cur_idx] = pad_token
                new_time_seq[i, cur_idx] = origin_time_seq[i, insert_pos]
                new_insert_pos.append(cur_idx)
                cur_idx += 1
                pre_insert_pos = insert_pos
            new_item_seq[i, cur_idx:] = origin_item_seq[i, pre_insert_pos + 1:]
            new_time_seq[i, cur_idx:] = origin_time_seq[i, pre_insert_pos + 1:]
            new_insert_pos = [0] * (max_insert_len - len(new_insert_pos)) + new_insert_pos
            new_insert_positions.append(new_insert_pos)

        new_insert_positions = torch.tensor(new_insert_positions, device=dev)
        real_pos_mask = insert_idx != (max_len - 1)  # (batch, max_insert_len) True indicates real insertion positions
        insert_ner_ids = torch.gather(origin_item_seq, dim=1, index=insert_idx)  # (batch, max_insert_len) the ID of the preceding neighbor at the insertion positions
        insert_ner_embeds = self.model.item_embedding(insert_ner_ids)  # (batch, max_insert_len, hidden_size)
        similar_scores = torch.matmul(insert_ner_embeds,
                                      self.model.item_embedding.weight.unsqueeze(0).transpose(-2, -1))
        # (batch, max_insert_len, item_n) scores of all items at each insertion position
        sim_rows = torch.arange(batch_size, device=dev).unsqueeze(1).expand(-1, max_insert_len)  # (batch, max_insert_len)
        pos_idx = torch.arange(max_insert_len, device=dev).unsqueeze(0).expand(batch_size, -1)
        # (batch, max_insert_len, item_n)
        similar_scores[sim_rows, pos_idx, insert_ner_ids] = float('-inf')
        # Select the highest-scoring item at each position, yielding (batch, max_insert_len) storing the items to insert
        _, indices = torch.max(similar_scores, dim=-1)  # (batch, max_insert_len)
        insert_values = indices[real_pos_mask]  # select the values to insert
        insert_row_ids = torch.arange(batch_size, device=dev).unsqueeze(1).expand(-1, max_insert_len)
        insert_row_ids = insert_row_ids[real_pos_mask]
        insert_col_ids = new_insert_positions[real_pos_mask]
        new_item_seq[insert_row_ids, insert_col_ids] = insert_values
        return new_item_seq, new_time_seq


class ReOrderAugmentation(DataAugmentation):
    def __init__(self, config, model):
        super(ReOrderAugmentation, self).__init__(config, model)
        self.mask_ratios = config['data_augments']['reorder_ratio']  # Use the same ratio range to control segment length

    def augment(self, interaction):
        origin_item_seq = interaction[self.model.ITEM_SEQ]  # (batch, max_len)
        origin_time_seq = interaction[self.model.TIME_SEQ]
        batch_size, max_len = origin_item_seq.shape
        dev = origin_item_seq.device

        # ====== Randomly generate the segment length for each sequence ======
        b, e = self.mask_ratios
        rand_ratios = (e - b) * torch.rand(batch_size, device=dev) + b
        seq_lens = interaction[self.model.ITEM_SEQ_LEN]  # (batch,)
        seg_lens = (seq_lens.float() * rand_ratios).long()  # the length of each sequence to be shuffled

        # ====== Randomly generate the segment start for each sequence ======
        origin_starts = (max_len - seq_lens).long()  # the start of the left padding
        max_starts = (max_len - seg_lens).long()  # the upper bound of the start
        rand = torch.rand(batch_size, device=dev)
        real_start_idx = (origin_starts + rand * (max_starts - origin_starts)).long()  # (batch,)

        # ====== Generate new sequences ======
        new_item_seq = origin_item_seq.clone()
        new_time_seq = origin_time_seq.clone()  # keep time unchanged

        for i in range(batch_size):
            seg_len = seg_lens[i].item()
            if seg_len > 1:  # shuffle only when length > 1
                start = real_start_idx[i].item()
                end = start + seg_len

                # Shuffle the item sub-segment while keeping time unchanged
                sub_items = new_item_seq[i, start:end]
                perm = torch.randperm(seg_len, device=dev)
                new_item_seq[i, start:end] = sub_items[perm]

        return new_item_seq, new_time_seq


class MaskAugmentation(DataAugmentation):
    def __init__(self, config, model):
        super(MaskAugmentation, self).__init__(config, model)
        self.mask_ratios = config['data_augments']['mask_ratio']

    def augment(self, interaction):
        origin_item_seq = interaction[self.model.ITEM_SEQ]  # the original user interaction sequence (batch_size, max_seq_len)
        origin_time_seq = interaction[self.model.TIME_SEQ]
        batch_size = origin_item_seq.shape[0]
        max_len = origin_item_seq.shape[1]
        cur_device = origin_item_seq.device
        b = self.mask_ratios[0]
        e = self.mask_ratios[1]
        rand_ratios = (e - b) * torch.rand(batch_size, device=cur_device) + b
        seq_lens = interaction[self.model.ITEM_SEQ_LEN]  # batch length, the length of each interaction sequence
        mask_len = torch.mul(seq_lens, rand_ratios).long()  # the length to mask for each sequence
        # Compute the start index for cropping each sequence
        origin_starts = max_len - seq_lens  # (batch,)  # minimum index
        max_starts = max_len - mask_len  # (batch,)  # maximum index, may exceed max_len
        rand = torch.rand(batch_size, device=cur_device)
        real_start_idx = (origin_starts + rand * (max_starts - origin_starts)).int()  # store the actual crop index for each sequence
        # Prepare output
        pad_token = self.model.n_items  # adjust according to your data
        new_item_seq = origin_item_seq.clone()
        new_time_seq = origin_time_seq.clone()
        # Obtain a (batch, max_len) index matrix
        arrange_ids = torch.arange(max_len, dtype=torch.long, device=cur_device).unsqueeze(0).expand(batch_size, -1)
        # The mask start index for each sequence (batch_size, max_len), left-closed (inclusive)
        real_start_matrix = real_start_idx.unsqueeze(1).expand(batch_size, max_len)
        # The random mask end index for each sequence (batch_size, max_len), right-open (exclusive)
        real_end_matrix = (real_start_idx + mask_len).unsqueeze(1).expand(batch_size, max_len)
        token_mask = (arrange_ids >= real_start_matrix) & (arrange_ids < real_end_matrix)
        new_item_seq[token_mask] = pad_token
        return new_item_seq, new_time_seq


class CropAugmentation(DataAugmentation):
    def __init__(self, config, model):
        super(CropAugmentation, self).__init__(config, model)
        self.ratio_range = config['data_augments']['crop_ratio']

    def augment(self, interaction):
        origin_item_seq = interaction[self.model.ITEM_SEQ]  # the original user interaction sequence (batch_size, max_seq_len)
        origin_time_seq = interaction[self.model.TIME_SEQ]
        batch_size = origin_item_seq.shape[0]
        max_len = origin_item_seq.shape[1]
        cur_device = origin_item_seq.device
        b = self.ratio_range[0]
        e = self.ratio_range[1]
        rand_ratios = (e - b) * torch.rand(batch_size, device=cur_device) + b
        seq_lens = interaction[self.model.ITEM_SEQ_LEN]  # batch length, the length of each interaction sequence
        crop_len = torch.clamp_min(torch.mul(seq_lens, rand_ratios).long(), 1)  # the length to crop for each sequence
        # Compute the start index for cropping each sequence
        origin_starts = max_len - seq_lens  # (batch,)  # minimum index
        max_starts = max_len - crop_len  # (batch,)  # maximum index
        rand = torch.rand(batch_size, device=cur_device)
        real_start_idx = (origin_starts + rand * (max_starts - origin_starts)).int()  # store the actual crop index for each sequence
        # Prepare output
        pad_token = 0  # adjust according to your data
        new_item_seq = torch.full_like(origin_item_seq, pad_token, device=cur_device)
        new_time_seq = torch.full_like(origin_time_seq, pad_token, device=cur_device)
        for i in range(batch_size):
            start = real_start_idx[i].item()
            end = start + crop_len[i].item()
            sub_item = origin_item_seq[i, start:end]
            sub_time = origin_time_seq[i, start:end]

            # Place on the right
            new_item_seq[i, -crop_len[i]:] = sub_item

            # Use the earliest time of the cropped sequence for the left padding
            left_pad_len = max_len - crop_len[i]
            if left_pad_len > 0:
                new_time_seq[i, :left_pad_len] = sub_time[0]
            new_time_seq[i, -crop_len[i]:] = sub_time
        return new_item_seq, new_time_seq


class NoDataAugmentation(DataAugmentation):
    """
    No data augmentation.
    """

    def __init__(self, config, model):
        super(NoDataAugmentation, self).__init__(config, model)

    def augment(self, interaction):
        origin_item_seq = interaction[self.model.ITEM_SEQ]  # the original user interaction sequence (batch_size, max_seq_len)
        origin_time_seq = interaction[self.model.TIME_SEQ]
        return origin_item_seq, origin_time_seq


class HawkesInsertAugmentation(DataAugmentation):
    """
    Perform Hawkes insertion on the sequence to be predicted.
    """

    def __init__(self, config, model):
        super(HawkesInsertAugmentation, self).__init__(config, model)
        # Set the parameters of the uniformity generator
        self.insert_ratio = config['generator_args']['insert_ratio']
        self.insert_num = int(self.model.max_seq_length * self.insert_ratio)
        self.top_k = config['generator_args']['top_k']
        self.uni_method = config['uni_method']
        self.insert_mode = config['insert_mode']

    def augment_interaction(self, interaction):
        """
        Based on the user's original interaction sequence, insert placeholder masks at the positions with the
        largest time intervals (for a specified number), and generate a placeholder sequence ready for insertion.
        Only non-padding positions are masked to avoid introducing noise.
        Args:
            interaction: user interaction data
        Returns (tuple): item_seq (batch_size, max_seq) the user sequence with placeholders inserted,
            time_seq (batch_size, max_seq) the augmented user time sequence,
            insert_place (batch, insert_num) marks which positions each user needs to insert,
            insert_num_mat (batch) marks the actual number of insertions for each user sequence
        """
        item_seq_len = interaction[self.model.ITEM_SEQ_LEN]
        origin_item_seq = interaction[self.model.ITEM_SEQ]  # the original user interaction sequence (batch_size, max_seq_len)
        origin_time_seq = interaction[self.model.TIME_SEQ]  # the original user interaction timestamp sequence
        seq_id_seq = interaction[self.model.SEQ_ID_LIST]
        neighbor_interval_mat = origin_time_seq[:, 1:] - origin_time_seq[:, :-1]  # neighboring time-difference matrix (batch_size, max_len-1)
        desc_ids_interval = torch.argsort(neighbor_interval_mat, dim=1, descending=True)
        # index matrix of the time-difference matrix sorted by interval size, (batch_size, max_seq_len-1)
        place_v = self.model.n_items  # placeholder element to be placed at future inserted augmentation positions

        # Compute how many elements to insert for each sequence
        left_seq_len = self.model.max_seq_length - item_seq_len
        # """
        # To change to per-sequence proportional insertion, replace self.insert_num here with a (512) vector so each sequence has its own insertion count
        # """
        #
        insert_num_mat = None
        if self.insert_mode == 'abs':
            insert_num_mat = torch.clamp_max(left_seq_len, self.insert_num)  # obtain the number of insertions for each interaction target
        elif self.insert_mode == 'rel':
            insert_num_mat = (item_seq_len * self.insert_ratio).long()  # relative insertion ratio
            insert_num_mat = torch.min(insert_num_mat, left_seq_len)
        assert insert_num_mat is not None
        # To prevent the case where the original sequence's valid length has already reached the maximum length,
        # or the relative insertion length exceeds the sequence's allowable insertion length,
        # a size comparison is performed between the current relative insertion count and the sequence's allowable insertion count
        # to ensure the relative insertion count is less than or equal to the allowable insertion count.

        """
        Build a new collection of interaction sequences according to insert_num_mat and desc_ids_interval.
        In the new interaction sequences, each sequence selects a target number of positions from front to back
        in desc_ids_interval according to insert_num_mat and inserts the place_v placeholder.
        """
        new_item_seq = []
        new_time_seq = []
        insert_place = []
        origin_item_seq = origin_item_seq.tolist()  # the original user interaction sequence (batch_size, max_seq_len)
        origin_time_seq = origin_time_seq.tolist()  # the original user interaction timestamp sequence
        seq_id_seq = seq_id_seq.tolist()
        for b, (inter_seq, inter_time_seq, seq_id) in enumerate(zip(origin_item_seq, origin_time_seq, seq_id_seq)):
            insert_num = insert_num_mat[b].item()
            insert_pos = desc_ids_interval[b][:insert_num].sort()[0].tolist()
            new_items = []
            new_times = []
            insert_marks = []
            if seq_id in self.model.no_aug_seq:  # for sequences that do not need data augmentation, set all new sequences to the original ones; insertion coordinates are all 0-padding positions so later augmentation does not insert at position 0
                new_items = inter_seq
                new_times = inter_time_seq
                insert_marks = [0] * self.insert_num
            else:
                prev_idx = insert_num  # record the position of the last inserted element in the original interaction sequence, which is also the start index of the original sequence to be prepended for the next insertion; should be initialized to the insertion count
                # prev_idx is initialized to insert_num because insert_num is smaller than the number of padding entries; to be able to insert enough elements,
                # directly discarding the first insert_num elements does not affect the valid items of the original sequence.
                for idx in insert_pos:  # in ascending index order
                    idx = idx + 1  # insert to the right of the interval
                    if inter_seq[idx - 1] == 0:  # skip the padding insertion position to avoid adding noise
                        insert_num_mat[b] -= 1
                        prev_idx -= 1
                        continue
                    new_items.extend(inter_seq[prev_idx:idx])
                    new_times.extend(inter_time_seq[prev_idx:idx])
                    # Insert place_v, with time averaged between the two ends
                    t_left = inter_time_seq[idx - 1]
                    # t_right = inter_time_seq[idx].item() if idx < len(inter_time_seq) else t_left
                    t_right = inter_time_seq[idx]  # no need to check out-of-bounds, since idx+1 will not exceed bounds
                    place_t = (t_left + t_right) // 2
                    new_items.append(place_v)
                    new_times.append(place_t)
                    insert_marks.append(len(new_items) - 1)  # insertion point position index
                    prev_idx = idx

                # Append the remaining trailing original interactions
                new_items.extend(inter_seq[prev_idx:])
                new_times.extend(inter_time_seq[prev_idx:])
                padding_list = [0] * (self.insert_num - insert_num_mat[b].item())
                insert_marks = padding_list + insert_marks

            # Append the current interaction sequence's augmentation-related content
            new_item_seq.append(new_items)
            new_time_seq.append(new_times)
            insert_place.append(insert_marks)

        # Convert all results to tensors
        new_item_seq = torch.tensor(new_item_seq, dtype=torch.long, device=self.model.device)
        new_time_seq = torch.tensor(new_time_seq, dtype=torch.long, device=self.model.device)
        insert_place = torch.tensor(insert_place, dtype=torch.long, device=self.model.device)  # 0 represents padding, since no augmentation element is added at the very front of the sequence
        return new_item_seq, new_time_seq, insert_place, insert_num_mat

    def augment(self, interaction):
        new_item_seq, new_time_seq, insert_place, insert_num_mat = self.augment_interaction(interaction)
        """
        Use bert_forward to bidirectionally encode the sequence after placeholder insertion, then extract the hidden
        representation matrix at the masked positions according to the insertion index matrix, perform a full-item
        dot product, take the top-k at each position, compute the Hawkes intensity, and finally insert the highest-scoring
        elements into the original sequence.
        """
        """
        TODO: self.model is the model, and the embedding layer can be obtained via self.item_embedding.
        new_item_seq: the sequence awaiting augmentation insertion, where 0 represents padding.
        self.model.n_items: the placeholder element used for future inserted augmentation positions.
        """
        bert_seq_output = self.model.bert_forward(new_item_seq)  # returns the hidden representation of the masked sequence (batch, seq_len, hidden_size)
        bert_index = insert_place.unsqueeze(-1).expand(-1, -1, self.model.hidden_size)
        # (batch, insert_num, hidden_size) index tensor for extracting hidden representations at insertion positions
        insert_hidden = torch.gather(bert_seq_output, index=bert_index, dim=1)
        if self.uni_method == 'topk':
            all_items_scores = torch.matmul(insert_hidden,
                                            self.model.item_embedding.weight[1:self.model.n_items].transpose(0, 1))
            # (batch, masked_len, n_items-1) ignores similarity with 0 and n_items since they are not to be recommended
            _, top_k_indices = torch.topk(all_items_scores, k=self.top_k, dim=-1)
            top_k_indices += 1
            # Obtain the (batch, masked_len, top_k) list of scores and IDs of the most similar items; since similarity calculation ignored position 0, each top_k index should be incremented by 1
            x_tensors = self.model.item_embedding(top_k_indices)
            inspire_scores, seq_mask = self.model.calculate_all_inspire_score(x_tensors, new_item_seq, new_time_seq,
                                                                              insert_place, insert_place, self.top_k)
            # (batch, masked_len, top_k, seq_len), (batch, masked_len)
            native_scores = self.model.calculate_native_scores(insert_place, bert_seq_output, new_item_seq, self.top_k)
            # (batch, masked_len, top_k, seq_len)
            hawkes_scores = (native_scores * inspire_scores).sum(dim=-1)  # element-wise weighted sum, yielding (batch, masked_len, top_k)
            """
            Find the ID of the highest-scoring item at each position via the maximum, then use seq_mask to extract the
            corresponding row and column indices and values, and fill them into the sequence.
            """
            _, sorted_ids = torch.sort(hawkes_scores, dim=-1, descending=True)
            sorted_items = torch.gather(top_k_indices, index=sorted_ids, dim=-1)
            # (batch, masked_len, top_k) item list at each masked position sorted by final score
            insert_items = sorted_items[:, :, 0:1].squeeze(-1)  # (batch, masked_len) tensor of the element to insert at each position of each interaction
            insert_items = insert_items[seq_mask]
            row_ids = (torch.arange(new_item_seq.size(0), dtype=torch.long, device=self.model.device)
                       .unsqueeze(1).expand(-1, insert_place.size(-1)))
            row_ids = row_ids[seq_mask]
            col_ids = insert_place[seq_mask]
            new_item_seq[row_ids, col_ids] = insert_items
        return new_item_seq, new_time_seq
