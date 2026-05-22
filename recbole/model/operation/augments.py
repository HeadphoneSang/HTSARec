import torch


class DataAugmentation():
    def __init__(self, config, model):
        self.data_augment = config.data_augment
        self.model = model

    def augment(self, interaction):
        raise NotImplementedError("You haven't specified a clear data augmentation method")

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
            # Calculate the theoretical insertion quantity
            num_insert = int(seq_len * self.insert_ratio)
            if num_insert < 1:
                new_insert_positions.append([0] * max_insert_len)
                new_item_seq[i] = origin_item_seq[i]
                new_time_seq[i] = origin_time_seq[i]
                continue

            # max_len
            num_insert = min(num_insert, max_len - seq_len)
            cur_idx = max_len - (num_insert + seq_len)
            if cur_idx > 0:
                new_time_seq[i, 0:cur_idx] = origin_time_seq[i, 0]
            time_intervals_seq = origin_time_seq[i, max_len-seq_len+1:] - origin_time_seq[i, max_len-seq_len:-1]  #(max_seq_len)
            _, insert_positions = torch.sort(time_intervals_seq, dim=-1, descending=True)
            insert_positions = insert_positions + (max_len-seq_len)
            insert_positions = insert_positions[:num_insert]
            insert_positions, _ = torch.sort(insert_positions)
            insert_idx[i, max_insert_len - num_insert:] = insert_positions
            pre_insert_pos = max_len - seq_len - 1
            new_insert_pos = []
            for insert_pos in insert_positions:
                # 构建新的序列
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
        real_pos_mask = insert_idx != (max_len - 1)  # (batch,max_insert_len)
        insert_ner_ids = torch.gather(origin_item_seq, dim=1, index=insert_idx)  # (batch,max_insert_len)
        insert_ner_embeds = self.model.item_embedding(insert_ner_ids)  # (batch,max_insert_len,hidden_size)
        similar_scores = torch.matmul(insert_ner_embeds,
                                      self.model.item_embedding.weight.unsqueeze(0).transpose(-2, -1))
        # (batch,max_insert_len,item_n)
        sim_rows = torch.arange(batch_size, device=dev).unsqueeze(1).expand(-1,
                                                                            max_insert_len)  # (batch,max_insert_len)
        pos_idx = torch.arange(max_insert_len, device=dev).unsqueeze(0).expand(batch_size, -1)
        # (batch,max_insert_len,item_n)
        similar_scores[sim_rows, pos_idx, insert_ner_ids] = float('-inf')
        _, indices = torch.max(similar_scores, dim=-1)  # (batch,max_insert_len)
        insert_values = indices[real_pos_mask]
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
            num_insert = int(seq_len * self.insert_ratio)
            if num_insert < 1:
                new_insert_positions.append([0] * max_insert_len)
                new_item_seq[i] = origin_item_seq[i]
                new_time_seq[i] = origin_time_seq[i]
                continue

            num_insert = min(num_insert, max_len - seq_len)
            cur_idx = max_len - (num_insert + seq_len)
            if cur_idx > 0:
                new_time_seq[i, 0:cur_idx] = origin_time_seq[i, 0]
            insert_positions = torch.randint(low=max_len - seq_len, high=max_len - 1, size=(num_insert,), device=dev)
            insert_positions, _ = torch.sort(insert_positions)  # The index of the original sequence corresponding to
            # which elements to insert after
            insert_idx[i, max_insert_len - num_insert:] = insert_positions
            pre_insert_pos = max_len - seq_len - 1

            new_insert_pos = []
            for insert_pos in insert_positions:
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
        real_pos_mask = insert_idx != (max_len - 1)  #(batch,max_insert_len)
        insert_ner_ids = torch.gather(origin_item_seq, dim=1, index=insert_idx)  #(batch,max_insert_len)
        insert_ner_embeds = self.model.item_embedding(insert_ner_ids)  #(batch,max_insert_len,hidden_size)
        similar_scores = torch.matmul(insert_ner_embeds,
                                      self.model.item_embedding.weight.unsqueeze(0).transpose(-2, -1))
        sim_rows = torch.arange(batch_size, device=dev).unsqueeze(1).expand(-1, max_insert_len)  #(batch,max_insert_len)
        pos_idx = torch.arange(max_insert_len, device=dev).unsqueeze(0).expand(batch_size, -1)
        # (batch,max_insert_len,item_n)
        similar_scores[sim_rows, pos_idx, insert_ner_ids] = float('-inf')
        _, indices = torch.max(similar_scores, dim=-1)  #(batch,max_insert_len)
        insert_values = indices[real_pos_mask]
        insert_row_ids = torch.arange(batch_size, device=dev).unsqueeze(1).expand(-1, max_insert_len)
        insert_row_ids = insert_row_ids[real_pos_mask]
        insert_col_ids = new_insert_positions[real_pos_mask]
        new_item_seq[insert_row_ids, insert_col_ids] = insert_values
        return new_item_seq, new_time_seq


class ReOrderAugmentation(DataAugmentation):
    def __init__(self, config, model):
        super(ReOrderAugmentation, self).__init__(config, model)
        self.mask_ratios = config['data_augments']['reorder_ratio']

    def augment(self, interaction):
        origin_item_seq = interaction[self.model.ITEM_SEQ]  # (batch, max_len)
        origin_time_seq = interaction[self.model.TIME_SEQ]
        batch_size, max_len = origin_item_seq.shape
        dev = origin_item_seq.device

        # ====== Randomly generate the fragment length of each sequence ======
        b, e = self.mask_ratios
        rand_ratios = (e - b) * torch.rand(batch_size, device=dev) + b
        seq_lens = interaction[self.model.ITEM_SEQ_LEN]  # (batch,)
        seg_lens = (seq_lens.float() * rand_ratios).long()

        # ====== Randomly generate the fragment starting point of each sequence ======
        origin_starts = (max_len - seq_lens).long()
        max_starts = (max_len - seg_lens).long()
        rand = torch.rand(batch_size, device=dev)
        real_start_idx = (origin_starts + rand * (max_starts - origin_starts)).long()  # (batch,)

        # ====== Generate a new sequence ======
        new_item_seq = origin_item_seq.clone()
        new_time_seq = origin_time_seq.clone()

        for i in range(batch_size):
            seg_len = seg_lens[i].item()
            if seg_len > 1:
                start = real_start_idx[i].item()
                end = start + seg_len

                sub_items = new_item_seq[i, start:end]
                perm = torch.randperm(seg_len, device=dev)
                new_item_seq[i, start:end] = sub_items[perm]

        return new_item_seq, new_time_seq


class MaskAugmentation(DataAugmentation):
    def __init__(self, config, model):
        super(MaskAugmentation, self).__init__(config, model)
        self.mask_ratios = config['data_augments']['mask_ratio']

    def augment(self, interaction):
        origin_item_seq = interaction[self.model.ITEM_SEQ]
        origin_time_seq = interaction[self.model.TIME_SEQ]
        batch_size = origin_item_seq.shape[0]
        max_len = origin_item_seq.shape[1]
        cur_device = origin_item_seq.device
        b = self.mask_ratios[0]
        e = self.mask_ratios[1]
        rand_ratios = (e - b) * torch.rand(batch_size, device=cur_device) + b
        seq_lens = interaction[self.model.ITEM_SEQ_LEN]
        mask_len = torch.mul(seq_lens, rand_ratios).long()
        origin_starts = max_len - seq_lens  # (batch,)
        max_starts = max_len - mask_len  # (batch,)
        rand = torch.rand(batch_size, device=cur_device)
        real_start_idx = (origin_starts + rand * (max_starts - origin_starts)).int()
        pad_token = self.model.n_items
        new_item_seq = origin_item_seq.clone()
        new_time_seq = origin_time_seq.clone()
        arrange_ids = torch.arange(max_len, dtype=torch.long, device=cur_device).unsqueeze(0).expand(batch_size, -1)
        real_start_matrix = real_start_idx.unsqueeze(1).expand(batch_size, max_len)
        real_end_matrix = (real_start_idx + mask_len).unsqueeze(1).expand(batch_size, max_len)
        token_mask = (arrange_ids >= real_start_matrix) & (arrange_ids < real_end_matrix)
        new_item_seq[token_mask] = pad_token
        return new_item_seq, new_time_seq


class CropAugmentation(DataAugmentation):
    def __init__(self, config, model):
        super(CropAugmentation, self).__init__(config, model)
        self.ratio_range = config['data_augments']['crop_ratio']

    def augment(self, interaction):
        origin_item_seq = interaction[self.model.ITEM_SEQ]  # (batch_size,max_seq_len)
        origin_time_seq = interaction[self.model.TIME_SEQ]
        batch_size = origin_item_seq.shape[0]
        max_len = origin_item_seq.shape[1]
        cur_device = origin_item_seq.device
        b = self.ratio_range[0]
        e = self.ratio_range[1]
        rand_ratios = (e - b) * torch.rand(batch_size, device=cur_device) + b
        seq_lens = interaction[self.model.ITEM_SEQ_LEN]
        crop_len = torch.clamp_min(torch.mul(seq_lens, rand_ratios).long(), 1)
        origin_starts = max_len - seq_lens  #(batch,)
        max_starts = max_len - crop_len  #(batch,)
        rand = torch.rand(batch_size, device=cur_device)
        real_start_idx = (origin_starts + rand * (max_starts - origin_starts)).int()
        pad_token = 0
        new_item_seq = torch.full_like(origin_item_seq, pad_token, device=cur_device)
        new_time_seq = torch.full_like(origin_time_seq, pad_token, device=cur_device)
        for i in range(batch_size):
            start = real_start_idx[i].item()
            end = start + crop_len[i].item()
            sub_item = origin_item_seq[i, start:end]
            sub_time = origin_time_seq[i, start:end]
            new_item_seq[i, -crop_len[i]:] = sub_item
            left_pad_len = max_len - crop_len[i]
            if left_pad_len > 0:
                new_time_seq[i, :left_pad_len] = sub_time[0]
            new_time_seq[i, -crop_len[i]:] = sub_time
        return new_item_seq, new_time_seq


class NoDataAugmentation(DataAugmentation):

    def __init__(self, config, model):
        super(NoDataAugmentation, self).__init__(config, model)

    def augment(self, interaction):
        origin_item_seq = interaction[self.model.ITEM_SEQ]
        origin_time_seq = interaction[self.model.TIME_SEQ]
        return origin_item_seq, origin_time_seq


class HawkesInsertAugmentation(DataAugmentation):
    """
    Perform a Hawkes insertion operation on the sequence to be predicted
    """

    def __init__(self, config, model):
        super(HawkesInsertAugmentation, self).__init__(config, model)
        # Set the parameters of the homogenization generator
        self.insert_ratio = config['generator_args']['insert_ratio']
        self.insert_num = int(self.model.max_seq_length * self.insert_ratio)
        self.top_k = config['generator_args']['top_k']
        self.uni_method = config['uni_method']
        self.insert_mode = config['insert_mode']

    def augment_interaction(self, interaction):
        """
        According to the user's original interaction sequence,
        perform an insertion mask operation at the position of
        the specified number of maximum time intervals to generate
        a placeholder sequence ready for insertion
        Mask only up to non-padding to avoid introducing noise
        Args:
        Returns (tuple): item_seq (batch_size,max_seq)
        insert_place(batch,insert_num)
        interaction: User interaction data
        """
        item_seq_len = interaction[self.model.ITEM_SEQ_LEN]
        origin_item_seq = interaction[self.model.ITEM_SEQ]  #(batch_size,max_seq_len)
        origin_time_seq = interaction[self.model.TIME_SEQ]
        seq_id_seq = interaction[self.model.SEQ_ID_LIST]
        neighbor_interval_mat = origin_time_seq[:, 1:] - origin_time_seq[:, :-1]  # (batch_size,max_len-1)
        desc_ids_interval = torch.argsort(neighbor_interval_mat, dim=1, descending=True)
        #（batch_size,max_seq_len—1）
        place_v = self.model.n_items  # A placeholder element used for future insertion to enhance interaction positions

        # Calculate how many elements to insert into each sequence target
        left_seq_len = self.model.max_seq_length - item_seq_len
        """
        If it is changed to insert each sequence at its own proportion, 
        here change self.insert_num to a vector of (512), where each sequence inserts its own quantity
        """
        insert_num_mat = None
        if self.insert_mode == 'abs':
            insert_num_mat = torch.clamp_max(left_seq_len, self.insert_num)  # Obtain the number of interactions
            # inserted for each interaction target
        elif self.insert_mode == 'rel':
            insert_num_mat = (item_seq_len * self.insert_ratio).long()  # Relative insertion ratio
            insert_num_mat = torch.min(insert_num_mat, left_seq_len)
        assert insert_num_mat is not None
        """
        Construct a new set of interaction sequences according to insert_num_mat and desc_ids_interval
        Among the new interaction sequences, for each sequence, insert the place_v placeholder at the position 
        where insert_num_mat selects the target quantity from the front to the back in desc_ids_interval
        """
        new_item_seq = []
        new_time_seq = []
        insert_place = []
        origin_item_seq = origin_item_seq.tolist()  # (batch_size,max_seq_len)
        origin_time_seq = origin_time_seq.tolist()
        seq_id_seq = seq_id_seq.tolist()
        for b, (inter_seq, inter_time_seq, seq_id) in enumerate(zip(origin_item_seq, origin_time_seq, seq_id_seq)):
            insert_num = insert_num_mat[b].item()
            insert_pos = desc_ids_interval[b][:insert_num].sort()[0].tolist()
            new_items = []
            new_times = []
            insert_marks = []
            if seq_id in self.model.no_aug_seq:
                new_items = inter_seq
                new_times = inter_time_seq
                insert_marks = [0] * self.insert_num
            else:
                prev_idx = insert_num
                for idx in insert_pos:
                    idx = idx + 1
                    if inter_seq[idx - 1] == 0:
                        insert_num_mat[b] -= 1
                        prev_idx -= 1
                        continue
                    new_items.extend(inter_seq[prev_idx:idx])
                    new_times.extend(inter_time_seq[prev_idx:idx])
                    t_left = inter_time_seq[idx - 1]
                    # t_right = inter_time_seq[idx].item() if idx < len(inter_time_seq) else t_left
                    t_right = inter_time_seq[idx]
                    place_t = (t_left + t_right) // 2
                    new_items.append(place_v)
                    new_times.append(place_t)
                    insert_marks.append(len(new_items) - 1)
                    prev_idx = idx

                new_items.extend(inter_seq[prev_idx:])
                new_times.extend(inter_time_seq[prev_idx:])
                padding_list = [0] * (self.insert_num - insert_num_mat[b].item())
                insert_marks = padding_list + insert_marks

            new_item_seq.append(new_items)
            new_time_seq.append(new_times)
            insert_place.append(insert_marks)

        new_item_seq = torch.tensor(new_item_seq, dtype=torch.long, device=self.model.device)
        new_time_seq = torch.tensor(new_time_seq, dtype=torch.long, device=self.model.device)
        insert_place = torch.tensor(insert_place, dtype=torch.long, device=self.model.device)
        # 0 indicates padding, as no enhancing element will be added at the very beginning of the sequence
        return new_item_seq, new_time_seq, insert_place, insert_num_mat

    def augment(self, interaction):
        new_item_seq, new_time_seq, insert_place, insert_num_mat = self.augment_interaction(interaction)
        bert_seq_output = self.model.bert_forward(new_item_seq)
        bert_index = insert_place.unsqueeze(-1).expand(-1, -1, self.model.hidden_size)
        # (batch,insert_num,hidden_size)
        insert_hidden = torch.gather(bert_seq_output, index=bert_index, dim=1)
        if self.uni_method == 'topk':
            all_items_scores = torch.matmul(insert_hidden,
                                            self.model.item_embedding.weight[1:self.model.n_items].transpose(0, 1))
            _, top_k_indices = torch.topk(all_items_scores, k=self.top_k, dim=-1)
            top_k_indices += 1
            # Obtain the list of scores and ids of the items with the highest similarity (batch,masked_len,top_k).
            # Since the similarity calculation ignores the 0 bit, each top_k index should be increased by 1
            x_tensors = self.model.item_embedding(top_k_indices)
            inspire_scores, seq_mask = self.model.calculate_all_inspire_score(x_tensors, new_item_seq, new_time_seq,
                                                                              insert_place, insert_place, self.top_k)
            # (batch,masked_len,top_k,seq_len)，(batch,masked_len)
            native_scores = self.model.calculate_native_scores(insert_place, bert_seq_output, new_item_seq, self.top_k)
            # (batch,masked_len,top_k,seq_len)
            hawkes_scores = (native_scores * inspire_scores).sum(dim=-1)
            """
            Find the id of the item with the highest score at each position through the maximum value,
            and then extract the corresponding row and column table and values 
            through seq_mask and fill them into the sequence
            """
            _, sorted_ids = torch.sort(hawkes_scores, dim=-1, descending=True)
            sorted_items = torch.gather(top_k_indices, index=sorted_ids, dim=-1)
            # (batch,masked_len,top_k)
            insert_items = sorted_items[:, :, 0:1].squeeze(-1)  # (batch,masked_len)
            insert_items = insert_items[seq_mask]
            row_ids = (torch.arange(new_item_seq.size(0), dtype=torch.long, device=self.model.device)
                       .unsqueeze(1).expand(-1, insert_place.size(-1)))
            row_ids = row_ids[seq_mask]
            col_ids = insert_place[seq_mask]
            new_item_seq[row_ids, col_ids] = insert_items
        return new_item_seq, new_time_seq
