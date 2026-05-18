import torch


class DataAugmentation():
    def __init__(self, config, model):
        self.data_augment = config.data_augment
        self.model = model

    def augment(self, interaction):
        """
        传入recbole一个batch的数据类型，对这个batch的所有数据进行增强，然后返回增强后的项目序列和时间序列
        Args:
            interaction: 一个batch下的所有用户交互数据

        Returns:
        new_item_seq(batch,max_seq_len)增强后的用户交互序列,
        new_time_seq(batch,max_seq_len)增强后的用户时间序列

        """
        raise NotImplementedError("你没有指定明确的数据增强方法")

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
            # 计算理论插入数量
            num_insert = int(seq_len * self.insert_ratio)
            if num_insert < 1:
                new_insert_positions.append([0] * max_insert_len)
                new_item_seq[i] = origin_item_seq[i]
                new_time_seq[i] = origin_time_seq[i]
                continue

            # 不能超过 max_len
            num_insert = min(num_insert, max_len - seq_len)
            cur_idx = max_len - (num_insert + seq_len)  # 新的序列的开始索引
            if cur_idx > 0:
                new_time_seq[i, 0:cur_idx] = origin_time_seq[i, 0]
            #获得时间间隔序列计算间隔最大的位置
            time_intervals_seq = origin_time_seq[i, max_len-seq_len+1:] - origin_time_seq[i, max_len-seq_len:-1]  #(max_seq_len)表示i与i+1的时间间隔
            _, insert_positions = torch.sort(time_intervals_seq, dim=-1, descending=True)
            insert_positions = insert_positions + (max_len-seq_len)
            insert_positions = insert_positions[:num_insert]
            # 从小到大排序，保证插入时右移不打乱后面插入
            insert_positions, _ = torch.sort(insert_positions)  # 要在哪些元素的后面插入，对应的原始序列的索引
            insert_idx[i, max_insert_len - num_insert:] = insert_positions
            pre_insert_pos = max_len - seq_len - 1
            # 遍历插入位置（注意每次插入后 seq_len 要更新 +1）
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
        real_pos_mask = insert_idx != (max_len - 1)  # (batch,max_insert_len)True为真实的插入位置
        insert_ner_ids = torch.gather(origin_item_seq, dim=1, index=insert_idx)  # (batch,max_insert_len)插入位置的前一项邻居的id
        insert_ner_embeds = self.model.item_embedding(insert_ner_ids)  # (batch,max_insert_len,hidden_size)
        similar_scores = torch.matmul(insert_ner_embeds,
                                      self.model.item_embedding.weight.unsqueeze(0).transpose(-2, -1))
        # (batch,max_insert_len,item_n)获得每个插入位置的所有项目的得分
        sim_rows = torch.arange(batch_size, device=dev).unsqueeze(1).expand(-1,
                                                                            max_insert_len)  # (batch,max_insert_len)
        pos_idx = torch.arange(max_insert_len, device=dev).unsqueeze(0).expand(batch_size, -1)
        # (batch,max_insert_len,item_n)
        similar_scores[sim_rows, pos_idx, insert_ner_ids] = float('-inf')
        # 选出来每个位置的得分最高的那个，得到一个(batch,max_insert_len)的，存储插入项目的
        _, indices = torch.max(similar_scores, dim=-1)  # (batch,max_insert_len)
        insert_values = indices[real_pos_mask]  # 选出来要插入的值
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
            # 计算理论插入数量
            num_insert = int(seq_len * self.insert_ratio)
            if num_insert < 1:
                new_insert_positions.append([0] * max_insert_len)
                new_item_seq[i] = origin_item_seq[i]
                new_time_seq[i] = origin_time_seq[i]
                continue

            # 不能超过 max_len
            num_insert = min(num_insert, max_len - seq_len)
            cur_idx = max_len - (num_insert + seq_len)  #新的序列的开始索引
            if cur_idx > 0:
                new_time_seq[i, 0:cur_idx] = origin_time_seq[i, 0]
            insert_positions = torch.randint(low=max_len - seq_len, high=max_len - 1, size=(num_insert,), device=dev)
            # 从小到大排序，保证插入时右移不打乱后面插入
            insert_positions, _ = torch.sort(insert_positions)  #要在哪些元素的后面插入，对应的原始序列的索引
            insert_idx[i, max_insert_len - num_insert:] = insert_positions
            pre_insert_pos = max_len - seq_len - 1
            # 遍历插入位置（注意每次插入后 seq_len 要更新 +1）
            new_insert_pos = []
            for insert_pos in insert_positions:
                #构建新的序列
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
        real_pos_mask = insert_idx != (max_len - 1)  #(batch,max_insert_len)True为真实的插入位置
        insert_ner_ids = torch.gather(origin_item_seq, dim=1, index=insert_idx)  #(batch,max_insert_len)插入位置的前一项邻居的id
        insert_ner_embeds = self.model.item_embedding(insert_ner_ids)  #(batch,max_insert_len,hidden_size)
        similar_scores = torch.matmul(insert_ner_embeds,
                                      self.model.item_embedding.weight.unsqueeze(0).transpose(-2, -1))
        #(batch,max_insert_len,item_n)获得每个插入位置的所有项目的得分
        sim_rows = torch.arange(batch_size, device=dev).unsqueeze(1).expand(-1, max_insert_len)  #(batch,max_insert_len)
        pos_idx = torch.arange(max_insert_len, device=dev).unsqueeze(0).expand(batch_size, -1)
        # (batch,max_insert_len,item_n)
        similar_scores[sim_rows, pos_idx, insert_ner_ids] = float('-inf')
        #选出来每个位置的得分最高的那个，得到一个(batch,max_insert_len)的，存储插入项目的
        _, indices = torch.max(similar_scores, dim=-1)  #(batch,max_insert_len)
        insert_values = indices[real_pos_mask]  #选出来要插入的值
        insert_row_ids = torch.arange(batch_size, device=dev).unsqueeze(1).expand(-1, max_insert_len)
        insert_row_ids = insert_row_ids[real_pos_mask]
        insert_col_ids = new_insert_positions[real_pos_mask]
        new_item_seq[insert_row_ids, insert_col_ids] = insert_values
        return new_item_seq, new_time_seq


class ReOrderAugmentation(DataAugmentation):
    def __init__(self, config, model):
        super(ReOrderAugmentation, self).__init__(config, model)
        self.mask_ratios = config['data_augments']['reorder_ratio']  # 用同样的比例区间控制片段长度

    def augment(self, interaction):
        origin_item_seq = interaction[self.model.ITEM_SEQ]  # (batch, max_len)
        origin_time_seq = interaction[self.model.TIME_SEQ]
        batch_size, max_len = origin_item_seq.shape
        dev = origin_item_seq.device

        # ====== 随机生成每个序列的片段长度 ======
        b, e = self.mask_ratios
        rand_ratios = (e - b) * torch.rand(batch_size, device=dev) + b
        seq_lens = interaction[self.model.ITEM_SEQ_LEN]  # (batch,)
        seg_lens = (seq_lens.float() * rand_ratios).long()  # 每个序列要打乱的长度

        # ====== 随机生成每个序列的片段起点 ======
        origin_starts = (max_len - seq_lens).long()  # 左边 padding 的起点
        max_starts = (max_len - seg_lens).long()  # 起点的上界
        rand = torch.rand(batch_size, device=dev)
        real_start_idx = (origin_starts + rand * (max_starts - origin_starts)).long()  # (batch,)

        # ====== 生成新序列 ======
        new_item_seq = origin_item_seq.clone()
        new_time_seq = origin_time_seq.clone()  # 时间保持不变

        for i in range(batch_size):
            seg_len = seg_lens[i].item()
            if seg_len > 1:  # 只在长度>1时打乱
                start = real_start_idx[i].item()
                end = start + seg_len

                # 打乱 item 子片段，保持时间不变
                sub_items = new_item_seq[i, start:end]
                perm = torch.randperm(seg_len, device=dev)
                new_item_seq[i, start:end] = sub_items[perm]

        return new_item_seq, new_time_seq


class MaskAugmentation(DataAugmentation):
    def __init__(self, config, model):
        super(MaskAugmentation, self).__init__(config, model)
        self.mask_ratios = config['data_augments']['mask_ratio']

    def augment(self, interaction):
        origin_item_seq = interaction[self.model.ITEM_SEQ]  # 原始的用户交互序列(batch_size,max_seq_len)
        origin_time_seq = interaction[self.model.TIME_SEQ]
        batch_size = origin_item_seq.shape[0]
        max_len = origin_item_seq.shape[1]
        cur_device = origin_item_seq.device
        b = self.mask_ratios[0]
        e = self.mask_ratios[1]
        rand_ratios = (e - b) * torch.rand(batch_size, device=cur_device) + b
        seq_lens = interaction[self.model.ITEM_SEQ_LEN]  # batch长度，每个交互序列的长度
        mask_len = torch.mul(seq_lens, rand_ratios).long()  # 每个序列要掩码的长度
        # 计算每个序列裁剪的索引开始位置
        origin_starts = max_len - seq_lens  # (batch,)  #索引最小值
        max_starts = max_len - mask_len  # (batch,)  索引最大值,可能会有max_len越界
        rand = torch.rand(batch_size, device=cur_device)
        real_start_idx = (origin_starts + rand * (max_starts - origin_starts)).int()  # 存储每个序列实际裁剪的索引的下标
        # 准备输出
        pad_token = self.model.n_items  # 根据你的数据情况修改
        new_item_seq = origin_item_seq.clone()
        new_time_seq = origin_time_seq.clone()
        #获得一个(batch,max_len)的索引矩阵
        arrange_ids = torch.arange(max_len, dtype=torch.long, device=cur_device).unsqueeze(0).expand(batch_size, -1)
        # 每个序列掩码开始的索引(batch_size,max_len)，包含这个位置左闭
        real_start_matrix = real_start_idx.unsqueeze(1).expand(batch_size, max_len)
        # 每个序列随机掩码结束的索引(batch_size,max_len)，不包含这个位置，右开
        real_end_matrix = (real_start_idx + mask_len).unsqueeze(1).expand(batch_size, max_len)
        token_mask = (arrange_ids >= real_start_matrix) & (arrange_ids < real_end_matrix)
        new_item_seq[token_mask] = pad_token
        return new_item_seq, new_time_seq


class CropAugmentation(DataAugmentation):
    def __init__(self, config, model):
        super(CropAugmentation, self).__init__(config, model)
        self.ratio_range = config['data_augments']['crop_ratio']

    def augment(self, interaction):
        origin_item_seq = interaction[self.model.ITEM_SEQ]  # 原始的用户交互序列(batch_size,max_seq_len)
        origin_time_seq = interaction[self.model.TIME_SEQ]
        batch_size = origin_item_seq.shape[0]
        max_len = origin_item_seq.shape[1]
        cur_device = origin_item_seq.device
        b = self.ratio_range[0]
        e = self.ratio_range[1]
        rand_ratios = (e - b) * torch.rand(batch_size, device=cur_device) + b
        seq_lens = interaction[self.model.ITEM_SEQ_LEN]  #batch长度，每个交互序列的长度
        crop_len = torch.clamp_min(torch.mul(seq_lens, rand_ratios).long(), 1)  #每个序列要裁剪的长度
        #计算每个序列裁剪的索引开始位置
        origin_starts = max_len - seq_lens  #(batch,)  #索引最小值
        max_starts = max_len - crop_len  #(batch,)  索引最大值
        rand = torch.rand(batch_size, device=cur_device)
        real_start_idx = (origin_starts + rand * (max_starts - origin_starts)).int()  #存储每个序列实际裁剪的索引的下标
        # 准备输出
        pad_token = 0  # 根据你的数据情况修改
        new_item_seq = torch.full_like(origin_item_seq, pad_token, device=cur_device)
        new_time_seq = torch.full_like(origin_time_seq, pad_token, device=cur_device)
        for i in range(batch_size):
            start = real_start_idx[i].item()
            end = start + crop_len[i].item()
            sub_item = origin_item_seq[i, start:end]
            sub_time = origin_time_seq[i, start:end]

            # 放到右侧
            new_item_seq[i, -crop_len[i]:] = sub_item

            # 左侧 pad 使用裁剪序列最前面的时间
            left_pad_len = max_len - crop_len[i]
            if left_pad_len > 0:
                new_time_seq[i, :left_pad_len] = sub_time[0]
            new_time_seq[i, -crop_len[i]:] = sub_time
        return new_item_seq, new_time_seq


class NoDataAugmentation(DataAugmentation):
    """
    没有数据增强
    """

    def __init__(self, config, model):
        super(NoDataAugmentation, self).__init__(config, model)

    def augment(self, interaction):
        origin_item_seq = interaction[self.model.ITEM_SEQ]  # 原始的用户交互序列(batch_size,max_seq_len)
        origin_time_seq = interaction[self.model.TIME_SEQ]
        return origin_item_seq, origin_time_seq


class HawkesInsertAugmentation(DataAugmentation):
    """
    对进行预测的序列进行霍克斯插入操作
    """

    def __init__(self, config, model):
        super(HawkesInsertAugmentation, self).__init__(config, model)
        # 设置均匀化生成器的参数
        self.insert_ratio = config['generator_args']['insert_ratio']
        self.insert_num = int(self.model.max_seq_length * self.insert_ratio)
        self.top_k = config['generator_args']['top_k']
        self.uni_method = config['uni_method']
        self.insert_mode = config['insert_mode']

    def augment_interaction(self, interaction):
        """
        按照用户的原始交互序列，对指定个数的最大时间间隔的位置进行插入掩码操作，生成准备插入的占位序列
        只对非padding为止进行掩码，避免引入噪声
        Args:
        Returns (tuple): item_seq (batch_size,max_seq) 拆入占位符的用户序列 time_seq (batch_size,max_seq) 增强后的用户时间序列
        insert_place(batch,insert_num) 标记每个用户哪些位置需要插入 insert_num_mat(batch) 标记用户序列实际插入了几个
            interaction: 用户交互数据
        """
        item_seq_len = interaction[self.model.ITEM_SEQ_LEN]
        origin_item_seq = interaction[self.model.ITEM_SEQ]  #原始的用户交互序列(batch_size,max_seq_len)
        origin_time_seq = interaction[self.model.TIME_SEQ]  #原始的用户交互的时间戳序列
        seq_id_seq = interaction[self.model.SEQ_ID_LIST]
        neighbor_interval_mat = origin_time_seq[:, 1:] - origin_time_seq[:, :-1]  #临近时间差值矩阵(batch_size,max_len-1)
        desc_ids_interval = torch.argsort(neighbor_interval_mat, dim=1, descending=True)
        #根绝时间间隔大小排序的时间差矩阵的索引矩阵,（batch_size,max_seq_len—1）
        place_v = self.model.n_items  #用来放在未来插入增强交互位置的占位元素

        #计算每个序列目标插入几个元素
        left_seq_len = self.model.max_seq_length - item_seq_len
        # """
        # 要是改成每个序列自己的比例插入，就在这里把self.insert_num改成一个(512)的向量，每个序列自己插入数量
        # """
        #
        insert_num_mat = None
        if self.insert_mode == 'abs':
            insert_num_mat = torch.clamp_max(left_seq_len, self.insert_num)  #获得每个交互目标插入的交互数目
        elif self.insert_mode == 'rel':
            insert_num_mat = (item_seq_len * self.insert_ratio).long()  #相对插入比例
            insert_num_mat = torch.min(insert_num_mat, left_seq_len)
        assert insert_num_mat is not None
        #为了防止如果原始序列的有效长度已经到达了最大长度，或者是相对插入的长度超过了序列的可以拆入的长度
        #所以要进行一次大小比较，用当前计算出的相对插入数量和序列的可插入数量做对比，确保相对插入小于等于可插入数量

        """
        按照insert_num_mat、desc_ids_interval构建新的交互序列集合
        其中新的交互序列，每个序列按照insert_num_mat在desc_ids_interval中从前面向后选取目标数量的位置，插入place_v占位符
        """
        new_item_seq = []
        new_time_seq = []
        insert_place = []
        origin_item_seq = origin_item_seq.tolist()  # 原始的用户交互序列(batch_size,max_seq_len)
        origin_time_seq = origin_time_seq.tolist()  # 原始的用户交互的时间戳序列
        seq_id_seq = seq_id_seq.tolist()
        for b, (inter_seq, inter_time_seq, seq_id) in enumerate(zip(origin_item_seq, origin_time_seq, seq_id_seq)):
            insert_num = insert_num_mat[b].item()
            insert_pos = desc_ids_interval[b][:insert_num].sort()[0].tolist()
            new_items = []
            new_times = []
            insert_marks = []
            if seq_id in self.model.no_aug_seq:  #如果是不需要数据增强的序列，就将所有的新序列都设为原来的。插入的坐标都是0的填充位，让后面的数据增强不对0位置进行插入
                new_items = inter_seq
                new_times = inter_time_seq
                insert_marks = [0] * self.insert_num
            else:
                prev_idx = insert_num  # 记录上一次插入的元素在原始的交互序列的位置，也是下一次插入需要前面补充的原始序列的开始索引，这里应该初始化为插入数目
                #这里初始化prev_idx为insert_num的原因是，insert_num的值比padding的数量少，为了能够插入足够数量的元素
                #直接舍弃掉前面insert_num的元素，不会影响原始序列的有效项目
                for idx in insert_pos:  #从索引顺序的从小到大排序
                    idx = idx + 1  # 插入在间隔的右边
                    if inter_seq[idx - 1] == 0:  # 跳过padding的插入位，避免加入噪声
                        insert_num_mat[b] -= 1
                        prev_idx -= 1
                        continue
                    new_items.extend(inter_seq[prev_idx:idx])
                    new_times.extend(inter_time_seq[prev_idx:idx])
                    # 插入 place_v，时间是两端平均
                    t_left = inter_time_seq[idx - 1]
                    # t_right = inter_time_seq[idx].item() if idx < len(inter_time_seq) else t_left
                    t_right = inter_time_seq[idx]  # 不需要检查越界，因为idx+1的最大取值不会越界
                    place_t = (t_left + t_right) // 2
                    new_items.append(place_v)
                    new_times.append(place_t)
                    insert_marks.append(len(new_items) - 1)  # 插入点位置索引
                    prev_idx = idx

                # 剩下尾部原始交互
                new_items.extend(inter_seq[prev_idx:])
                new_times.extend(inter_time_seq[prev_idx:])
                padding_list = [0] * (self.insert_num - insert_num_mat[b].item())
                insert_marks = padding_list + insert_marks

            #将当前交互序列的相关增强内容加入
            new_item_seq.append(new_items)
            new_time_seq.append(new_times)
            insert_place.append(insert_marks)

        #将所有的结果转化为tensor
        new_item_seq = torch.tensor(new_item_seq, dtype=torch.long, device=self.model.device)
        new_time_seq = torch.tensor(new_time_seq, dtype=torch.long, device=self.model.device)
        insert_place = torch.tensor(insert_place, dtype=torch.long, device=self.model.device)  #0代表是填充，因为不会在序列的最前面添加增强元素
        return new_item_seq, new_time_seq, insert_place, insert_num_mat

    def augment(self, interaction):
        new_item_seq, new_time_seq, insert_place, insert_num_mat = self.augment_interaction(interaction)
        """
        通过bert_forward对插入占位后的序列进行双向编码，根据拆入下标矩阵提取出掩码位置的隐藏表示矩阵
        ，然后做全item点积，取出每个位置的topk，计算霍克斯强度，然后再取得分最高的元素插入原始序列
        """
        """
        TODO:self.model是模型，可以通过self.item_embedding获得嵌入层，
        new_item_seq 待插入增强的序列，其中0表示padding
        self.model.n_items  用来放在未来插入增强交互位置的占位元素
        """
        bert_seq_output = self.model.bert_forward(new_item_seq)  # 返回的是掩码序列的隐藏更表示(batch,seq_len,hidden_size)
        bert_index = insert_place.unsqueeze(-1).expand(-1, -1, self.model.hidden_size)
        # (batch,insert_num,hidden_size)提取插入位置隐藏表示的索引张量
        insert_hidden = torch.gather(bert_seq_output, index=bert_index, dim=1)
        if self.uni_method == 'topk':
            all_items_scores = torch.matmul(insert_hidden,
                                            self.model.item_embedding.weight[1:self.model.n_items].transpose(0, 1))
            # (batch,masked_len,n_items-1) 忽略了0和n_items的相似度，因为不需要推荐他们
            _, top_k_indices = torch.topk(all_items_scores, k=self.top_k, dim=-1)
            top_k_indices += 1
            # 获得(batch,masked_len,top_k)的相似度最高的项目的得分和id列表,因为计算相似度忽略了0位，所以每一个top_k索引都应该+1
            x_tensors = self.model.item_embedding(top_k_indices)
            inspire_scores, seq_mask = self.model.calculate_all_inspire_score(x_tensors, new_item_seq, new_time_seq,
                                                                              insert_place, insert_place, self.top_k)
            # (batch,masked_len,top_k,seq_len)，(batch,masked_len)
            native_scores = self.model.calculate_native_scores(insert_place, bert_seq_output, new_item_seq, self.top_k)
            # (batch,masked_len,top_k,seq_len)
            hawkes_scores = (native_scores * inspire_scores).sum(dim=-1)  # 逐点加权求和，得到(batch,masked_len,top_k)
            """
            通过最大值找到每个位置得分最高的项目的id，然后通过seq_mask取出对应的行列下表和值，填充序列中
            """
            _, sorted_ids = torch.sort(hawkes_scores, dim=-1, descending=True)
            sorted_items = torch.gather(top_k_indices, index=sorted_ids, dim=-1)
            # (batch,masked_len,top_k)按照最终得分排序后的每个掩码位置的项目列表
            insert_items = sorted_items[:, :, 0:1].squeeze(-1)  # (batch,masked_len)每个交互每个位置要插入的元素的张量
            insert_items = insert_items[seq_mask]
            row_ids = (torch.arange(new_item_seq.size(0), dtype=torch.long, device=self.model.device)
                       .unsqueeze(1).expand(-1, insert_place.size(-1)))
            row_ids = row_ids[seq_mask]
            col_ids = insert_place[seq_mask]
            new_item_seq[row_ids, col_ids] = insert_items
        return new_item_seq, new_time_seq
