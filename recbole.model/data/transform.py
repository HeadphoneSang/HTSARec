# -*- coding: utf-8 -*-
# @Time   : 2022/7/19
# @Author : Gaowei Zhang
# @Email  : zgw15630559577@163.com

import math
import time

import numpy as np
import random
import torch
from copy import deepcopy
from recbole.data.interaction import Interaction, cat_interactions


def construct_transform(config):
    """
    Transformation for batch data.
    """
    if config["transform"] is None:
        return Equal(config)
    else:
        str2transform = {
            "mask_itemseq": MaskItemSequence,
            "inverse_itemseq": InverseItemSequence,
            "crop_itemseq": CropItemSequence,
            "reorder_itemseq": ReorderItemSequence,
            "user_defined": UserDefinedTransform,
            "mask_sample_itemseq": MaskItemSequenceSampleNeg,
        }
        if config["transform"] not in str2transform:
            raise NotImplementedError(
                f"There is no transform named '{config['transform']}'"
            )

        return str2transform[config["transform"]](config)


class Equal:
    def __init__(self, config):
        pass

    def __call__(self, dataset, interaction):
        return interaction


class MaskItemSequence:
    """
    Mask item sequence for training.
    """

    def __init__(self, config):
        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.MASK_ITEM_SEQ = "Mask_" + self.ITEM_SEQ
        self.POS_ITEMS = "Pos_" + config["ITEM_ID_FIELD"]
        self.NEG_ITEMS = "Neg_" + config["ITEM_ID_FIELD"]
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.mask_ratio = config["mask_ratio"]
        self.ft_ratio = 0 if not hasattr(config, "ft_ratio") else config["ft_ratio"]
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)
        self.MASK_INDEX = "MASK_INDEX"
        config["MASK_INDEX"] = "MASK_INDEX"
        config["MASK_ITEM_SEQ"] = self.MASK_ITEM_SEQ
        config["POS_ITEMS"] = self.POS_ITEMS
        config["NEG_ITEMS"] = self.NEG_ITEMS
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.config = config

    def _neg_sample(self, item_set, n_items):
        item = random.randint(1, n_items - 1)
        while item in item_set:
            item = random.randint(1, n_items - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence  #就是将每个序列填充到统一大小，从前面用0填充，如果少的填充，多的截断，从前面截断
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence

    def _append_mask_last(self, interaction, n_items, device):
        batch_size = interaction[self.ITEM_SEQ].size(0)
        pos_items, neg_items, masked_index, masked_item_sequence = [], [], [], []
        seq_instance = interaction[self.ITEM_SEQ].cpu().numpy().tolist()
        item_seq_len = interaction[self.ITEM_SEQ_LEN].cpu().numpy().tolist()
        for instance, lens in zip(seq_instance, item_seq_len):
            mask_seq = instance.copy()
            ext = instance[lens - 1]
            mask_seq[lens - 1] = n_items
            masked_item_sequence.append(mask_seq)
            pos_items.append(self._padding_sequence([ext], self.mask_item_length))
            neg_items.append(
                self._padding_sequence(
                    [self._neg_sample(instance, n_items)], self.mask_item_length
                )
            )
            masked_index.append(
                self._padding_sequence([lens - 1], self.mask_item_length)
            )
        # [B Len]
        masked_item_sequence = torch.tensor(
            masked_item_sequence, dtype=torch.long, device=device
        ).view(batch_size, -1)
        # [B mask_len]
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(
            batch_size, -1
        )
        # [B mask_len]
        neg_items = torch.tensor(neg_items, dtype=torch.long, device=device)
        # [B mask_len]
        masked_index = torch.tensor(masked_index, dtype=torch.long, device=device).view(
            batch_size, -1
        )
        new_dict = {
            self.MASK_ITEM_SEQ: masked_item_sequence,
            self.POS_ITEMS: pos_items,
            self.NEG_ITEMS: neg_items,
            self.MASK_INDEX: masked_index,
        }
        ft_interaction = deepcopy(interaction)
        ft_interaction.update(Interaction(new_dict))
        return ft_interaction

    def __call__(self, dataset, interaction):
        """
        对原始的交互序列进行随机的掩码操作
        Args:
            dataset: 对应的数据集，比如是训练集，验证机或者测试集
            interaction: 原始的一个batch的交互

        Returns: 返回处理后的交互，包含了掩码交互序列，掩码位置，掩码项目，掩码负采样项目

        """
        item_seq = interaction[self.ITEM_SEQ]
        device = item_seq.device
        batch_size = item_seq.size(0)
        n_items = dataset.num(self.ITEM_ID)
        sequence_instances = item_seq.cpu().numpy().tolist()
        #获得对应批次的所有用户交互序列的list列表矩阵

        # Masked Item Prediction
        # [B * Len]
        masked_item_sequence = []  #存放每一个被掩码后的序列，形状和输入的item_seq相等。里面就是标记了掩码的位置
        pos_items = []  #(batch,mask_item_len) 存放每个序列被掩码的项目的id
        neg_items = []  #(batch,mask_item_len) 存放每个序列被掩码的项目的下标
        masked_index = []  #(batch,mask_item_len) 存放每个序列被掩码的项目的负采样样本的id

        if random.random() < self.ft_ratio:  #只在最后一位加掩码，我用不到，可以注释掉
            interaction = self._append_mask_last(interaction, n_items, device)
        else:
            for instance in sequence_instances:  #便利整个batch，对每一个交互序列进行处理
                # WE MUST USE 'copy()' HERE!
                masked_sequence = instance.copy()
                pos_item = []
                neg_item = []
                index_ids = []
                for index_id, item in enumerate(instance):
                    # padding is 0, the sequence is end
                    if item == 0:  #这里我需要改改，我是从前面padding的，应该是先略过0，然后才开始操作
                        break
                    prob = random.random()
                    if prob < self.mask_ratio:
                        """
                        通过随机数，对每一个位置进行掩码与否的操作
                        如果掩码:
                        先将当前位置的原始交互项目id放进pos_item的list中
                        然后对当前位置进行负采样，将负采样的数据放进neg_item的list中
                        并且将当前位置设置为掩码占位符，并且将当前位置的下标记录到一个临时的index_ids的list里面
                        """
                        pos_item.append(item)
                        neg_item.append(self._neg_sample(instance, n_items))  #这里采样一个负样本，可以通过一个超惨控制
                        masked_sequence[index_id] = n_items  #将当前位置替换为掩码占位符
                        index_ids.append(index_id)

                masked_item_sequence.append(masked_sequence)

                #下面三个向量都是[mask_item_len,]对应位置分别代表，掩码的下标，掩码的项目和负采样的项目
                pos_items.append(
                    self._padding_sequence(pos_item, self.mask_item_length)
                )  #记录掩码的位置原来的项目id，并且统一处理为最大掩码长度，多的话就舍弃前面部分的掩码
                neg_items.append(
                    self._padding_sequence(neg_item, self.mask_item_length)
                )  #和上面相同，就是记录的负采样样本
                masked_index.append(
                    self._padding_sequence(index_ids, self.mask_item_length)
                )  #和上面一样，就是记录的掩码的下标的位置

            # [B Len]
            masked_item_sequence = torch.tensor(
                masked_item_sequence, dtype=torch.long, device=device
            ).view(batch_size, -1)
            # [B mask_len]
            pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(
                batch_size, -1
            )
            # [B mask_len]
            neg_items = torch.tensor(neg_items, dtype=torch.long, device=device).view(
                batch_size, -1
            )
            # [B mask_len]
            masked_index = torch.tensor(
                masked_index, dtype=torch.long, device=device
            ).view(batch_size, -1)
            new_dict = {
                self.MASK_ITEM_SEQ: masked_item_sequence,
                self.POS_ITEMS: pos_items,
                self.NEG_ITEMS: neg_items,
                self.MASK_INDEX: masked_index,
            }
            interaction.update(Interaction(new_dict))  #将新构建的这些属性追加到原始的interaction后面，不改变原来的已有的内容
        return interaction


class MaskItemSequenceSampleNeg(MaskItemSequence):

    def __init__(self, config):
        self.item_list = None
        config["mask_ratio"] = config["mask_seq_args"]["mask_ratio"]
        config["ft_ratio"] = config["mask_seq_args"]["ft_ratio"]
        super().__init__(config)
        self.config = config
        self.neg_sample_num = config["mask_seq_args"]["neg_sample_num"]

    def _padding_neg_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        padding_v = [0] * self.neg_sample_num
        sequence = [padding_v] * pad_len + sequence  #就是将每个序列填充到统一大小，从前面用0填充，如果少的填充，多的截断，从前面截断
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence

    def get_negative_candidates(self, item_seq, all_items):
        """
        返回不包含item_seq的item_id集合
        Args:
            item_seq:  用户交互列表
            all_items:  所有项目

        Returns: 不包含用户交互的item列表

        """
        return all_items
        # return list(all_items - set(item_seq))

    import numpy as np

    def n_neg_sample(self, neg_item_list):
        return np.random.choice(neg_item_list, size=self.neg_sample_num, replace=False)

    def __call__(self, dataset, interaction):
        if self.config["data_augment"] != "hawkes_insert":
            return interaction
        """
        对原始的交互序列进行随机的掩码操作
        Args:
            dataset: 对应的数据集，比如是训练集，验证机或者测试集
            interaction: 原始的一个batch的交互

        Returns: 返回处理后的交互，包含了掩码交互序列，掩码位置，掩码项目，掩码负采样项目

        """

        item_seq = interaction[self.ITEM_SEQ]  #(batch,seq_len)
        device = item_seq.device
        batch_size = item_seq.size(0)
        n_items = dataset.num(self.ITEM_ID)
        sequence_instances = item_seq.cpu().numpy().tolist()

        #获得每个用户的负采样候选列表掩码
        if self.item_list is None:
            self.item_list = torch.arange(1, n_items, device=device, dtype=torch.long)
        pos_mask = (item_seq.unsqueeze(2) == self.item_list.unsqueeze(0).unsqueeze(1)).any(dim=1)
        neg_can_mask = ~pos_mask

        # 获得对应批次的所有用户交互序列的list列表矩阵
        masked_item_sequence = []  # 存放每一个被掩码后的序列，形状和输入的item_seq相等。里面就是标记了掩码的位置
        pos_items = []  # (batch,mask_item_len) 存放每个序列被掩码的项目的id
        neg_n = self.mask_item_length * self.neg_sample_num  #每个用户的采样数目
        neg_items = torch.zeros(batch_size, neg_n, dtype=torch.long, device=device)
        masked_index = []  # (batch,mask_item_len) 存放每个序列被掩码的项目的负采样样本的id
        index_s = np.arange(self.max_seq_length)
        # neg_can_list = interaction['neg_can_list'].cpu().numpy().tolist()
        for batch_id, instance in enumerate(sequence_instances):  # 便利整个batch，对每一个交互序列进行处理
            #为当前交互序列创建掩码负采样列表
            neg_items_list = self.item_list[neg_can_mask[batch_id]]
            idx = torch.randint(0, len(neg_items_list), (neg_n,))
            neg_items[batch_id] = neg_items_list[idx]

            #创建掩码序列和掩码序列下标列表
            padding_mask = np.array(instance) != 0
            line_mask = (np.random.uniform(0, 1, size=self.max_seq_length) < self.mask_ratio) & padding_mask
            masked_sequence = np.array(instance)
            pos_item = masked_sequence[line_mask].tolist()
            index_ids = index_s[line_mask].tolist()
            masked_sequence[line_mask] = n_items
            masked_sequence = masked_sequence.tolist()
            masked_item_sequence.append(masked_sequence)
            # 下面三个向量都是[mask_item_len,]对应位置分别代表，掩码的下标，掩码的项目和负采样的项目
            pos_items.append(
                self._padding_sequence(pos_item, self.mask_item_length)
            )  # 记录掩码的位置原来的项目id，并且统一处理为最大掩码长度，多的话就舍弃前面部分的掩码
            masked_index.append(
                self._padding_sequence(index_ids, self.mask_item_length)
            )  # 和上面一样，就是记录的掩码的下标的位置

        # [B Len]
        masked_item_sequence = torch.tensor(
            masked_item_sequence, dtype=torch.long, device=device
        )
        # [B mask_len]
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device)
        # [B mask_len]
        neg_items = neg_items.view(batch_size, self.mask_item_length, -1)
        # [B mask_len]
        masked_index = torch.tensor(
            masked_index, dtype=torch.long, device=device
        )
        new_dict = {
            self.MASK_ITEM_SEQ: masked_item_sequence,
            self.POS_ITEMS: pos_items,
            self.NEG_ITEMS: neg_items,
            self.MASK_INDEX: masked_index,
        }
        interaction.update(Interaction(new_dict))  # 将新构建的这些属性追加到原始的interaction后面，不改变原来的已有的内容
        return interaction

    def set_random_sample(self, s, k):
        """
        蓄水池负采样
        Args:
            k: 负采样的数目
            s: 负采样的样本集合set
        Returns: 返回负采样的集合

        """
        assert k <= len(s)
        result = []
        it = iter(s)
        for _ in range(k):
            result.append(next(it))  # 预先抽样第 k 个
        for i, val in enumerate(it, k):
            j = random.randint(0, i)
            if j < k:
                result[j] = val
        return result

    def neg_filter_sample(self, filtered_item_set, sample_num, min_id, max_id):
        """
        进行一次指定数量的负采样，目标不包含filtered_item
        这里采用随机采样后进行决绝策略判断的方法
        最好情况时间复杂度是O(sample_num)
        Args:
            filtered_item_set: 正样本set，要拒绝的样本
            sample_num: 采样的数量
            min_id: 随机采样的最小值
            max_id: 随机采样的最大值

        Returns: 返回一个目标大小的id的list

        """
        assert max_id - min_id + 1 - len(filtered_item_set) >= sample_num, \
            "采样空间太小，无法生成足够负样本"
        rand_ids = random.sample(range(min_id, max_id + 1), sample_num)
        sampled_set = set()
        for i, rand_id in enumerate(rand_ids):  #或是构建一个set()用来记录当前的采样，如果有重的就重新采样
            while rand_id in filtered_item_set or rand_id in sampled_set:
                rand_id = random.randint(min_id, max_id)
            rand_ids[i] = rand_id
            sampled_set.add(rand_id)
        return rand_ids

    def __call0__(self, dataset, interaction):
        """
        对原始的交互序列进行随机的掩码操作
        1.对于每个序列，随机选mask_len * neg_n个样本，
        然后拿每个位置的样本和每个位置的正样本比较，
        最好的情况是O(mask_len*neg_n)的时间复杂度，最坏是O(mask_len*n*neg_n)
        Args:
            dataset: 对应的数据集，比如是训练集，验证机或者测试集
            interaction: 原始的一个batch的交互

        Returns: 返回处理后的交互，包含了掩码交互序列，掩码位置，掩码项目，掩码负采样项目

        """

        item_seq = interaction[self.ITEM_SEQ]  #(batch,seq_len)
        device = item_seq.device
        batch_size = item_seq.size(0)
        n_items = dataset.num(self.ITEM_ID)
        sequence_instances = item_seq.cpu().numpy().tolist()
        # 获得对应批次的所有用户交互序列的list列表矩阵
        masked_item_sequence = []  # 存放每一个被掩码后的序列，形状和输入的item_seq相等。里面就是标记了掩码的位置
        pos_items = []  # (batch,mask_item_len) 存放每个序列被掩码的项目的id
        neg_items = []
        masked_index = []  # (batch,mask_item_len) 存放每个序列被掩码的下标
        index_s = np.arange(self.max_seq_length)
        neg_n = self.mask_item_length * self.neg_sample_num  # 每个用户的目标负采样数目
        for batch_id, instance in enumerate(sequence_instances):  # 便利整个batch，对每一个交互序列进行处理
            #创建掩码序列和掩码序列下标列表
            padding_mask = np.array(instance) != 0  #标记出非掩码位置的元素为True
            line_mask = (np.random.uniform(0, 1, size=self.max_seq_length) < self.mask_ratio) & padding_mask
            #生成非padding位的随机掩码的掩码向量
            masked_sequence = np.array(instance)
            pos_item = masked_sequence[line_mask].tolist()  #被掩码位置的正样本id的list
            index_ids = index_s[line_mask].tolist()
            masked_sequence[line_mask] = n_items
            masked_sequence = masked_sequence.tolist()
            masked_item_sequence.append(masked_sequence)

            #对每个掩码位置进行负采样
            neg_list = self.neg_filter_sample(set(instance), neg_n, 1, dataset.item_num - 1)
            # for cur_pos_item in pos_item:
            #     cur_k_negs = self.neg_filter_sample(cur_pos_item, self.neg_sample_num, 1, dataset.item_num - 1)
            #     neg_list.extend(cur_k_negs)
            # 下面三个向量都是[mask_item_len,]对应位置分别代表，掩码的下标，掩码的项目和负采样的项目
            neg_items.append(
                self._padding_sequence(neg_list, neg_n)
            )
            pos_items.append(
                self._padding_sequence(pos_item, self.mask_item_length)
            )  # 记录掩码的位置原来的项目id，并且统一处理为最大掩码长度，多的话就舍弃前面部分的掩码
            masked_index.append(
                self._padding_sequence(index_ids, self.mask_item_length)
            )  # 和上面一样，就是记录的掩码的下标的位置

        # [B Len]
        masked_item_sequence = torch.tensor(
            masked_item_sequence, dtype=torch.long, device=device
        )
        # [B mask_len]
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device)
        # [B mask_len]
        neg_items = torch.tensor(neg_items, dtype=torch.long, device=device).view(batch_size, self.mask_item_length, -1)
        # [B mask_len]
        masked_index = torch.tensor(
            masked_index, dtype=torch.long, device=device
        )
        new_dict = {
            self.MASK_ITEM_SEQ: masked_item_sequence,
            self.POS_ITEMS: pos_items,
            self.NEG_ITEMS: neg_items,
            self.MASK_INDEX: masked_index,
        }
        interaction.update(Interaction(new_dict))  # 将新构建的这些属性追加到原始的interaction后面，不改变原来的已有的内容
        return interaction


class InverseItemSequence:
    """
    inverse the seq_item, like this
        [1,2,3,0,0,0,0] -- after inverse -->> [0,0,0,0,1,2,3]
    """

    def __init__(self, config):
        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.INVERSE_ITEM_SEQ = "Inverse_" + self.ITEM_SEQ
        config["INVERSE_ITEM_SEQ"] = self.INVERSE_ITEM_SEQ

    def __call__(self, dataset, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        device = item_seq.device
        item_seq = item_seq.cpu().numpy()
        item_seq_len = item_seq_len.cpu().numpy()
        new_item_seq = []
        for items, length in zip(item_seq, item_seq_len):
            item = list(items[:length])
            zeros = list(items[length:])
            seqs = zeros + item
            new_item_seq.append(seqs)
        inverse_item_seq = torch.tensor(new_item_seq, dtype=torch.long, device=device)
        new_dict = {self.INVERSE_ITEM_SEQ: inverse_item_seq}
        interaction.update(Interaction(new_dict))
        return interaction


class CropItemSequence:
    """
    Random crop for item sequence.
    """

    def __init__(self, config):
        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
        self.CROP_ITEM_SEQ = "Crop_" + self.ITEM_SEQ
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.CROP_ITEM_SEQ_LEN = self.CROP_ITEM_SEQ + self.ITEM_SEQ_LEN
        self.crop_eta = config["eta"]
        config["CROP_ITEM_SEQ"] = self.CROP_ITEM_SEQ
        config["CROP_ITEM_SEQ_LEN"] = self.CROP_ITEM_SEQ_LEN

    def __call__(self, dataset, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        device = item_seq.device
        crop_item_seq_list, crop_item_seqlen_list = [], []

        for seq, length in zip(item_seq, item_seq_len):
            crop_len = math.floor(length * self.crop_eta)
            crop_begin = random.randint(0, length - crop_len)
            crop_item_seq = np.zeros(seq.shape[0])
            if crop_begin + crop_len < seq.shape[0]:
                crop_item_seq[:crop_len] = seq[crop_begin: crop_begin + crop_len]
            else:
                crop_item_seq[:crop_len] = seq[crop_begin:]
            crop_item_seq_list.append(
                torch.tensor(crop_item_seq, dtype=torch.long, device=device)
            )
            crop_item_seqlen_list.append(
                torch.tensor(crop_len, dtype=torch.long, device=device)
            )
        new_dict = {
            self.CROP_ITEM_SEQ: torch.stack(crop_item_seq_list),
            self.CROP_ITEM_SEQ_LEN: torch.stack(crop_item_seqlen_list),
        }
        interaction.update(Interaction(new_dict))
        return interaction


class ReorderItemSequence:
    """
    Reorder operation for item sequence.
    """

    def __init__(self, config):
        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
        self.REORDER_ITEM_SEQ = "Reorder_" + self.ITEM_SEQ
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.reorder_beta = config["beta"]
        config["REORDER_ITEM_SEQ"] = self.REORDER_ITEM_SEQ

    def __call__(self, dataset, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        device = item_seq.device
        reorder_seq_list = []

        for seq, length in zip(item_seq, item_seq_len):
            reorder_len = math.floor(length * self.reorder_beta)
            reorder_begin = random.randint(0, length - reorder_len)
            reorder_item_seq = seq.cpu().detach().numpy().copy()

            shuffle_index = list(range(reorder_begin, reorder_begin + reorder_len))
            random.shuffle(shuffle_index)
            reorder_item_seq[reorder_begin: reorder_begin + reorder_len] = (
                reorder_item_seq[shuffle_index]
            )

            reorder_seq_list.append(
                torch.tensor(reorder_item_seq, dtype=torch.long, device=device)
            )
        new_dict = {self.REORDER_ITEM_SEQ: torch.stack(reorder_seq_list)}
        interaction.update(Interaction(new_dict))
        return interaction


class UserDefinedTransform:
    def __init__(self, config):
        pass

    def __call__(self, dataset, interaction):
        pass
