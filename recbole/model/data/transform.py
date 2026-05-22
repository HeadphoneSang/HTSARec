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
        sequence = [0] * pad_len + sequence
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
        Perform a random mask operation on the original interaction sequence
        Args:
            dataset: The corresponding datasets, such as training sets, validators or test sets
            interaction: The interaction of an original batch
        Returns: Return the processed interaction,
        which includes the mask interaction sequence, mask position, mask item, and mask negative sampling item
        """
        item_seq = interaction[self.ITEM_SEQ]
        device = item_seq.device
        batch_size = item_seq.size(0)
        n_items = dataset.num(self.ITEM_ID)
        sequence_instances = item_seq.cpu().numpy().tolist()

        # Masked Item Prediction
        # [B * Len]
        masked_item_sequence = []  # Store each masked sequence with a shape equal to the input item_seq.
        # Inside is the position marked with the mask
        pos_items = []  #(batch,mask_item_len)
        neg_items = []  #(batch,mask_item_len)
        masked_index = []  #(batch,mask_item_len)

        if random.random() < self.ft_ratio:
            interaction = self._append_mask_last(interaction, n_items, device)
        else:
            for instance in sequence_instances:
                # WE MUST USE 'copy()' HERE!
                masked_sequence = instance.copy()
                pos_item = []
                neg_item = []
                index_ids = []
                for index_id, item in enumerate(instance):
                    # padding is 0, the sequence is end
                    if item == 0:
                        break
                    prob = random.random()
                    if prob < self.mask_ratio:
                        pos_item.append(item)
                        neg_item.append(self._neg_sample(instance, n_items))
                        masked_sequence[index_id] = n_items
                        index_ids.append(index_id)

                masked_item_sequence.append(masked_sequence)

                pos_items.append(
                    self._padding_sequence(pos_item, self.mask_item_length)
                )
                neg_items.append(
                    self._padding_sequence(neg_item, self.mask_item_length)
                )
                masked_index.append(
                    self._padding_sequence(index_ids, self.mask_item_length)
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
            interaction.update(Interaction(new_dict))
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
        sequence = [padding_v] * pad_len + sequence
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence

    def get_negative_candidates(self, item_seq, all_items):
        return all_items
        # return list(all_items - set(item_seq))

    def n_neg_sample(self, neg_item_list):
        return np.random.choice(neg_item_list, size=self.neg_sample_num, replace=False)

    def __call__(self, dataset, interaction):
        if self.config["data_augment"] != "hawkes_insert":
            return interaction
        """
        Perform a random mask operation on the original interaction sequence
        """

        item_seq = interaction[self.ITEM_SEQ]  #(batch,seq_len)
        device = item_seq.device
        batch_size = item_seq.size(0)
        n_items = dataset.num(self.ITEM_ID)
        sequence_instances = item_seq.cpu().numpy().tolist()
        if self.item_list is None:
            self.item_list = torch.arange(1, n_items, device=device, dtype=torch.long)
        pos_mask = (item_seq.unsqueeze(2) == self.item_list.unsqueeze(0).unsqueeze(1)).any(dim=1)
        neg_can_mask = ~pos_mask
        masked_item_sequence = []
        pos_items = []  # (batch,mask_item_len)
        neg_n = self.mask_item_length * self.neg_sample_num
        neg_items = torch.zeros(batch_size, neg_n, dtype=torch.long, device=device)
        masked_index = []  # (batch,mask_item_len)
        index_s = np.arange(self.max_seq_length)
        # neg_can_list = interaction['neg_can_list'].cpu().numpy().tolist()
        for batch_id, instance in enumerate(sequence_instances):
            neg_items_list = self.item_list[neg_can_mask[batch_id]]
            idx = torch.randint(0, len(neg_items_list), (neg_n,))
            neg_items[batch_id] = neg_items_list[idx]
            padding_mask = np.array(instance) != 0
            line_mask = (np.random.uniform(0, 1, size=self.max_seq_length) < self.mask_ratio) & padding_mask
            masked_sequence = np.array(instance)
            pos_item = masked_sequence[line_mask].tolist()
            index_ids = index_s[line_mask].tolist()
            masked_sequence[line_mask] = n_items
            masked_sequence = masked_sequence.tolist()
            masked_item_sequence.append(masked_sequence)
            pos_items.append(
                self._padding_sequence(pos_item, self.mask_item_length)
            )
            masked_index.append(
                self._padding_sequence(index_ids, self.mask_item_length)
            )

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
        interaction.update(Interaction(new_dict))
        return interaction

    def set_random_sample(self, s, k):
        """
        Negative sampling of the reservoir
        Args:
            k: The number of negative samples
            s: The sample set of negative sampling
        Returns: Return the set of negative samples
        """
        assert k <= len(s)
        result = []
        it = iter(s)
        for _ in range(k):
            result.append(next(it))
        for i, val in enumerate(it, k):
            j = random.randint(0, i)
            if j < k:
                result[j] = val
        return result

    def neg_filter_sample(self, filtered_item_set, sample_num, min_id, max_id):
        """
        Perform a specified number of negative samplings, with the target not including filtered_item
        Here, the method of determining the decision strategy after random sampling is adopted
        The best-case time complexity is O(sample_num)
        Args:
            filtered_item_set: Positive sample set, the sample to be rejected
            sample_num: number
            min_id: min
            max_id: max

        Returns: list

        """
        assert max_id - min_id + 1 - len(filtered_item_set) >= sample_num, \
            "The sampling space is too small to generate enough negative samples"
        rand_ids = random.sample(range(min_id, max_id + 1), sample_num)
        sampled_set = set()
        for i, rand_id in enumerate(rand_ids):
            while rand_id in filtered_item_set or rand_id in sampled_set:
                rand_id = random.randint(min_id, max_id)
            rand_ids[i] = rand_id
            sampled_set.add(rand_id)
        return rand_ids

    def __call0__(self, dataset, interaction):

        item_seq = interaction[self.ITEM_SEQ]  #(batch,seq_len)
        device = item_seq.device
        batch_size = item_seq.size(0)
        n_items = dataset.num(self.ITEM_ID)
        sequence_instances = item_seq.cpu().numpy().tolist()
        masked_item_sequence = []
        pos_items = []  # (batch,mask_item_len)
        neg_items = []
        masked_index = []  # (batch,mask_item_len)
        index_s = np.arange(self.max_seq_length)
        neg_n = self.mask_item_length * self.neg_sample_num
        for batch_id, instance in enumerate(sequence_instances):
            padding_mask = np.array(instance) != 0
            line_mask = (np.random.uniform(0, 1, size=self.max_seq_length) < self.mask_ratio) & padding_mask
            masked_sequence = np.array(instance)
            pos_item = masked_sequence[line_mask].tolist()
            index_ids = index_s[line_mask].tolist()
            masked_sequence[line_mask] = n_items
            masked_sequence = masked_sequence.tolist()
            masked_item_sequence.append(masked_sequence)

            neg_list = self.neg_filter_sample(set(instance), neg_n, 1, dataset.item_num - 1)
            neg_items.append(
                self._padding_sequence(neg_list, neg_n)
            )
            pos_items.append(
                self._padding_sequence(pos_item, self.mask_item_length)
            )
            masked_index.append(
                self._padding_sequence(index_ids, self.mask_item_length)
            )

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
        interaction.update(Interaction(new_dict))
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
