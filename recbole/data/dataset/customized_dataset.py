# @Time   : 2020/10/19
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2021/7/9
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

"""
recbole.data.customized_dataset
##################################

We only recommend building customized datasets by inheriting.

Customized datasets named ``[Model Name]Dataset`` can be automatically called.
"""

import numpy as np
import torch

from recbole.data.dataset import KGSeqDataset, SequentialDataset
from recbole.data.interaction import Interaction
from recbole.sampler import SeqSampler
from recbole.utils.enum_type import FeatureType


class HTSARecDataset(SequentialDataset):
    def __init__(self, config):
        super(HTSARecDataset, self).__init__(config)
        self.min_time_gap = float('inf')
        self.TIMESTAMP_LIST = config['timestamp_list_field']
        self.max_time_gap = 0
        self.asc_std_slist = []  # list of sequence IDs sorted in ascending order by time-interval std

    def _change_feat_format(self):
        """
        Convert the DataFrame format (one interaction per row) to an Interaction format
        (one user interaction sequence per row).
        Change feat format from :class:`pandas.DataFrame` to :class:`Interaction`,
        then perform data augmentation.
        """
        for feat_name in self.feat_name_list:  # for each feature
            feat = getattr(self, feat_name)
            setattr(self, feat_name, self._dataframe_to_interaction(feat))
        self.logger.debug("Augmentation for sequential recommendation.")
        self.compute_timeinterval()
        self.data_augmentation()

    def process_time_std_list(self, time_std_list):
        """
        Sort by std in ascending order and return the sorted sequence ID index.
        Returns: the sorted sequence ID index list.
        """
        return sorted(range(len(time_std_list)), key=lambda i: time_std_list[i], reverse=True)

    def data_augmentation(self):
        """
        Same logic as SequentialDataset, but with a rewritten padding strategy: padding is applied at the front,
        so sequences shorter than the sequence length are padded at the front with padding elements.
        For the time sequence, the front is not padded with 0, but with the earliest timestamp element.
        First sort the original interactions by timestamp in ascending order, then group by user ID so that
        adjacent interactions of the same user are placed together, sorted by timestamp.
        Then iterate from the first interaction onward. Whenever a new user interaction sub-sequence is
        encountered, treat the previous sub-sequence as the history, the current interaction as the target
        prediction item for this sub-sequence, save the target item in the sub-interaction's item_id attribute,
        save the target interaction's timestamp in the timestamp attribute, save the previous user sub-sequence
        (i.e., the current sub-sequence's historical interaction sequence) in the item_id_list attribute,
        save the timestamps of the historical interactions in the timestamp_list attribute, and save the
        length of the interaction list in the item_list_length attribute.
        """
        self.logger.debug("data_augmentation")

        self._aug_presets()

        self._check_field("uid_field", "time_field")
        max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, sid_list, item_list_index, target_index, item_list_length = [], [], [], [], []
        seq_start = 0

        # Create negative sampling list
        # n_nums = self.item_num
        # all_items = torch.arange(1, n_nums, dtype=torch.long)
        # neg_can_num = self.config["neg_can_num"]
        # max_neg_num = self.config["max_neg_num"]
        # if n_nums > neg_can_num:
        #     indices = torch.randperm(len(all_items))[:neg_can_num]  # shuffle indices and take the first 500
        #     all_items = all_items[indices]
        # neg_can_list = []
        sid = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:  # create a sub-interaction sequence
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                sid_list.append(sid)
                sid += 1
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        time_std_list = []  # record the std of each interaction sequence; the index corresponds to the interaction sequence's index
        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f"{field}_list_field")
                list_len = self.field2seqlen[list_field]
                shape = (
                    (new_length, list_len)
                    if isinstance(list_len, int)
                    else (new_length,) + list_len
                )
                if (
                        self.field2type[field] in [FeatureType.FLOAT, FeatureType.FLOAT_SEQ]
                        and field in self.config["numerical_features"]
                ):
                    shape += (2,)
                new_dict[list_field] = torch.zeros(
                    shape, dtype=self.inter_feat[field].dtype
                )
                value = self.inter_feat[field]
                padding_value = torch.tensor(0)
                for i, (index, length) in enumerate(zip(item_list_index, item_list_length)):
                    place_list = value[index]
                    if list_field == self.TIMESTAMP_LIST:  # handle time sequences separately
                        padding_value = place_list[0]
                        time_std = torch.std(place_list, unbiased=False)
                        time_std_list.append(time_std.item())
                    if len(place_list) > list_len:
                        place_list = place_list[-list_len:]
                        length = len(place_list)
                    new_dict[list_field][i][-length:] = place_list
                    new_dict[list_field][i][
                    :-length] = padding_value  # replace the padding 0-list with the converted tensor item list of the index pair corresponding to current interaction sequence i; e.g., for a max-50 tensor
        # neg_can_list = torch.stack(neg_can_list, dim=0)
        # new_dict['neg_can_list'] = neg_can_list
        new_dict['sid'] = torch.tensor(sid_list, dtype=torch.long)
        new_data.update(Interaction(new_dict))
        self.asc_std_slist = self.process_time_std_list(time_std_list)
        self.inter_feat = new_data

    def compute_timeinterval(self):
        """
        Compute the maximum time interval across all user interactions.
        Returns: the maximum time interval (float).
        """
        # Compute the maximum interaction time interval here
        self.logger.info("computing_max_timeinterval")
        inter_feat = self.inter_feat  # already converted to Interaction format
        user_list = inter_feat[self.uid_field].unique()

        max_time_gap = 0.0  # the maximum interval across all sequences
        min_time_gap = float('inf')
        device = self.config.device
        for user_id in user_list:
            idx = (inter_feat[self.uid_field] == user_id)
            user_ts = inter_feat[self.time_field][idx]  # Tensor of timestamps
            user_ts = user_ts.to(device)
            if len(user_ts) < 2:
                continue  # cannot compute interval
            sorted_ts, indices = user_ts.sort()  # sort in ascending order
            ori_aug = sorted_ts.unsqueeze(-1).repeat(1, sorted_ts.size(0))
            re_aug = sorted_ts.unsqueeze(0).repeat(ori_aug.size(0), 1)
            time_diff = ori_aug - re_aug
            time_diff = time_diff[time_diff > 0]
            if len(time_diff) <= 0:
                continue
            time_diff_max = time_diff.max()
            time_diff_min = time_diff.min()
            min_time_gap = min(min_time_gap, time_diff_min)
            max_time_gap = max(time_diff_max, max_time_gap)

        # Save to the class; the model can access via self.max_time_gap
        self.max_time_gap = max_time_gap
        self.min_time_gap = min_time_gap


class GRU4RecKGDataset(KGSeqDataset):
    def __init__(self, config):
        super().__init__(config)


class KSRDataset(KGSeqDataset):
    def __init__(self, config):
        super().__init__(config)


class DIENDataset(SequentialDataset):
    """:class:`DIENDataset` is based on :class:`~recbole.data.dataset.sequential_dataset.SequentialDataset`.
    It is different from :class:`SequentialDataset` in `data_augmentation`.
    It add users' negative item list to interaction.

    The original version of sampling negative item list is implemented by Zhichao Feng (fzcbupt@gmail.com) in 2021/2/25,
    and he updated the codes in 2021/3/19. In 2021/7/9, Yupeng refactored SequentialDataset & SequentialDataLoader,
    then refactored DIENDataset, either.

    Attributes:
        augmentation (bool): Whether the interactions should be augmented in RecBole.
        seq_sample (recbole.sampler.SeqSampler): A sampler used to sample negative item sequence.
        neg_item_list_field (str): Field name for negative item sequence.
        neg_item_list (torch.tensor): all users' negative item history sequence.
    """

    def __init__(self, config):
        super().__init__(config)

        list_suffix = config["LIST_SUFFIX"]
        neg_prefix = config["NEG_PREFIX"]
        self.seq_sampler = SeqSampler(self)
        self.neg_item_list_field = neg_prefix + self.iid_field + list_suffix
        self.neg_item_list = self.seq_sampler.sample_neg_sequence(
            self.inter_feat[self.iid_field]
        )

    def data_augmentation(self):
        """Augmentation processing for sequential dataset.

        E.g., ``u1`` has purchase sequence ``<i1, i2, i3, i4>``,
        then after augmentation, we will generate three cases.

        ``u1, <i1> | i2``

        (Which means given user_id ``u1`` and item_seq ``<i1>``,
        we need to predict the next item ``i2``.)

        The other cases are below:

        ``u1, <i1, i2> | i3``

        ``u1, <i1, i2, i3> | i4``
        """
        self.logger.debug("data_augmentation")

        self._aug_presets()

        self._check_field("uid_field", "time_field")
        max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        uid_list = np.array(uid_list)
        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f"{field}_list_field")
                list_len = self.field2seqlen[list_field]
                shape = (
                    (new_length, list_len)
                    if isinstance(list_len, int)
                    else (new_length,) + list_len
                )
                if (
                        self.field2type[field] in [FeatureType.FLOAT, FeatureType.FLOAT_SEQ]
                        and field in self.config["numerical_features"]
                ):
                    shape += (2,)
                list_ftype = self.field2type[list_field]
                dtype = (
                    torch.int64
                    if list_ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ]
                    else torch.float64
                )
                new_dict[list_field] = torch.zeros(shape, dtype=dtype)

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(
                        zip(item_list_index, item_list_length)
                ):
                    new_dict[list_field][i][:length] = value[index]

                # DIEN
                if field == self.iid_field:
                    new_dict[self.neg_item_list_field] = torch.zeros(shape, dtype=dtype)
                    for i, (index, length) in enumerate(
                            zip(item_list_index, item_list_length)
                    ):
                        new_dict[self.neg_item_list_field][i][:length] = (
                            self.neg_item_list[index]
                        )

        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data
