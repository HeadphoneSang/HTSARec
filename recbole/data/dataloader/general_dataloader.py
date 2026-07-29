# @Time   : 2020/7/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/9/9, 2020/9/29, 2021/7/15, 2022/7/6
# @Author : Zhen Tian, Yupeng Hou, Yushuo Chen, Xingyu Pan, Gaowei Zhang
# @email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, xy_pan@foxmail.com, zgw15630559577@163.com

"""
recbole.data.dataloader.general_dataloader
################################################
"""
import copy

import numpy as np
import torch
from logging import getLogger
import time
from recbole.data.dataloader.abstract_dataloader import (
    AbstractDataLoader,
    NegSampleDataLoader,
)
from recbole.data.interaction import Interaction, cat_interactions
from recbole.utils import InputType, ModelType


class TrainDataLoader(NegSampleDataLoader):
    """:class:`TrainDataLoader` is a dataloader for training.
    It can generate negative interaction when :attr:`training_neg_sample_num` is not zero.
    For the result of every batch, we permit that every positive interaction and its negative interaction
    must be in the same batch.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        self._set_neg_sample_args(
            config, dataset, config["MODEL_INPUT_TYPE"], config["train_neg_sample_args"]
        )
        self.sample_size = len(dataset)
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["train_batch_size"]
        if self.neg_sample_args["distribution"] != "none":
            batch_num = max(batch_size // self.times, 1)
            new_batch_size = batch_num * self.times
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    def update_config(self, config):
        self._set_neg_sample_args(
            config,
            self._dataset,
            config["MODEL_INPUT_TYPE"],
            config["train_neg_sample_args"],
        )
        super().update_config(config)

    def collate_fn(self, index):
        index = np.array(index)
        data = self._dataset[index]
        transformed_data = self.transform(self._dataset, data)
        return self._neg_sampling(transformed_data)


class NegSampleEvalDataLoader(NegSampleDataLoader):
    """:class:`NegSampleEvalDataLoader` is a dataloader for neg-sampling evaluation.
    It is similar to :class:`TrainDataLoader` which can generate negative items,
    and this dataloader also permits that all the interactions corresponding to each user are in the same batch
    and positive interactions are before negative interactions.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        phase = sampler.phase if sampler is not None else "test"  #当前阶段的字符串，valid test
        self._set_neg_sample_args(
            config, dataset, InputType.POINTWISE, config[f"{phase}_neg_sample_args"]
        )
        if (
                self.neg_sample_args["distribution"] != "none"
                and self.neg_sample_args["sample_num"] != "none"
        ):  #如果是制定样本数的负采样
            user_num = dataset.user_num
            dataset.sort(by=dataset.uid_field, ascending=True)
            self.uid_list = []
            start, end = dict(), dict()  #存储每个用户最早的交互的下标和最晚的
            for i, uid in enumerate(dataset.inter_feat[dataset.uid_field].numpy()):  #将用户ID转成np列表一个个便利
                if uid not in start:
                    self.uid_list.append(uid)
                    start[uid] = i
                end[uid] = i
            self.uid2index = np.array([None] * user_num)  #存储的用户对应的交互的索引，感觉没意义
            self.uid2items_num = np.zeros(user_num, dtype=np.int64)  #存储的用户的交互序列的数量
            for uid in self.uid_list:
                self.uid2index[uid] = slice(start[uid], end[uid] + 1)
                self.uid2items_num[uid] = end[uid] - start[uid] + 1
            self.uid_list = np.array(self.uid_list)  #存储所有用户的id
            self.sample_size = len(self.uid_list)  #存储用户的数量
        else:
            self.sample_size = len(dataset)
        if shuffle:
            self.logger.warning("NegSampleEvalDataLoader can't shuffle")
            shuffle = False
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        """
        这个方法在创建dataloader的时候会在init里面调用一次来设定
        当前loader的批次大小，只会不会再调用
        Returns:

        """
        batch_size = self.config["eval_batch_size"]  #获取配置文件的size
        if (
                self.neg_sample_args["distribution"] != "none"
                and self.neg_sample_args["sample_num"] != "none"
        ):  #如果是指定数量的负采样
            inters_num = sorted(self.uid2items_num * self.times, reverse=True)  #对原始的记录用户交互序列数量的数据进行广播乘法
            #每个位置乘（负样本采样数量+1个正样本），然后按照交互序列数量降序排序
            batch_num = 1  #记录一个batch最多装下几个用户的样本，默认第一个用户可以装下
            new_batch_size = inters_num[0]
            for i in range(1, len(inters_num)):  #一个个用户检查，贪心策略，希望可以装下多的样本
                #最后得到一个batch可以装的最多的用户数和新的batch大小
                if new_batch_size + inters_num[i] > batch_size:
                    break
                batch_num = i + 1
                new_batch_size += inters_num[i]
            self.step = batch_num  #记录当前batch有几个用户，一定是包含用户的所有样本集合
            self.set_batch_size(new_batch_size)  #记录当前batch的实际大小
        else:  #如果不是制定样本负采样，直接记录配置文件的，用户数量等于负样本数量
            self.step = batch_size
            self.set_batch_size(batch_size)

    def update_config(self, config):
        """
        用来将新的配置更新到采样器里面
        Args:
            config:

        Returns:

        """
        phase = self._sampler.phase if self._sampler.phase is not None else "test"
        self._set_neg_sample_args(
            config,
            self._dataset,
            InputType.POINTWISE,
            config[f"{phase}_neg_sample_args"],
        )
        super().update_config(config)

    def collate_fn(self, index):
        """
        每次调用loader的next_batch时调用
        Args:
            index:

        Returns:

        """
        index = np.array(index)  #送进来的
        if (
                self.neg_sample_args["distribution"] != "none"
                and self.neg_sample_args["sample_num"] != "none"
        ):
            uid_list = self.uid_list[index]
            data_list = []
            idx_list = []
            positive_u = []
            positive_i = torch.tensor([], dtype=torch.int64)

            for idx, uid in enumerate(uid_list):  #按照用户id处理数据，为同一个用户下的数据分配相同的索引
                index = self.uid2index[uid]  #获取用户对应的交互的索引元组(start,end+1)
                origin_interactions = self._dataset[index]
                transformed_data = self.transform(self._dataset, origin_interactions)  #对该用户的交互数据进行预处理
                #返回interaction对象
                data_list.append(self._neg_sampling(transformed_data))
                #存放用户对应的正样本交互和负样本交互 interaction
                idx_list += [idx for i in range(self.uid2items_num[uid] * self.times)]
                #标记每个构建的新的交互，属于当前batch的第几个用户 长度为(user_num * seq_num * neg_num+1)
                positive_u += [idx for i in range(self.uid2items_num[uid])]
                #标记每个正样本所属的用户在当前batch的索引，长度为(user_num * seq_num)
                positive_i = torch.cat(
                    (positive_i, self._dataset[index][self.iid_field]), 0
                )  #将每个用户的正样本存进去 长度为(user_num * seq_len)
                #positive_i 和 positive_u 对应的

            cur_data = cat_interactions(data_list)  #合并所有用户interaction
            idx_list = torch.from_numpy(np.array(idx_list)).long()
            positive_u = torch.from_numpy(np.array(positive_u)).long()

            return cur_data, idx_list, positive_u, positive_i
        else:
            data = self._dataset[index]
            transformed_data = self.transform(self._dataset, data)
            cur_data = self._neg_sampling(transformed_data)  #返回正样本和负样本对应的交互
            return cur_data, None, None, None


class FullSortEvalDataLoader(AbstractDataLoader):
    """:class:`FullSortEvalDataLoader` is a dataloader for full-sort evaluation. In order to speed up calculation,
    this dataloader would only return then user part of interactions, positive items and used items.
    It would not return negative items.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.is_sequential = config["MODEL_TYPE"] == ModelType.SEQUENTIAL
        if not self.is_sequential:
            user_num = dataset.user_num
            self.uid_list = []
            self.uid2items_num = np.zeros(user_num, dtype=np.int64)
            self.uid2positive_item = np.array([None] * user_num)
            self.uid2history_item = np.array([None] * user_num)

            dataset.sort(by=self.uid_field, ascending=True)
            last_uid = None
            positive_item = set()
            uid2used_item = sampler.used_ids
            for uid, iid in zip(
                    dataset.inter_feat[self.uid_field].numpy(),
                    dataset.inter_feat[self.iid_field].numpy(),
            ):
                if uid != last_uid:
                    self._set_user_property(
                        last_uid, uid2used_item[last_uid], positive_item
                    )
                    last_uid = uid
                    self.uid_list.append(uid)
                    positive_item = set()
                positive_item.add(iid)
            self._set_user_property(last_uid, uid2used_item[last_uid], positive_item)
            self.uid_list = torch.tensor(self.uid_list, dtype=torch.int64)
            self.user_df = dataset.join(Interaction({self.uid_field: self.uid_list}))

        self.sample_size = len(self.user_df) if not self.is_sequential else len(dataset)
        if shuffle:
            self.logger.warning("FullSortEvalDataLoader can't shuffle")
            shuffle = False
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _set_user_property(self, uid, used_item, positive_item):
        if uid is None:
            return
        history_item = used_item - positive_item
        self.uid2positive_item[uid] = torch.tensor(
            list(positive_item), dtype=torch.int64
        )
        self.uid2items_num[uid] = len(positive_item)
        self.uid2history_item[uid] = torch.tensor(list(history_item), dtype=torch.int64)

    def _init_batch_size_and_step(self):
        batch_size = self.config["eval_batch_size"]
        if not self.is_sequential:
            batch_num = max(batch_size // self._dataset.item_num, 1)
            new_batch_size = batch_num * self._dataset.item_num
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    def update_config(self, config):
        super().update_config(config)

    def collate_fn(self, index):
        index = np.array(index)
        if not self.is_sequential:
            user_df = self.user_df[index]
            uid_list = list(user_df[self.uid_field])

            history_item = self.uid2history_item[uid_list]
            positive_item = self.uid2positive_item[uid_list]

            history_u = torch.cat(
                [
                    torch.full_like(hist_iid, i)
                    for i, hist_iid in enumerate(history_item)
                ]
            )
            history_i = torch.cat(list(history_item))

            positive_u = torch.cat(
                [torch.full_like(pos_iid, i) for i, pos_iid in enumerate(positive_item)]
            )
            positive_i = torch.cat(list(positive_item))

            return user_df, (history_u, history_i), positive_u, positive_i
        else:
            interaction = self._dataset[index]
            transformed_interaction = self.transform(self._dataset, interaction)
            inter_num = len(transformed_interaction)
            positive_u = torch.arange(inter_num)
            positive_i = transformed_interaction[self.iid_field]

            return transformed_interaction, None, positive_u, positive_i


class FastNegSampleEvalDataLoader(NegSampleDataLoader):
    """:class:`NegSampleEvalDataLoader` is a dataloader for neg-sampling evaluation.
    It is similar to :class:`TrainDataLoader` which can generate negative items,
    and this dataloader also permits that all the interactions corresponding to each user are in the same batch
    and positive interactions are before negative interactions.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        phase = sampler.phase if sampler is not None else "test"  #当前阶段的字符串，valid test
        self._set_neg_sample_args(
            config, dataset, InputType.POINTWISE, config[f"{phase}_neg_sample_args"]
        )
        if (
                self.neg_sample_args["distribution"] != "none"
                and self.neg_sample_args["sample_num"] != "none"
        ):  #如果是制定样本数的负采样
            user_num = dataset.user_num
            dataset.sort(by=dataset.uid_field, ascending=True)
            self.uid_list = []
            start, end = dict(), dict()  #存储每个用户最早的交互的下标和最晚的
            for i, uid in enumerate(dataset.inter_feat[dataset.uid_field].numpy()):  #将用户ID转成np列表一个个便利
                if uid not in start:
                    self.uid_list.append(uid)
                    start[uid] = i
                end[uid] = i
            self.uid2index = np.array([None] * user_num)  #存储的用户对应的交互的索引，感觉没意义
            self.uid2items_num = np.zeros(user_num, dtype=np.int64)  #存储的用户的交互序列的数量
            for uid in self.uid_list:
                self.uid2index[uid] = slice(start[uid], end[uid] + 1)
                self.uid2items_num[uid] = end[uid] - start[uid] + 1
            self.uid_list = np.array(self.uid_list)  #存储所有用户的id
            self.sample_size = len(self.uid_list)  #存储用户的数量
        else:
            self.sample_size = len(dataset)
        if shuffle:
            self.logger.warning("NegSampleEvalDataLoader can't shuffle")
            shuffle = False
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        self.NEG_FIELD = config['eval_args']['neg_field']

    def _init_batch_size_and_step(self):
        """
        这个方法在创建dataloader的时候会在init里面调用一次来设定
        当前loader的批次大小，只会不会再调用
        Returns:

        """
        batch_size = self.config["eval_batch_size"]  #获取配置文件的size
        self.step = batch_size
        self.set_batch_size(batch_size)

    def update_config(self, config):
        """
        用来将新的配置更新到采样器里面
        Args:
            config:

        Returns:

        """
        phase = self._sampler.phase if self._sampler.phase is not None else "test"
        self._set_neg_sample_args(
            config,
            self._dataset,
            InputType.POINTWISE,
            config[f"{phase}_neg_sample_args"],
        )
        super().update_config(config)

    def collate_fn(self, index):
        """
        每次调用loader的next_batch时调用
        Args:
            index:

        Returns:

        """
        index = np.array(index)  #送进来的
        if (
                self.neg_sample_args["distribution"] != "none"
                and self.neg_sample_args["sample_num"] != "none"
        ):
            uid_list = self.uid_list[index]
            data_list = []
            idx_list = []
            positive_u = []
            positive_i = torch.tensor([], dtype=torch.int64)

            for idx, uid in enumerate(uid_list):  #按照用户id处理数据，为同一个用户下的数据分配相同的索引
                index = self.uid2index[uid]  #获取用户对应的交互的索引元组(start,end+1)
                origin_interactions = self._dataset[index]
                transformed_data = origin_interactions
                #返回interaction对象
                data_list.append(self._neg_sampling(transformed_data))
                #存放用户对应的正样本交互和负样本交互 interaction
                idx_list += [idx for i in range(self.uid2items_num[uid] * self.times)]
                #标记每个构建的新的交互，属于当前batch的第几个用户 长度为(user_num * seq_num * neg_num+1)
                positive_u += [idx for i in range(self.uid2items_num[uid])]
                #标记每个正样本所属的用户在当前batch的索引，长度为(user_num * seq_num)
                positive_i = torch.cat(
                    (positive_i, self._dataset[index][self.iid_field]), 0
                )  #将每个用户的正样本存进去 长度为(user_num * seq_len)
                #positive_i 和 positive_u 对应的

            cur_data = cat_interactions(data_list)  #合并所有用户interaction
            idx_list = torch.from_numpy(np.array(idx_list)).long()
            positive_u = torch.from_numpy(np.array(positive_u)).long()

            return cur_data, idx_list, positive_u, positive_i
        else:
            data = self._dataset[index]
            transformed_data = self.transform(self._dataset, data)
            cur_data = self._neg_sampling(transformed_data)  #返回正样本和负样本对应的交互
            return cur_data, None, None, None

    def _neg_sampling(self, inter_feat):
        if self.neg_sample_args.get("dynamic", False):
            #动态采样
            candidate_num = self.neg_sample_args["candidate_num"]
            user_ids = inter_feat[self.uid_field].numpy()
            item_ids = inter_feat[self.iid_field].numpy()
            neg_candidate_ids = self._sampler.sample_by_user_ids(
                user_ids, item_ids, self.neg_sample_num * candidate_num
            )
            self.model.eval()
            interaction = copy.deepcopy(inter_feat).to(self.model.device)
            interaction = interaction.repeat(self.neg_sample_num * candidate_num)
            neg_item_feat = Interaction(
                {self.iid_field: neg_candidate_ids.to(self.model.device)}
            )
            interaction.update(neg_item_feat)
            scores = self.model.predict(interaction).reshape(candidate_num, -1)
            indices = torch.max(scores, dim=0)[1].detach()
            neg_candidate_ids = neg_candidate_ids.reshape(candidate_num, -1)
            neg_item_ids = neg_candidate_ids[
                indices, [i for i in range(neg_candidate_ids.shape[1])]
            ].view(-1)
        elif (
                self.neg_sample_args["distribution"] != "none"
                and self.neg_sample_args["sample_num"] != "none"
        ):
            #均匀采样
            user_ids = inter_feat[self.uid_field].numpy()
            item_ids = inter_feat[self.iid_field].numpy()
            neg_item_ids = self._sampler.sample_by_user_ids(
                user_ids, item_ids, self.neg_sample_num
            )  #采样对应用户的负样本集合
        else:
            neg_item_ids = torch.zeros(self.neg_sample_num, dtype=torch.long)
        inter_feat[self.NEG_FIELD] = neg_item_ids.unsqueeze(0)
        return inter_feat
