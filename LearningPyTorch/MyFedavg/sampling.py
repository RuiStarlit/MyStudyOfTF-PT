# -*- coding:utf-8 -*-
"""
Author: RuiStarlit
File: sampling
Project: LearningPyTorch
Create Time: 2021-07-07
Primary Objective：对CIFAR数据集进行NOIID采样
"""
import numpy as np
from torchvision import datasets, transforms
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


def cifar10_noniid(dataset, n_users):
    """
    Sample Non-I.I.D data from CIFAR-10
    :param dataset:
    :param n_users:
    :return: a dict of clients with each clients assigned certain
    number of training imgs
    """
    n_shards, n_imgs = 200, 250  # 200碎片，一个碎片250图像
    idx_shard = [i for i in range(n_shards)]
    dict_users = {i: np.array([]) for i in range(n_users)}
    idxs = np.arange(n_shards * n_imgs)
    labels = np.array(dataset.targets)

    # sort the data labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # 分割并指派 每个用户分到的碎片可能重复
    for i in range(n_users):
        rand_set = np.random.choice(idx_shard, 2, replace=False)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * n_imgs:(rand + 1) * n_imgs]), axis=0)
    return dict_users

def cifar100_noniid(dataset, n_users):
    """
    Sample Non-I.I.D data from CIFAR-100
    :param dataset:
    :param n_users:
    :return: a dict of clients with each clients assigned certain
    number of training imgs
    """
    raise NotImplementedError()