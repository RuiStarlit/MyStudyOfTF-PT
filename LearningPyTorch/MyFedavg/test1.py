# -*- coding:utf-8 -*-
"""
Author: RuiStarlit
File: test1
Project: LearningPyTorch
Create Time: 2021-07-07

"""
import numpy as np

num_shards, num_imgs = 5, 2
num_users = 5
dict_users = {i: np.array([]) for i in range(num_users)}
idxs = np.arange(num_shards*num_imgs)
train_labels = [2, 1, 3, 4, 5, 6, 7, 8, 9, 10]
labels = np.array(train_labels)
idxs_labels = np.vstack((idxs, labels))
idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
idx_shard = [i for i in range(num_shards)]

for i in range(num_users):
    rand_set = np.random.choice(idx_shard, 2, replace=False)
    print(rand_set)
    for rand in rand_set:
        dict_users[i] = np.concatenate(
            (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
print(dict_users)