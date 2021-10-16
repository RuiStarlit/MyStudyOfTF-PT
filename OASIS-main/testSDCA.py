# -*- coding: UTF-8 -*-
"""
written by Rui
file:testSDCA.py
create time:2021/05/03
"""

import numpy as np
import scipy.io as sio
import random
import SDCA
import EMF

# loading data
# 先在matlab中把稀疏矩阵转化为稠密矩阵（对于python中稀疏矩阵的存取还需要学习)

mat = 'testdata1.mat'  # filename
data = sio.loadmat(mat)
M = data['x0']
L = data['l0']
# generate random label sequence for training and test
idx_tr = np.empty((15, 660))
idx_te = np.empty((15, 330))

sq = list(range(1, 991))
for i in range(15):
    a = random.sample(range(1, 991), 990)
    idx_tr[i, :] = a[0:660]
    idx_te[i, :] = np.setdiff1d(sq, a[0:660], assume_unique=True)

map = np.zeros((1, 15))
aps = np.empty((330, 15))
MPrecK = np.empty((15, 11))

for i in range(5):
    # 5 trains and tests
    # generate train data and groundtruth
    M_train = np.empty((660, 10))
    V_label = np.empty((660, 1))
    for j in range(660):  # new matrix
        M_train[j, :] = M[int(idx_tr[i, j]) - 1, :]  # train data
        V_label[j, :] = L[int(idx_tr[i, j]) - 1, :]  # train groundtruth

    ite = 20000
    eta = 0.001
    model = SDCA.sdca(dim=M_train.shape[1], T=ite)

    model.train(M_train, V_label, T=ite, eta=eta)
    loss = model.loss
    mistake = model.mistake

    # generate test data and groundtruth
    M_test = np.empty((330, 10))
    U_label = np.empty((330, 1))
    for j in range(330):
        M_test[j, :] = M[int(idx_te[i, j]) - 1, :]  # test data
        U_label[j, :] = L[int(idx_te[i, j]) - 1, :]  # test groundtruth
    S = np.dot(np.dot(M_test, model.M), M_test.T)
    idx_s = np.argsort(S, axis=1)[:, ::-1]
    S_n = S[idx_s]
    # remove original query
    numing = idx_s.shape[0]
    idx_clean = np.empty((numing, numing - 1), dtype='int')
    for j in range(numing):
        idx_row = idx_s[j, :]
        idx_clean[j, :] = idx_row[idx_row != j]
    r = U_label[idx_clean][:, :, 0]
    map[0, i], aps[0:330, i] = EMF.computeMAP(r, U_label, 0)
    MPrecK[i, 0:11], PrecKs = EMF.computePrecK2(r, U_label, 11, 0)

idx_map = np.argsort(map, axis=1)[:, ::-1]
map_n = map[0, idx_map[0, :]]
map_mean = np.mean(map_n[1:5])
map_var = np.var(map_n[1:5])
x25 = np.mean(MPrecK[idx_map[0, 0:5], :])
print(map_mean)
print(x25)