# -*- coding: UTF-8 -*-
"""
written by Rui
file:testEuclidean.py
create time:2021/05/04
"""
import numpy as np
import scipy.io as sio
import random
import Oasis


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

map3 = np.zeros((1, 5))
aps3 = np.empty((330, 5))
MPrecK3 = np.empty((5, 11))

for i in range(5):
    # 5 trains and tests
    # generate train data and groundtruth
    M_train3 = np.empty((660, 10))
    V_label3 = np.empty((660, 1))
    for j in range(660):  # new matrix
        M_train3[j, :] = M[int(idx_tr[i, j]) - 1, :]  # train data
        V_label3[j, :] = L[int(idx_tr[i, j]) - 1, :]  # train groundtruth

    # generate test data and groundtruth
    M_test3 = np.empty((330, 10))
    U_label3 = np.empty((330, 1))
    for j in range(330):
        M_test3[j, :] = M[int(idx_te[i, j]) - 1, :]  # test data
        U_label3[j, :] = L[int(idx_te[i, j]) - 1, :]  # test groundtruth

    S3 = np.empty((330,330))
    for m in range(330):
        for n in range(330):
            S3[m, n] = -np.dot((M_test3[m, :]-M_test3[n,:]),(M_test3[m,:]-M_test3[n,:]).T)

    idx_s3 = np.argsort(S3, axis=1)[:, ::-1]
    numing = idx_s3.shape[0]
    idx_clean = np.empty((numing, numing - 1), dtype='int')
    for j in range(numing):
        idx_row = idx_s3[j, :]
        idx_clean[j, :] = idx_row[idx_row != j]
    r = U_label3[idx_clean][:, :, 0]

    map3[0, i], aps3[0:330, i] = Oasis.computeMAP(r, U_label3, 0)
    MPrecK3[i, 0:11], PrecKs3 = Oasis.computePrecK2(r, U_label3, 11, 0)

map3_mean = np.mean(map3)
xEu = np.mean(MPrecK3)
print('map3_mean:%0.6f' % float(map3_mean))
print('xEu:%0.6f' % float(xEu))