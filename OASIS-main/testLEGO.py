# -*- coding: UTF-8 -*-
"""
written by Rui
file:testLEGO.py
create time:2021/05/03
"""
import numpy as np
import scipy.io as sio
import random
import EMF
import LEGO


mat = 'testdata1.mat'  # filename
data = sio.loadmat(mat)
M = data['x0']
L = data['l0']

mat2 = 'tri100000.mat'
data2 = sio.loadmat(mat2)
TT = np.array(data2['y'])

# generate random label sequence for training and test
idx_tr = np.empty((5, 660))
idx_te = np.empty((5, 330))

sq = list(range(1, 991))
for i in range(5):
    a = random.sample(range(1, 991), 990)
    idx_tr[i, :] = a[0:660]
    idx_te[i, :] = np.setdiff1d(sq, a[0:660], assume_unique=True)

map = np.zeros((1, 5))
aps = np.zeros((330, 5))
MPrecK = np.zeros((5, 11))

for i in range(5):
    # 5 trains and tests
    # generate train data and groundtruth
    M_train = np.empty((660, 10))
    V_label = np.empty((660, 1))
    for j in range(660):  # new matrix
        M_train[j, :] = M[int(idx_tr[i, j]) - 1, :]  # train data
        V_label[j, :] = L[int(idx_tr[i, j]) - 1, :]  # train groundtruth
    A0 = np.eye(10)
    # Determine similarity/dissimilarity constraints from the true labels
    l, u = EMF.ComputeDistanceExtremes(M_train, 5, 95, A0)
    n, d =TT.shape
    C = np.zeros((2*n, 4))
    C[0::2, 0] = TT[:, 0]
    # notice the difference slice between maltab and python
    C[0::2, 1] = TT[:, 1]
    C[0::2, 2] = 1
    C[0::2, 3] = 1

    C[1::2, 0] = TT[:, 0]
    C[1::2, 1] = TT[:, 2]
    C[1::2, 2] = -1
    C[1::2, 3] = u

    # C.shape[0] = 2n
    A1 = A0 / 2*n

    r = random.sample(range(2*n), 2*n)
    D = np.zeros((2*n, 4))
    for ii in range(2*n):
        D[ii, :] = C[r[ii], :]

    model = LEGO.LEGO(10, D.T)
    model.train(M_train.T, D.T, A1, 0)
    loss = model.loss
    mis = model.mis

    # generate test data and groundtruth
    M_test = np.empty((330, 10))
    U_label = np.empty((330, 1))
    for j in range(330):
        M_test[j, :] = M[int(idx_te[i, j]) - 1, :]  # test data
        U_label[j, :] = L[int(idx_te[i, j]) - 1, :]  # test groundtruth

    S = np.empty((330,330))
    for m in range(330):
        for n in range(330):
            S[m,n] = np.dot(np.dot(-(M_test[m, :] - M_test[n, :]), model.A),
                            (M_test[m, :]-M_test[n, :]).T)

    # remove original query
    idx_s = np.argsort(S, axis=1)[:, ::-1]
    numing = idx_s.shape[0]
    idx_clean = np.empty((numing, numing - 1), dtype='int')
    for j in range(numing):
        idx_row = idx_s[j, :]
        idx_clean[j, :] = idx_row[idx_row != j]
    r = U_label[idx_clean][:, :, 0]

    map[0, i], aps[0:330, i] = EMF.computeMAP(r, U_label, 0)
    MPrecK[i, 0:11], PrecKs = EMF.computePrecK2(r, U_label, 11, 0)

map_mean = np.mean(map)
map_var = np.var(map)
x25 = np.mean(MPrecK)
print('map_mean:%.6f, map_var:%.6f' % (float(map_mean), float(map_var)))
print('x25:%.6f' % float(x25))
