# -*- coding: UTF-8 -*-
"""
written by Rui
file:testITML.py
create time:2021/05/04
"""
import numpy as np
import scipy.io as sio
import random
import EMF


class IMTL:
    def __init__(self, C, A):
        num = C.shape[0]
        self.A = A.copy()
        self.loss = np.zeros((1, num))
        self.mis = np.zeros((1, num))

    def train(self, C, X, A0):
        """use itmlonlineALG trainning model"""

        # check to make sure that no 2 constrained vectors are identical
        valid = np.ones((C.shape[0], 1))
        for i in range(C.shape[0]):
            i1 = int(C[i, 0])
            i2 = int(C[i, 1])
            v = X[i1 - 1, :].T - X[i2 - 1, :].T
            if np.linalg.norm(v) < 10e-10:
                valid[i] = 0
        C = C[(valid > 0)[:, 0], :]

        num = C.shape[0]
        A = A0.copy()
        eta0 = 1e-2
        loss = np.zeros((1, num))
        mis = np.zeros((1, num))
        lt = 0
        pos = 0
        neg = 0
        t_tick = 100
        verbose = 1
        err_count = 0
        err_count_inbatch = 0
        n_inbatch = 0
        t_tick_count = 0

        for i in range(num):
            u = X[int(C[i, 0])-1, :].T
            v = X[int(C[i, 1])-1, :].T

            z = u - v
            y = C[i, 3]
            y_hat = np.dot(np.dot(z.T, A), z)

            if i == 0:
                V = 1 / (X.shape[1] - 1) * np.eye(X.shape[1])

            lm = (y_hat - y) ** 2
            lt += lm
            loss[0, i] = lt / i
            n_inbatch += 1

            if y_hat - y >= 0:
                eta = eta0
                pos += 1
            else:
                eta = min(eta0, 1 / (2 * (y - y_hat) * (np.dot(np.dot(z.T, np.eye(X.shape[1]) + V), z)) ** -1))
                neg += 1
                err_count += 1
                err_count_inbatch += 1
            if i > 0:
                mis[0, i] = neg / i
            alpha = -2 * eta * (y_hat - y) / (1 + 2 * eta * (y_hat - y) * y_hat)
            beta = -alpha / (1 + alpha * y_hat)

            V = V + -beta * np.dot(V * np.dot(z, z.T), V) / (1 + beta * np.dot(np.dot(z.T, V), z))
            A = A + alpha * np.dot(np.dot(A, np.dot(z, z.T)), A)

            # verbose
            if (i % t_tick == 0 or i == num) and i > 0:
                t_tick_count += 1
                if verbose:
                    print('t:%d/T:%d, n_err:%d, err_rate:%.4f;, loss:%.4f;'
                          '  n_inbatch:%d, errrate_inbatch:%.4f .' %
                          (i, num, err_count, err_count / i,
                           loss[0, i], n_inbatch, err_count_inbatch / n_inbatch))
                else:
                    print('.')
                n_inbatch = 0
                err_count_inbatch = 0

        self.loss = loss
        self.mis = mis
        self.A = A
        return self


if __name__ == '__main__':
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
        n, d = TT.shape
        C = np.zeros((2 * n, 4))
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
        A1 = A0 / 2 * n

        r = random.sample(range(2 * n), 2 * n)
        D = np.zeros((2 * n, 4))
        for ii in range(2 * n):
            D[ii, :] = C[r[ii], :]

        model = IMTL(D, A1)
        model.train(D, M_train, A1)
        loss = model.loss
        mis = model.mis

        # generate test data and groundtruth
        M_test = np.empty((330, 10))
        U_label = np.empty((330, 1))
        for j in range(330):
            M_test[j, :] = M[int(idx_te[i, j]) - 1, :]  # test data
            U_label[j, :] = L[int(idx_te[i, j]) - 1, :]  # test groundtruth

        S = np.empty((330, 330))
        for m in range(330):
            for n in range(330):
                S[m, n] = np.dot(np.dot(-(M_test[m, :] - M_test[n, :]), model.A),
                                 (M_test[m, :] - M_test[n, :]).T)

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
    # map_var = np.var(map)
    x25 = np.mean(MPrecK)
    print('map_mean:%.6f' % float(map_mean) )
    print('x25:%.6f' % float(x25) )
