# -*- coding: UTF-8 -*-
"""
written by Rui
file:Oasis_new.py
create time:
"""
from sys import stdout

import numpy as np
import random
import datetime
import gzip
import os
import pickle


# import matplotlib.pyplot as plt
# from sklearn.base import BaseEstimator


def make_psd(W):
    """ Make matrix positive semi-definite.
        通过使用W[:]可以使得数组里面的元素被修改 """
    w, V = np.linalg.eig(0.5 * (W + W.T))
    D = np.diagflat(np.maximum(w, 0))
    W[:] = np.dot(np.dot(V, D), V.T)


def symmetrize(W):
    """ Symmetrize matrix."""
    W[:] = 0.5 * (W + W.T)


def computeAP(predict, groundtruth, pos_size):
    # old_recall = 0
    # old_precision = 0
    AP = 0
    match_size = 0
    matches = predict == groundtruth
    try:
        if matches.any() == 0:
            raise ValueError("there is not any match in dataset for this query, why?")
    except ValueError as e:
        print('Error：', repr(e))
        raise
    if pos_size.any() == 0:
        pos_size = np.sum(matches)
    for i in range(len(matches)):
        if matches[i] == True:
            match_size += 1
            AP += match_size / (i + 1)
    # recall = match_size / pos_size;
    # precision = match_size / i;
    # AP = AP + ( recall - old_recall ) * ( ( old_precision + precision ) / 2.0);
    # old_recall = recall;
    # old_precision = precision;
    AP = AP / pos_size
    return AP


def computeMAP(predicts, groundtruth, pos_size_each_row):
    """Input:
            predicts:    m x n, m queries, top n predicts.
            groundtruth:    m x n, m queries, top n groundtruth.
                or m x 1, m queries, identical ground truth for top n predicts
            pos_size_each_row:    m x 1, positive instance num for m queries.
                if value is zero, pos_size = sum(predict == groundtruth).
        OUTPUT:
             MAP:  MAP value.
    """
    m, n = predicts.shape
    MAP = 0
    if isinstance(pos_size_each_row, int):
        pos_size_each_row_n = np.ones((1, m)) * pos_size_each_row
    elif len(pos_size_each_row) == 1:
        pos_size_each_row_n = np.ones((1, m)) * pos_size_each_row
    APs = np.empty(m)
    for i in range(m):
        AP = computeAP(predicts[i, :], groundtruth[i, :], pos_size_each_row_n[0, i])
        MAP += AP
        APs[i] = AP
    MAP = MAP / m
    return MAP, APs


def computeP(predict, groundtruth, K):
    # K = predict.shape[0]
    matches = predict == groundtruth
    l = len(matches)
    if l < K:
        matches = np.concatenate((matches, np.zeros((K - l), dtype='bool')), axis=0)
    p2 = np.cumsum(matches) / np.arange(1, l + 1)
    PrecK = p2[0:K]
    return PrecK


def computePrecK2(predicts, groundtruth, K, pos_size_each_row):
    """
    INPUT:
        predicts:    m x n, m queries, top n predicts.
        groundtruth:    m x n, m queries, top n groundtruth.
            or m x 1, m queries, identical ground truth for top n predicts
        pos_size_each_row:    m x 1, positive instance num for m queries.
             if value is zero, pos_size = sum(predict == groundtruth).
        OUTPUT:
            MAP: MAP value
    """
    predicts = predicts[:, 0:K]
    if groundtruth.shape[1] > K:
        groundtruth = groundtruth[:, 0:K]

    m, n = predicts.shape
    MPrecK = np.zeros((m, K))
    # if length(pos_size_each_row) == 1 :
    # pos_size_each_row = pos_size_each_row * ones(1, m);
    PrecKs = np.zeros((m, K))
    for i in range(m):
        Preck = computeP(predicts[i, :], groundtruth[i, :], K)
        MPrecK[i, :] = Preck
        PrecKs[i, :] = Preck
    MPrecK = np.mean(MPrecK, axis=0)
    return MPrecK, PrecKs


class Oasis:
    """The OASIS Algorithm"""

    def __init__(self, aggress=0.1, random_seed=None, do_sym=False,
                 do_psd=False, do_save=False, n_iter=10 ** 6, sym_every=1, save_every=None,
                 psd_every=1, save_path=None):
        self.aggress = aggress
        self.random_seed = random_seed
        self.n_iter = n_iter
        self.do_sym = do_sym
        self.do_psd = do_psd
        self.sym_every = sym_every
        self.psd_every = psd_every
        self.save_path = save_path
        self.do_save = do_save
        self.loss_steps = 0
        self.los = 0

        if save_every is None:
            self.save_every = int(np.ceil(self.n_iter / 10))

        if save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

    def _getstate(self):
        return (self._weights,)

    def _setstate(self, state):
        weights, = state
        self._weights = weights

    def _save(self, n=None):
        """ Pickle the model."""
        fname = self.save_path + "/model%04d.pklz" % n
        f = gzip.open(fname, 'wb')
        state = self._getstate()
        pickle.dump(state, f)
        f.close()

    def read_snapshot(self, fname):
        """ Read model state snapshot from gzipped pickle. """
        f = gzip.open(fname, 'rb')
        state = pickle.load(f)
        self._setstate(state)

    # 以上四个函数是用来保存模型文件的，还需要反复琢磨怎么用的

    def train(self, W, X, y, class_start, class_sizes, n_iter, verbose=False):
        """ AKA oasis_m
            Train batch inner loop
            X is data
            y is class label
            verbose: Display training progress information
        """
        loss_steps_batch = np.empty((n_iter,), dtype='bool')
        los = np.zeros(1, n_iter)
        lt = 0
        n_samples, n_features = X.shape

        assert (W.shape[0] == n_features), 'dimension mismatch'
        assert (W.shape[1] == n_features), 'dimension mismatch'

        for ii in range(n_iter):
            # ii is i_iter
            if verbose:
                if np.mod(ii + 1, 100) == 0:
                    print('.')
                if np.mod(ii + 1, 1000) == 0:
                    print('%d' % (ii + 1))
                if np.mod(ii + 1, 10000) == 0:
                    print("[%s]" % str(datetime.datetime.now()))
                stdout.flush()  # 刷新输出缓存

            # Sample a query image
            p_ind = random.randint(0, n_samples - 1)  # 可以调用self.init作为随机数种子
            label = y[p_ind]

            # Draw random postive sample
            pos_ind = class_start[label] + random.randint(0, class_sizes[label] - 1)
            # Draw random negative sample
            neg_ind = random.randint(0, n_samples - 1)
            while y[neg_ind] == label:
                neg_ind = random.randint(0, n_samples - 1)

            p = X[p_ind]

            sameples_delta = X[pos_ind] - X[neg_ind]

            loss = 1 - np.dot(np.dot(p, W), sameples_delta)

            if loss > 0:
                # Update W
                # p.T * samples_delta = outer(p, sameples_delta) i.e V^i in paper
                grad_W = np.outer(p, sameples_delta)
                loss_steps_batch[ii] = True
                norm_grad_W = np.dot(p, p) * np.dot(sameples_delta, sameples_delta)
                # constraint on the maximal update step size
                tau_val = loss / norm_grad_W  # loss / (V*V');
                tau = np.minimum(self.aggress, tau_val)

                W += tau * grad_W

            # los[1, ii] = lt / (ii+1)
        return W, loss_steps_batch, los

    def fit(self, X, y, overwrite_X=False, overwrite_Y=False, verbose=False):
        """ AKA oasis.m
            Fit an OASIS Model
            X is data, y is class_labels
            因为‘=’在python中是引用而不是传值，所以会更改到原来数据集X的值"""

        if not overwrite_X:
            X = X.copy()  # deep copy
        if not overwrite_Y:
            y = y.copy()
        n_sapmles, n_featrues = X.shape

        """
        self.init = np.random.RandomState(self.random_seed)

        # Parameter initialization
        self._weights = np.eye(n_features).flatten()
        # self._weights = np.random.randn(n_features,n_features).flatten()
        W = self._weights.view()
        W.shape = (n_features, n_features)
        """

        self._W = np.eye(n_featrues)
        W = self._W.view()
        inds = np.argsort(y, axis=0)
        y = y[inds[:, 0]]  # sort y
        X = X[inds[:, 0], :]
        classes = np.unique(y)
        classes.sort()
        num_classes = len(classes)
        # Translate class labels to serial numbers 1,2.....
        y_new = np.empty(n_sapmles, dtype='int')
        for i in range(num_classes):
            temp = y == classes[i]
            y_new[temp[:, 0]] = i
        y = y_new
        class_sizes = np.empty(num_classes, dtype='int')
        class_start = np.empty(num_classes, dtype='int')
        for i in range(num_classes):
            class_sizes[i] = np.sum(y == i)
            class_start[i] = ((y == i) != 0).argmax(axis=0)
            # or class_start[i] = np.flatnonzero(y == i)[0]

        # Optimize
        loss_step = np.empty((self.n_iter,), dtype='bool')
        n_batches = int(np.ceil(self.n_iter / self.save_every))
        steps_vec = np.ones(n_batches - 1, dtype='int') * self.save_every
        steps_vec = np.append(steps_vec, self.n_iter - (n_batches - 1) * self.save_every)

        if verbose:
            print('n_batches = %d, total n_iter = %d' % (n_batches, self.n_iter))

        for i_batch in range(n_batches):
            # 对每个Batch进行训练
            if verbose:
                print('run batch %d/%d, for %d steps ("." = 100 steps)\n' \
                      % (i_batch + 1, n_batches, self.save_every))
            W, loss_steps_batch, los = self.train(W, X, y, class_start, class_sizes,
                                                  steps_vec[i_batch], verbose)
            loss_steps_batch[-1:self.save_every + 1:-1] = False
            # loss_step[(i_batch - 1) * self.save_every + 1:i_batch * self.save_every + 1] \
            # = loss_steps_batch #???
            loss_step[i_batch * self.save_every:min(
                (i_batch + 1) * self.save_every, self.n_iter)] = loss_steps_batch

            if self.do_save:
                pass

            if self.do_sym:
                if np.mod(i_batch + 1, self.sym_every) == 0 or i_batch == n_batches - 1:
                    if verbose:
                        print("Symmetrizing")
                    symmetrize(W)

            if self.do_psd:
                if np.mod(i_batch + 1, self.psd_every) == 0 or i_batch == n_batches - 1:
                    if verbose:
                        print("Making matrix PSD")
                    make_psd(W)
        self.loss_steps = loss_step
        self.los = los
        return self

    def predict(self, X_test, X_train, y_test, y_train, maxk=200, topk=5):
        '''
        Evaluate an OASIS model by KNN classification
        Examples are in rows
        maxk is the most number of X which is using for training
        topk is the k nearest neighbors
        '''
        W = self._W.view()
        maxk = min(maxk, X_train.shape[0])  # you can't train more X than X_train
        n_queries = X_test.shape[0]

        # compute the similarity scores
        s = np.dot(X_test, np.dot(W, X_train.T))

        # argsort sorts in ascending order
        # so we need to reverse the second dimension
        ind = np.argsort(s, axis=1)[:, ::-1]
        # s(ind)[i,j] 是第j大的p_i*W*P_k
        # Then depand on s(ind), vote for p_i
        queryvotes = y_train[ind[:, :maxk]]
        # 其中y_train会被广播到n行
        err_sum = np.empty((maxk,))
        for k in range(maxk):
            # bincount 用于计算投票的票数
            labels = np.empty(n_queries, dtype='int')
            for i in range(n_queries):
                b = np.bincount(queryvotes[i, :k + 1])
                labels[i] = np.argmax(b)
            err = y_test != labels
            err_sum[k] = sum(err)

        errrate = err_sum / n_queries
        return errrate


if __name__ == "__main__":
    pass
