# -*- coding: UTF-8 -*-
"""
written by Rui
file:EMF.py
create time:2021/05/05
"""
import numpy as np
import random

"""
Some Evaluation Measures Funtcion
"""

def ComputeDistanceExtremes(X, a, b, M):
    """ [l, u] = ComputeDistanceExtremes(X, a, b, M)

    Computes sample histogram of the distances between rows of X and returns
     the value of these distances at the a^th and b^th percentiles.  This
    method is used to determine the upper and lower bounds for
     similarity / dissimilarity constraints.

    X: (n x m) data matrix
    a: lower bound percentile between 1 and 100
    b: upper bound percentile between 1 and 100
    M: Mahalanobis matrix to compute distances
    Returns l: distance corresponding to a^th percentile
    u: distance corresponding the b^th percentile
    """
    try:
        if a < 1 or a > 100:
            raise ValueError('a must be between 1 and 100')
        if b < 1 or b > 100:
            raise ValueError('b must be between 1 and 100')
    except ValueError as e:
        print('Error：', repr(e))
        raise

    n = X.shape[0]
    num_trials = min(100, n * (n - 1) / 2)
    # we will sample with replacement
    dists = np.zeros((num_trials, 1))
    for i in range(num_trials):
        j1 = random.randint(0, n - 1)
        j2 = random.randint(0, n - 1)
        dists[i] = np.dot(np.dot(X[j1:j1+1, :]-X[j2:j2+1, :], M), (X[j1:j1+1, :]-X[j2:j2+1, :]).T)

    mi = np.min(dists)
    ma = np.max(dists)
    le = (ma - mi) / 100
    c = np.empty((100, 1))
    for i in range(100):
        c[i] = mi + 0.5 * le * (i + 1)
    l = c[int(np.floor(a))]
    u = c[int(np.floor(b))]
    return l, u


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