# -*- coding: UTF-8 -*-
"""
written by Rui
file:LEGO.py
create time:2021/05/03
"""
import numpy as np

"""
    LEGO - Online metric learning (Jain et al., 2008a). LEGO learns a Mahalanobis distance
    in an online fashion using a regularized per instance loss, yielding a positive semidefinite
    matrix. The main variant of LEGO aims to fit a given set of pairwise distances. We used
    another variant of LEGO that, like OASIS, learns from relative distances. In our experimental
    setting, the loss is incurred for same-class examples being more than a certain distance away,
    and different class examples being less than a certain distance away. LEGO uses the LogDet
    divergence for regularization, as opposed to the Frobenius norm used in OASIS.
"""


class LEGO:

    def __init__(self, dim, C):
        self.A = np.eye(dim)
        T = C.shape[1]
        self.loss = np.zeros((1, T))
        self.mis = np.zeros((1, T))

    def train(self, X, C, A0, d):
        eta = 1e-2
        T = C.shape[1]
        loss = np.zeros((1, T))
        mis = np.zeros((1, T))
        lt = 0
        pos = 0
        neg = 0
        t_tick = 100
        verbose = 1
        err_count = 0; err_count_inbatch = 0; n_inbatch = 0
        t_tick_count = 0
        A = A0.copy()
        for t in range(T):
            u = X[:, int(C[0, t])-1]
            v = X[:, int(C[1, t])-1]
            z_t = u-v

            y_t_hat = np.dot(np.dot(z_t.T,A), z_t)
            y_t = C[3, t]
            b_t = C[2, t]

            ell = max(0, b_t*(y_t_hat-y_t))

            # count erros
            n_inbatch += 1
            if ell>0:
                err_count += 1
                err_count_inbatch += 1
                lt += ell**2
            if t>0:
                loss[0, t] = lt/t
                mis[0, t] = err_count/t

            #update
            if ell>0:
                y_t_bar = (eta*y_t*y_t_hat -1 + ((eta*y_t*y_t_hat -1)**2+4*eta*y_t_hat**2)**0.5 )/\
                          (2*eta*y_t_hat)
                A = A - (eta * (y_t_bar - y_t) * np.dot(np.dot(A, np.dot(z_t, z_t.T)), A))/\
                    (1 + eta * (y_t_bar - y_t) * y_t_hat)

            # save run time
            if (t % t_tick == 0 or t == T) and t>0 :
                if verbose:
                    print('t:%d/T:%d, n_err:%d, err_rate:%.4f;, loss:%.4f;'
                          '  n_inbatch:%d, errrate_inbatch:%.4f .' %
                          (t, T, err_count, err_count/t, loss[0, t],
                           n_inbatch, err_count_inbatch / n_inbatch))
                else:
                    print('.')
                n_inbatch = 0
                err_count_inbatch = 0

        self.loss = loss
        self.mis = mis
        self.A = A
        return self, A