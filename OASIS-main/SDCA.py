# -*- coding: UTF-8 -*-
"""
written by Rui
file:SDCA.py
create time:2021/05/03
"""
import numpy as np
import random

class sdca:

    def __init__(self, dim, T):
        self.M = np.zeros(dim, dtype='int')
        self.mistake = np.zeros((1, T), dtype='int')
        self.loss = np.zeros((1, T), dtype='int')

    def train(self, data, L, T, eta):
        N, dim = data.shape
        M = np.zeros((dim, dim), dtype='float')
        M_s = np.zeros((dim, dim), dtype='float')
        M_t = np.zeros((dim, dim), dtype='float')
        alpha = np.zeros((1, N), dtype='float')
        delta_alpha = np.zeros((1, N), dtype='float')
        lamb = eta / N     # AKA lambda
        pos = 0
        neg = 0
        mistake = np.zeros((1, T), dtype='float')
        loss = np.zeros((1, T), dtype='float')
        #lgap = np.zeros((1, T), dtype='int')
        #alp = 0
        lt = 0
        #lm = 0
        verbose = 1
        t_tick = 1000
        epoch = round(T/N)
        #gap = np.zeros((1, epoch), dtype='int')

        inds = np.argsort(L, axis=0)[:, 0]
        class_labels = L[inds]
        data = data[inds, :]
        classes = np.unique(class_labels)
        num_classes = len(classes)

        #Translate class labels to serial numbers 1,2,\cdots
        new_class_label = np.empty(class_labels.shape, dtype='int')
        for i in range(num_classes):
            temp = class_labels == classes[i]
            new_class_label[temp[:, 0]] = i
        class_labels = new_class_label
        class_sizes = np.empty((num_classes, 1), dtype='int')
        class_start = np.empty((num_classes, 1), dtype='int')
        for i in range(num_classes):
            class_sizes[i] = np.sum(class_labels == i)
            class_start[i] = ((class_labels == i) != 0).argmax(axis=0)

        # n_iters = num_steps
        # loss_steps_batch = np.zeros(num_steps, 1)
        num_objects = len(class_labels)

        for t in range(T):
            # Sample a query iamge
            p_ind = random.randint(0, num_objects - 1)  # 可以调用self.init作为随机数种子
            clas = class_labels[p_ind, 0]
            pos_ind = class_start[clas] + random.randint(0,class_sizes[clas]-1)
            neg_ind = random.randint(0, num_objects-1)

            while class_labels[neg_ind] == clas:
                neg_ind = random.randint(0, num_objects-1)

            p = data[p_ind:p_ind+1, :]
            samples_delta = data[pos_ind] - data[neg_ind]
            los = 1 - np.dot(np.dot(p, M), samples_delta.T)

            X_w = np.dot(p.T, samples_delta)
            norm_xw = np.linalg.norm(X_w)**2    # default norm is Frobenius norm
            f_t = (los-alpha[0, p_ind]/2)/(0.5+norm_xw/(lamb*N))
            delta_alpha[0, p_ind] = max(f_t, -alpha[0, p_ind])
            alpha[0, p_ind] = alpha[0, p_ind] + delta_alpha[0, p_ind]
            M += delta_alpha[0, p_ind]*X_w / (lamb*N)

            if t>T/2:
                M_t = M_t + M

            if 1-los >= 0:
                pos += 1
            else:
                neg += 1
            mistake[0, t] = neg/(pos+neg)

            if los>0:
                lt += los**2

            if t>0:
                loss[0, t] = lt/t


            M_s += M
            # if (alpha(1, p_ind) >= 0)
                # alp = alp + alpha(1, p_ind) - 0.25 * alpha(1, p_ind) ^ 2;
            # end

            # gap(1, t) = lt / t + lamb *norm(M, 'fro') ^ 2 / 2 - alp / t + ...
            # lambda /2 * norm(p / (lambda *t), 'fro') ^ 2;
            # lgap(1, t) = lt / t + 0.5 *...
            #lambda *norm(M, 'fro') ^ 2 - alp / t + 0.5 *...
            #lambda *norm(M, 'fro') ^ 2;
            # lm = lm + lgap(1, t);
            # gap(1, t) = lt / T + (lambda /2) * norm(M, 'fro') ^ 2 - alp / T +...
            #lambda /2 * norm(p / (lambda *T), 'fro') ^ 2;

            # if (mod(t, N) == 0)
                # gap(1, t / N) = lm / t;
            # gap(1, t / N) = lt / t + 0.5 *...
            #lambda *norm(M, 'fro') ^ 2 - alp / t + 0.5 *...
            #lambda *norm(M, 'fro');
            # end
            if t % t_tick == 0 or t == T:
                if verbose:
                    print('t:%d/T:%d, n_err:%d, err_rate:%.4f; loss:%.4f .'
                          % (t, T, neg, mistake[0, t], loss[0, t]))
                else:
                    print('.')

        M_m = (2/T)*M_t
        self.M = M_m
        self.mistake = mistake
        self.loss = loss
        return self

