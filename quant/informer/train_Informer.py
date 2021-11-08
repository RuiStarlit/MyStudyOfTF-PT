# -*- coding:utf-8 -*-
"""
Writer: RuiStarlit
File: train_Informer
Project: informer
Create Time: 2021-11-03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import glob
import gc
import sys
import h5py
import datetime

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

# from utils import *
from models import *
from models.model import Informer, InformerStack

import warnings

warnings.filterwarnings('ignore')


def fr2(x, y):
    return 1 - torch.sum((x - y) ** 2) / (torch.sum((y - torch.mean(y)) ** 2) + 1e-4)


def frate(x, y):
    return torch.mean(torch.gt(x * y, 0).float())


class Train_Informer_mul():
    def __init__(self, enc_in, dec_in, c_out, seq_len, out_len, d_model, d_ff, n_heads,
                 e_layers, d_layers, label_len,
                 dropout, batch_size, lr, device, train_f, test_f):
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.out_len = out_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.label_len = label_len
        self.dropout = dropout
        self.Batch_size = batch_size
        self.lr = lr
        self.device = device
        self.train_f = train_f
        self.test_f = test_f

    def _build_model(self):
        model = Informer(enc_in=self.enc_in, dec_in=self.dec_in, c_out=self.c_out, out_len=self.out_len,
                         d_model=self.d_model, d_ff=self.d_ff, n_heads=self.n_heads, e_layers=self.e_layers,
                         d_layers=self.d_layers, label_len=self.label_len, dropout=self.dropout
                         )

        model.to(self.device)
        self.time = time.strftime("%m-%d-%H-%M", time.localtime())
        self.name = 'Informer-' + 'mutilstep-' + str(self.out_len) + 's' + self.time
        self.model = model
        print(self.model)

    def _selct_optim(self, opt='adam'):
        if opt == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif opt =='sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        else:
            raise NotImplementedError()

    def _selct_scheduler(self, patience=8, factor=0.8):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
                                                                    patience=patience, factor=factor)
    def _set_lr(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print('Learning Rate is set to '+str(lr)+'\n')

    def _selct_criterion(self, criterion):
        self.criterion = criterion

    def train_log(self, log):
        f = open('log/{}.txt'.format(self.name), 'a+')
        epoch, avg_loss, r2_a, val_aloss, r2_avg, rate_avg, lr = log
        f.write('Epoch:{:>3d} |Train_Loss:{:.6f} |R2:{:.6f}|Val_Loss:{:.6f} |R2:{:.6f} |Rate:{:.3f}|lr:{:.6f}\n'.format(
            epoch,
            avg_loss, r2_a, val_aloss, r2_avg, rate_avg, lr))
        f.close()

    def train_log_head(self):
        f = open('log/{}.txt'.format(self.name), 'a+')
        f.write("""The Hyperparameter:
        d_model = {} d_ff = {}
        n_heads = {} Batch_size = {} lr = {}
        label_len = {} dropout = {}
        e_layers = {}  d_layers = {}
          """.format(self.d_model, self.d_ff, self.n_heads, self.Batch_size, self.lr, self.label_len,
                     self.dropout, self.e_layers, self.d_layers))
        f.close()

    def write_log(self, train_loss, val_loss, train_r2, val_r2):
        plt.figure()
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Val Loss')
        plt.plot(train_r2, label='Train R2')
        plt.plot(val_r2, label='Val R2')
        plt.legend()
        plt.title(self.name)
        plt.ylabel('loss/R2')
        plt.xlabel('epoch')
        plt.savefig('log/{}.png'.format(self.name))

        min_train_loss = np.argmin(train_loss)
        min_val_loss = np.argmin(val_loss)
        max_r2 = np.argmax(train_r2)
        f = open('log/{}.txt'.format(self.name), 'a+')
        f.write("""
        Min Train Loss is {} at {}
        Min Test Loss is {} at {}
        Max R2 is {} at {}
        """.format(train_loss[min_train_loss], min_train_loss,
                   val_loss[min_val_loss], min_val_loss,
                   train_r2[max_r2], max_r2
                   ))
        f.close()

    def process_one_batch(self, batch_x, batch_y):
        batch_y = batch_y.float()
        # decoder input
        dec_inp = torch.zeros([batch_y.shape[0], self.out_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:, :self.label_len], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        outputs = self.model(batch_x, dec_inp)
        batch_y = batch_y[:, -self.out_len:, 0].to(self.device)

        return outputs, batch_y

    def single_train(self):
        self.model.train()
        train_loss = np.empty((len(self.dataset),))
        train_r2 = np.empty((len(self.dataset),))
        for i in range(len(self.dataset)):
            x, y = self.dataset(i)
            self.optimizer.zero_grad()
            pred, Y = self.process_one_batch(x, y)
            pred = pred.squeeze(2)

            loss = torch.mean((pred - Y) ** 2) + \
                   F.binary_cross_entropy_with_logits(pred, torch.gt(Y, 0).float())

            train_loss[i] = loss.item()
            loss.backward()
            self.optimizer.step()

            r2 = fr2(pred[:, 0], Y[:, 0]).cpu().detach().numpy()
            train_r2[i] = r2

        train_loss = train_loss.mean()
        train_r2 = train_r2.mean()
        return train_loss, train_r2

    def train_one_epoch(self, train_all=True, f=None):
        if not train_all:
            self.dataset = MyDataset(file_name=f, batch_size=self.Batch_size,
                                     pred_len=self.out_len, label_len=self.label_len)
            loss, r2 = self.single_train()
        else:
            loss = np.empty((len(self.train_f),))
            r2 = np.empty((len(self.train_f),))
            conter = 0
            for f in self.train_f:
                self.dataset = MyDataset(file_name=f, batch_size=self.Batch_size,
                                         pred_len=self.out_len, label_len=self.label_len)
                train_loss, tran_r2 = self.single_train()
                loss[conter] = train_loss
                r2[conter] = tran_r2
                conter += 1
                del (self.dataset)
            loss = loss.mean()
            r2 = r2.mean()
        return loss, r2

    def val(self, val_all=False, f=None):
        self.model.eval()
        if not val_all:
            dataset = MyDataset(file_name=f, batch_size=self.Batch_size,
                                pred_len=self.out_len, label_len=self.label_len)
            val_loss = np.empty((len(dataset),))
            val_r2 = np.empty((len(dataset),))
            val_rate = np.empty((len(dataset),))
            conter = 0
            with torch.no_grad():
                for i in range(len(dataset)):
                    x, y = dataset(i)
                    pred, Y = self.process_one_batch(x, y)
                    pred = pred.squeeze(2)
                    loss = torch.mean((pred - Y) ** 2) + \
                           F.binary_cross_entropy_with_logits(pred, torch.gt(Y, 0).float())
                    val_loss[conter] = loss.item()
                    r2 = fr2(pred[:, 0], Y[:, 0]).cpu().detach().numpy()
                    rate = frate(pred[:, 0], Y[:, 0]).detach().cpu().numpy()
                    val_r2[conter] = r2
                    val_rate[conter] = rate
            val_loss = val_loss.mean()
            val_r2 = val_r2.mean()
            val_rate = val_rate.mean()
            return val_loss, val_r2, val_rate

        else:
            t_val_loss = np.empty((len(self.test_f),))
            t_val_r2 = np.empty((len(self.test_f), 1))
            t_val_rate = np.empty((len(self.test_f), 1))
            conter = 0
            for f in self.test_f:
                dataset = MyDataset(file_name=f, batch_size=self.Batch_size,
                                    pred_len=self.out_len, label_len=self.label_len)

                val_loss = np.empty((len(dataset),))
                val_r2 = np.empty((len(dataset),))
                val_rate = np.empty((len(dataset),))

                with torch.no_grad():
                    for i in range(len(dataset)):
                        x, y = dataset(i)
                        pred, Y = self.process_one_batch(x, y)
                        pred = pred.squeeze(2)
                        loss = torch.mean((pred - Y) ** 2) + \
                               F.binary_cross_entropy_with_logits(pred, torch.gt(Y, 0).float())
                        val_loss[i] = loss.item()
                        r2 = fr2(pred[:, 0], Y[:, 0]).cpu().detach().numpy()
                        rate = frate(pred[:, 0], Y[:, 0]).detach().cpu().numpy()
                        val_r2[i] = r2
                        val_rate[i] = rate
                val_loss = val_loss.mean()
                val_r2 = val_r2.mean()
                val_rate = val_rate.mean()
                t_val_loss[conter] = val_loss
                t_val_r2[conter] = val_r2
                t_val_rate[conter] = val_rate
                conter += 1
                del (dataset)
            val_loss = t_val_loss.mean()
            val_r2 = t_val_r2.mean()
            val_rate = t_val_rate.mean()

            return val_loss, val_r2, val_rate, t_val_loss, t_val_r2, t_val_rate

    def train(self, epochs=200, train_all=True, f=None, val_all=False, testfile=None, save='train'):
        best_train_r2 = float('-inf')
        best_val_r2 = float('-inf')
        self.train_log_head()

        train_loss = np.empty((epochs,))
        train_r2 = np.empty((epochs,))
        val_loss = np.empty((epochs,))
        val_r2 = np.empty((epochs,))
        val_rate = np.empty((epochs,))
        for epoch in tqdm(range(epochs)):
            loss, r2 = self.train_one_epoch(train_all, f)
            train_loss[epoch] = loss
            train_r2[epoch] = r2
            self.scheduler.step(loss)

            if not val_all:
                loss, r2, rate = self.val(val_all, testfile)
            else:
                loss, r2, rate, tloss, tr2, trate = self.val(val_all, testfile)
                max_r2 = np.argmax(tr2)
                print('The max r2 is test_f is' + str(tr2[max_r2]) + ' at' + str(max_r2))

            val_loss[epoch] = loss
            val_r2[epoch] = r2
            val_rate[epoch] = rate

            if save == 'train':
                if train_r2[epoch] > best_train_r2:
                    best_train_r2 = train_r2[epoch]
                    torch.save(self.model.state_dict(), 'checkpoint/' + self.name + '.pt')
                    print('Save here')
                    file = open('log/{}.txt'.format(self.name), 'a+')
                    file.write('Save here'+'\n')
                    file.close()
            elif save == 'test':
                if val_r2[epoch] > best_val_r2:
                    best_val_r2 = val_r2[epoch]
                    torch.save(self.model.state_dict(), 'checkpoint/' + self.name + '.pt')
                    print('Save here')
                    file = open('log/{}.txt'.format(self.name), 'a+')
                    file.write('Save here'+'\n')
                    file.close()
            else:
                raise NotImplementedError()

            print(
                'Epoch:{:>3d} |Train_Loss:{:.6f} |R2:{:.6f}|Val_Loss:{:.6f} |R2:{:.6f} |Rate:{:.3f} |lr:{:.6f}'.format(
                    epoch + 1,
                    train_loss[epoch], train_r2[epoch], val_loss[epoch], val_r2[epoch], val_rate[epoch],
                    self.optimizer.state_dict()['param_groups'][0]['lr']))
            log = [epoch + 1, train_loss[epoch], train_r2[epoch], val_loss[epoch], val_r2[epoch],
                   val_rate[epoch],
                   self.optimizer.state_dict()['param_groups'][0]['lr']]
            self.train_log(log)
            self.lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        print("Done")
        self.write_log(train_loss, val_loss, train_r2, val_r2)

    def test(self, ic_name):
        dataset = MyDataset(ic_name, self.Batch_size, self.out_len, label_len=self.label_len)
        r2 = np.empty((len(dataset),))
        test_loss = np.empty((len(dataset),))
        pred_list = []
        y_list = []

        self.model.eval()
        for i in range(len(dataset)):
            x, y = dataset(i)
            with torch.no_grad():
                pred, Y = self.process_one_batch(x, y)
                loss = torch.mean((pred - Y) ** 2) + \
                       F.binary_cross_entropy_with_logits(pred, torch.gt(Y, 0).float())
                test_loss[i] = loss.item()
            pred = pred.squeeze(2)
            r2_i = fr2(pred[:, 0], Y[:, 0]).cpu().detach().numpy()
            r2[i] = r2_i
            pred_list.append(pred[:, 0].detach().cpu().numpy())
            y_list.append(Y[:, 0].detach().cpu().numpy())

        return r2, test_loss, pred_list, y_list

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        print("success")


class MyDataset():
    def __init__(self, file_name, batch_size, pred_len=3, enc_seq_len=20, label_len=1):
        self.name = file_name
        self.__read_data__()
        self.batch_size = batch_size
        self.n = self.y.shape[0]
        self.indexes = np.arange(self.n)
        self.mask = list(range(15))  # [1, 3, 4, 5, 6, 7, 8, 9]# [0, 2, 10, 11, 12, 13, 14]
        self.enc_seq_len = 20
        self.label_len = label_len
        #         print(self.y.shape)
        #         self.ts = f['timestamp'][:]
        self.index = 0
        self.shift = 10
        self.device = 'cuda'
        self.pred_len = pred_len - 1

    def __read_data__(self):
        f = h5py.File(self.name, 'r')
        self.x = f['x'][:]
        self.y = f['y'][:]
        f.close()

    def __call__(self, i):
        batch_index = self.indexes[i * self.batch_size: (i + 1) * self.batch_size]
        y_len = batch_index.shape[0]

        # 向过去取历史时间序列
        if i == 0:
            temp = np.zeros((self.label_len + self.shift, 1))
            Y1 = self.y[batch_index]
            temp = np.concatenate((temp, Y1))
            for i in range(self.label_len):
                Y1 = np.hstack((temp[-1 - i - self.shift - self.batch_size: -1 - i - self.shift], Y1))
        else:
            r_index = self.indexes[i * self.batch_size - self.label_len - self.shift: (i + 1) * self.batch_size]
            temp = self.y[r_index]
            Y1 = self.y[batch_index]
            for i in range(self.label_len):
                Y1 = np.hstack((temp[-1 - i - self.shift - y_len:-1 - i - self.shift], Y1))
        # 向未来取趋势时间序列
        if i >= int(np.ceil(self.n / self.batch_size)):
            temp = np.full((self.pred_len, 1), self.y[-1, 0])
            Y2 = self.y[batch_index]
            temp = np.concatenate((Y2, temp))
            for i in range(self.pred_len):
                Y2 = np.hstack((Y2, temp[1 + i: 1 + i + y_len + 1]))

        else:
            r_index = self.indexes[i * self.batch_size: (i + 1) * self.batch_size + self.pred_len]
            temp = self.y[r_index]
            Y2 = self.y[batch_index]
            for i in range(self.pred_len):
                Y2 = np.hstack((Y2, temp[1 + i: 1 + i + y_len]))
        Y = np.hstack((Y1[:, :-1], Y2))

        X = self.x[batch_index, -self.enc_seq_len:, :]
        Y = Y[:, :, np.newaxis]
        X = torch.from_numpy(X).to(self.device).float()
        Y = torch.from_numpy(Y)
        return X, Y

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __del__(self):
        del self.x, self.y, self.indexes
