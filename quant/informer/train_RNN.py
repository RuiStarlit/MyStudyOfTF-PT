# -*- coding:utf-8 -*-
"""
Writer: RuiStarlit
File: train_RNN
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

from utils import *
# from Network import *

import warnings

warnings.filterwarnings('ignore')


def fr2(x, y):
    return 1 - torch.sum((x - y) ** 2) / (torch.sum((y - torch.mean(y)) ** 2) + 1e-4)


def frate(x, y):
    return torch.mean(torch.gt(x * y, 0).float())


class Train_RNN():
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

    def _build_model(self, opt='rnn'):
        if opt == 'attn_rnn':
            model = AttentionRNN(self.d_model, self.d_ff, self.enc_in, self.d_layers, self.n_heads, self.dropout)
        elif opt == 'rnn':
            model = RNN(self.d_model, self.d_ff, self.enc_in, self.d_layers, self.n_heads,self.dropout)
        elif opt == 'LSTM':
            model = MyLSTM(self.d_model, self.d_ff, self.enc_in, self.d_layers, self.n_heads, self.dropout)
        elif opt == 'gru':
            model = GRU(self.d_model, self.d_ff, self.enc_in, self.d_layers, self.n_heads, self.dropout)
        elif opt == 'attn_gru':
            model = AttentionGRU(self.d_model, self.d_ff, self.enc_in, self.d_layers, self.n_heads, self.dropout)
        else:
            raise NotImplementedError()

        model.to(self.device)
        self.time = time.strftime("%m-%d-%H-%M", time.localtime())
        self.name = 'LSTM-' + 'mutilstep-' + str(self.out_len) + 's' + self.time
        self.model = model
        print(self.model)

    def _selct_optim(self, opt='adam'):
        if opt == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif opt == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        else:
            raise NotImplementedError()

    def _selct_scheduler(self, patience=8, factor=0.8):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
                                                                    patience=patience, factor=factor)

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
            pred = self.model(x)

            loss = torch.mean((pred - y) ** 2) + \
                   F.binary_cross_entropy_with_logits(pred, torch.gt(y, 0).float())

            train_loss[i] = loss.item()
            loss.backward()
            self.optimizer.step()

            r2 = fr2(pred, y).cpu().detach().numpy()
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
                    pred = self.model(x)
                    loss = torch.mean((pred - y) ** 2) + \
                           F.binary_cross_entropy_with_logits(pred, torch.gt(y, 0).float())
                    val_loss[conter] = loss.item()
                    r2 = fr2(pred, y).cpu().detach().numpy()
                    rate = frate(pred, y).detach().cpu().numpy()
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
                        pred = self.model(x)
                        loss = torch.mean((pred - y) ** 2) + \
                               F.binary_cross_entropy_with_logits(pred, torch.gt(y, 0).float())
                        val_loss[i] = loss.item()
                        r2 = fr2(pred, y).cpu().detach().numpy()
                        rate = frate(pred, y).detach().cpu().numpy()
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
            elif save == 'test':
                if val_r2[epoch] > best_val_r2:
                    best_val_r2 = val_r2[epoch]
                    torch.save(self.model.state_dict(), 'checkpoint/' + self.name + '.pt')
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
                pred = self.model(x)
                loss = torch.mean((pred - y) ** 2) + \
                       F.binary_cross_entropy_with_logits(pred, torch.gt(y, 0).float())
                test_loss[i] = loss.item()
            r2_i = fr2(pred, y).cpu().detach().numpy()
            r2[i] = r2_i
            pred_list.append(pred.detach().cpu().numpy())
            y_list.append(y.detach().cpu().numpy())

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
        self.enc_seq_len = enc_seq_len
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
        X = self.x[batch_index, -self.enc_seq_len:, :]
        Y = self.y[batch_index]
        X = torch.tensor(X[:, :, np.array(self.mask)]).cuda()
        # X = torch.tensor(X).cuda()
        Y = torch.tensor(Y).cuda()

        return X, Y

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __del__(self):
        del self.x, self.y, self.indexes


class BoostRNN(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, n_layers, n_heads):
        super(BoostRNN, self).__init__()
        self.n_layers = n_layers
        self.dim_val = dim_val

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=dim_val, num_layers=n_layers, bidirectional=False,
                           dropout=0.2, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=7, hidden_size=dim_val, num_layers=n_layers, bidirectional=False, dropout=0.2,
                            batch_first=True)

        self.fc0 = nn.Linear(input_size, dim_val * 2)
        self.fc1 = nn.Linear(dim_val * 2, dim_val * 2)
        self.fc2 = nn.Linear(dim_val * 2, 1)

        self.fc3 = nn.Linear(dim_val * 2, dim_val * 2)
        self.fc4 = nn.Linear(dim_val * 2, 1)

        self.alpha = torch.nn.parameter.Parameter(torch.Tensor(1))

    def forward(self, x):
        xx = x[:, :, [0, 2, 10, 11, 12, 13, 14]]

        # x = self.fc0(x)
        x, (hn, cn) = self.rnn(x)
        hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        x = self.fc2(F.elu(self.fc1(hn)))

        xx, (hn, cn) = self.rnn2(xx)
        hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        xx = self.fc4(F.elu(self.fc3(hn)))

        x = torch.sigmoid(self.alpha) * x + (1 - torch.sigmoid(self.alpha)) * xx
        return x


class RNN(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, n_layers, n_heads, dropout):
        super(RNN, self).__init__()
        self.n_layers = n_layers
        self.dim_val = dim_val

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=dim_val, num_layers=n_layers, bidirectional=False,
                           dropout=dropout, batch_first=True)

        self.fc1 = nn.Linear(dim_val * 2, dim_val * 2)
        self.fc2 = nn.Linear(dim_val * 2, 1)

        # self.fc3 = nn.Linear(dim_val * 2, dim_val * 2)
        # self.fc4 = nn.Linear(dim_val * 2, 1)

    def forward(self, x):
        # x = self.fc0(x)
        x, (hn, cn) = self.rnn(x)
        hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        x = self.fc2(F.elu(self.fc1(hn)))
        # xx = self.fc4(F.elu(self.fc3(hn)))
        return x


class SelectRNN(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, n_layers, n_heads):
        super(SelectRNN, self).__init__()
        self.n_layers = n_layers
        self.dim_val = dim_val

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=dim_val, num_layers=n_layers, bidirectional=False,
                           dropout=0.2, batch_first=True)

        self.fc1 = nn.Linear(dim_val * 2, dim_val * 2)
        self.fc2 = nn.Linear(dim_val * 2, 1)

        self.key1 = Key(input_size, input_size)
        self.query1 = Query(input_size, input_size)

        self.key2 = Key(input_size, input_size)
        self.query2 = Query(input_size, input_size)

    def forward(self, x):
        m1 = torch.matmul(self.query1(x), self.key1(x).transpose(2, 1).float())
        m2 = torch.matmul(self.query2(x).transpose(2, 1).float(), self.key2(x))
        p1 = torch.softmax(torch.mean(m1, -1), -1).unsqueeze(2)
        p2 = torch.softmax(torch.mean(m2, -1), -1).unsqueeze(1)

        x = x * p2
        x, (hn, cn) = self.rnn(x)
        hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        x = self.fc2(F.elu(self.fc1(hn)))
        return x, (p1, p2)


class AttentionRNN(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, n_layers, n_heads,dropout):
        super(AttentionRNN, self).__init__()
        self.n_layers = n_layers
        self.dim_val = dim_val

        self.fc1 = nn.Linear(input_size, dim_val)

        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.norm = nn.LayerNorm(dim_val)
        self.rnn = nn.LSTM(input_size=dim_val, hidden_size=dim_val, num_layers=n_layers, bidirectional=False,
                           dropout=dropout, batch_first=True)

        self.fc2 = nn.Linear(dim_val * 2, dim_val * 2)
        self.fc3 = nn.Linear(dim_val * 2, 1)

    def forward(self, x):
        x = self.fc1(x)
        a = self.attn(x)
        x = self.norm(a + x)

        x, (hn, cn) = self.rnn(x)
        hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        x = self.fc3(F.elu(self.fc2(hn)))
        return x


class CNN(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, n_layers, n_heads):
        super(CNN, self).__init__()
        self.n_layers = n_layers
        self.dim_val = dim_val

        self.conv = nn.Conv2d(input_size, dim_val, (1, 1))

        self.conv1 = nn.Conv2d(input_size, dim_val, (3, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(input_size, dim_val, (5, 1), padding=(2, 0))
        self.conv3 = nn.Conv2d(input_size, dim_val, (7, 1), padding=(3, 0))

        self.pool = nn.AdaptiveMaxPool2d((1, 1))

        self.pool1 = nn.AdaptiveMaxPool2d((1, 1))
        self.pool2 = nn.AdaptiveMaxPool2d((1, 1))
        self.pool3 = nn.AdaptiveMaxPool2d((1, 1))

        self.bn = nn.BatchNorm2d(dim_val)

        self.bn1 = nn.BatchNorm2d(dim_val)
        self.bn2 = nn.BatchNorm2d(dim_val)
        self.bn3 = nn.BatchNorm2d(dim_val)

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(dim_val * 3, 1)
        self.fc1 = nn.Linear(dim_val * 3, dim_val)
        self.fc2 = nn.Linear(dim_val, 1)

    def forward(self, x):
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)

        x1 = F.elu(self.conv1(x))
        x2 = F.elu(self.conv2(x))
        x3 = F.elu(self.conv3(x))

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.pool(x)

        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        # x = self.fc2(F.tanh(self.fc1(x)))
        x = self.fc(x)
        return x


class SimpleCNN(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, n_layers, n_heads):
        super(SimpleCNN, self).__init__()
        self.n_layers = n_layers
        self.dim_val = dim_val

        self.conv1 = nn.Conv2d(1, dim_val, (3, 3))
        self.conv2 = nn.Conv2d(dim_val, dim_val * 2, (3, 3))
        self.conv3 = nn.Conv2d(dim_val * 2, dim_val * 2, (3, 3))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.bn1 = nn.BatchNorm2d(dim_val)
        self.bn2 = nn.BatchNorm2d(dim_val * 2)
        self.bn3 = nn.BatchNorm2d(dim_val * 2)

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(dim_val * 2, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.bn1(F.elu(self.conv1(x)))
        x = self.bn2(F.elu(self.conv2(x)))
        x = self.bn3(F.elu(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.fc(x.flatten(start_dim=1))
        return x


class GRU(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, n_layers, n_heads, dropout):
        super(GRU, self).__init__()
        self.n_layers = n_layers
        self.dim_val = dim_val

        self.rnn = nn.GRU(input_size=input_size, hidden_size=dim_val, num_layers=n_layers, bidirectional=False,
                          dropout=dropout, batch_first=True)

        self.fc1 = nn.Linear(dim_val * 2, dim_val * 2)
        self.fc2 = nn.Linear(dim_val * 2, 1)

        # self.fc3 = nn.Linear(dim_val * 2, dim_val * 2)
        # self.fc4 = nn.Linear(dim_val * 2, 1)

    def forward(self, x):
        # x = self.fc0(x)
        x, hn, = self.rnn(x)
        hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        x = self.fc2(F.elu(self.fc1(hn)))
        # xx = self.fc4(F.elu(self.fc3(hn)))
        return x


class AttentionGRU(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, n_layers, n_heads,dropout):
        super(AttentionGRU, self).__init__()
        self.n_layers = n_layers
        self.dim_val = dim_val

        self.fc1 = nn.Linear(input_size, dim_val)

        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.norm = nn.LayerNorm(dim_val)
        self.rnn = nn.GRU(input_size=dim_val, hidden_size=dim_val, num_layers=n_layers, bidirectional=False,
                           dropout=dropout, batch_first=True)

        self.fc2 = nn.Linear(dim_val * 2, dim_val * 2)
        self.fc3 = nn.Linear(dim_val * 2, 1)

    def forward(self, x):
        x = self.fc1(x)
        a = self.attn(x)
        x = self.norm(a + x)

        x, hn = self.rnn(x)
        hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        x = self.fc3(F.elu(self.fc2(hn)))
        return x


class MyLSTM(nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, n_layers, n_heads, dropout):
        super(MyLSTM, self).__init__()

        self.hidden_size = dim_val
        self.n_layers = n_layers
        self.input_size = input_size
        self.w_omega = nn.Parameter(torch.Tensor(dim_val, dim_val))
        self.u_omega = nn.Parameter(torch.Tensor(dim_val, 1))
        nn.init.uniform_(self.w_omega, -0.01, 0.01)
        nn.init.uniform_(self.u_omega, -0.01, 0.01)

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=dim_val,
                            num_layers=n_layers
                            )
        self.flat = nn.Flatten()
        self.relu = nn.ReLU()
        self.attn = self.attention
        self.fc1 = nn.Linear(dim_val * 21, dim_val * 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim_val * 4, 1)

    def attention(self, x):
        u = torch.tanh(torch.matmul(x, self.w_omega))
        attn = torch.matmul(u, self.u_omega)
        attn_score = F.softmax(attn, dim=1)
        scored_x = x * attn_score
        context = torch.sum(scored_x, dim=1)
        return context

    def forward(self, x):
        x.transpose_(1, 0)
        x, (h_t, c_t) = self.lstm(x)
        x = x.permute(1, 0, 2)
        attn_output = self.attn(x)
        x = self.flat(x)
        x = torch.cat((attn_output, x), 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
