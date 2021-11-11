# -*- coding:utf-8 -*-
"""
Writer: RuiStarlit
File: train_RNN_t
Project: informer
Create Time: 2021-11-09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import glob
import gc
import sys
import h5py
import datetime

import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils import *
# from Network import *
from Mytools import EarlyStopping_R2

import warnings

warnings.filterwarnings('ignore')


def fr2(x, y):
    return 1 - torch.sum((x - y) ** 2) / (torch.sum((y - torch.mean(y)) ** 2) + 1e-4)


def frate(x, y):
    return torch.mean(torch.gt(x * y, 0).float())


def get_tick_hour_minute_str(tick):
    dt = datetime.datetime(1, 1, 1) + datetime.timedelta(microseconds=tick / 10)
    return dt.strftime("%H%M")


def get_tick_date_time(tick):
    dt = datetime.datetime(1, 1, 1) + datetime.timedelta(microseconds=tick / 10)
    return dt


def get_tick_weekday(tick):
    dt = datetime.datetime(1, 1, 1) + datetime.timedelta(microseconds=tick / 10)
    return dt.weekday()


def encode_time(dt):
    hm = int(dt.strftime("%H%M"))
    if hm <= 1130:
        hm = 0
    elif 1300 <= hm and hm <= 1429:
        hm = 1
    elif 1430 <= hm:
        hm = 2
    else:
        raise ValueError('时间数据出错')
    wd = dt.weekday()
    return wd * 10 + hm


get_hour_minute_str_ = np.frompyfunc(get_tick_hour_minute_str, 1, 1)
get_tick_date_time_ = np.frompyfunc(get_tick_date_time, 1, 1)
get_tick_weekday_ = np.frompyfunc(get_tick_weekday, 1, 1)
encode_time_ = np.frompyfunc(encode_time, 1, 1)

# Global Variable
categories = [np.array([0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42],
                       dtype=object)]

is_scaler = False
scalers = []




class Train_RNN():
    def __init__(self, enc_in, dec_in, c_out, seq_len, out_len, d_model, d_ff, n_heads,
                 e_layers, d_layers, label_len,
                 dropout, batch_size, val_batch, lr, device, train_f, test_f, scaler, decay):
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
        self.print_r2 = False
        self.epoch = 0
        self.val_batch = val_batch
        self.scaler = scaler
        self.boost_index = {}
        self.clip_grad = True
        self.decay = 3000

    def _build_model(self, opt='attn_rnn'):
        if opt == 'attn_rnn':
            model = AttentionRNN(self.d_model, self.d_ff, self.enc_in, self.d_layers, self.n_heads, self.dropout)
        # elif opt == 'rnn':
        #     model = RNN(self.d_model, self.d_ff, self.enc_in, self.d_layers, self.n_heads, self.dropout)
        # elif opt == 'LSTM':
        #     model = MyLSTM(self.d_model, self.d_ff, self.enc_in, self.d_layers, self.n_heads, self.dropout)
        # elif opt == 'attn_gru':
        #     model = AttentionGRU(self.d_model, self.d_ff, self.enc_in, self.d_layers, self.n_heads, self.dropout)
        elif opt == 'gru':
            model = GRU(self.d_model, self.d_ff, self.enc_in, self.d_layers, self.n_heads, self.dropout)
        else:
            raise NotImplementedError()

        model.to(self.device)
        self.time = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%m-%d_%H:%M")
        self.name = opt + self.time
        self.model = model
        print(self.model)

    def _selct_optim(self, opt='adam'):
        if opt == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif opt == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        else:
            raise NotImplementedError()

    def _selct_scheduler(self, patience=8, factor=0.8, min_lr=0.00001):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
                                                                    patience=patience, factor=factor, min_lr=min_lr)

    def _set_lr(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print('Learning Rate is set to ' + str(lr))

    def _selct_criterion(self, criterion):
        self.criterion = criterion

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        print("success")

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

    def write_remarks(self, s):
        f = open('log/{}.txt'.format(self.name), 'a+')
        f.write(s + '\n')
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
            x, y, t = self.dataset(i)
            self.optimizer.zero_grad()
            pred = self.model(x, t)

            loss = torch.mean((pred - y) ** 2)
                   # F.binary_cross_entropy_with_logits(pred, torch.gt(y, 0).float())

            train_loss[i] = loss.item()
            loss.backward()
            if self.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            self.optimizer.step()

            if i % self.decay == 0:
                self.scheduler.step(loss)

            r2 = fr2(pred, y).cpu().detach().numpy()
            train_r2[i] = r2

        train_loss = train_loss.mean()
        train_r2 = train_r2.mean()
        return train_loss, train_r2

    def train_one_epoch(self, train_all=True, f=None):
        if not train_all:
            self.dataset = MyDataset(file_name=f, batch_size=self.Batch_size,
                                     pred_len=self.out_len, label_len=self.label_len, scaler=self.scaler)
            loss, r2 = self.single_train()
        else:
            loss = np.empty((len(self.train_f),))
            r2 = np.empty((len(self.train_f),))
            conter = 0
            for f in self.train_f:
                self.dataset = MyDataset(file_name=f, batch_size=self.Batch_size,
                                         pred_len=self.out_len, label_len=self.label_len, scaler=self.scaler)
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
            dataset = MyDataset(file_name=f, batch_size=self.val_batch,
                                pred_len=self.out_len, label_len=self.label_len, scaler=self.scaler)
            val_loss = np.empty((len(dataset),))
            val_r2 = np.empty((len(dataset),))
            val_rate = np.empty((len(dataset),))
            conter = 0
            with torch.no_grad():
                for i in range(len(dataset)):
                    x, y, t = dataset(i)
                    pred = self.model(x, t)
                    loss = torch.mean((pred - y) ** 2)
                           # F.binary_cross_entropy_with_logits(pred, torch.gt(y, 0).float())
                    val_loss[conter] = loss.item()
                    r2 = fr2(pred, y).cpu().detach().numpy()
                    rate = frate(pred, y).detach().cpu().numpy()
                    val_r2[conter] = r2
                    val_rate[conter] = rate
            val_loss = val_loss.mean()
            val_r2 = val_r2.mean()
            val_rate = val_rate.mean()

        else:
            t_val_loss = np.empty((len(self.test_f),))
            t_val_r2 = np.empty((len(self.test_f), 1))
            t_val_rate = np.empty((len(self.test_f), 1))
            conter = 0
            for f in self.test_f:
                dataset = MyDataset(file_name=f, batch_size=self.val_batch,
                                    pred_len=self.out_len, label_len=self.label_len, scaler=self.scaler)

                val_loss = np.empty((len(dataset),))
                val_r2 = np.empty((len(dataset),))
                val_rate = np.empty((len(dataset),))

                with torch.no_grad():
                    for i in range(len(dataset)):
                        x, y, t = dataset(i)
                        pred = self.model(x, t)
                        loss = torch.mean((pred - y) ** 2)
                               # F.binary_cross_entropy_with_logits(pred, torch.gt(y, 0).float())
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
            if self.print_r2:
                max_r2 = np.argmax(t_val_r2)
                print('The max r2 is test_f is' + str(t_val_r2[max_r2]) + ' at' + str(max_r2))

        return val_loss, val_r2, val_rate

    def train(self, epochs=200, train_all=True, f=None, val_all=False, testfile=None, save='train', continued=0, patience=15, boost=False):
        if boost:
            print('Training Mode: boost')
        best_train_r2 = float('-inf')
        best_val_r2 = float('-inf')
        self.train_log_head()
        early_stopping = EarlyStopping_R2(patience=patience, verbose=True)

        train_loss = np.empty((epochs,))
        train_r2 = np.empty((epochs,))
        val_loss = np.empty((epochs,))
        val_r2 = np.empty((epochs,))
        val_rate = np.empty((epochs,))
        start_epoch = self.epoch + continued
        for epoch in tqdm(range(epochs)):
            self.epoch = start_epoch + epoch
            if not boost:
                loss, r2 = self.train_one_epoch(train_all, f)
            else:
                loss, r2 = self.boost_train_one_epoch(train_all, f)
            train_loss[epoch] = loss
            train_r2[epoch] = r2
            # self.scheduler.step(loss)

            loss, r2, rate = self.val(val_all, testfile)

            val_loss[epoch] = loss
            val_r2[epoch] = r2
            val_rate[epoch] = rate

            if save == 'train':
                if train_r2[epoch] > best_train_r2:
                    best_train_r2 = train_r2[epoch]
                    torch.save(self.model.state_dict(), 'checkpoint/' + self.name + '.pt')
                    print('Save here')
                    file = open('log/{}.txt'.format(self.name), 'a+')
                    file.write('Save here' + '\n')
                    file.close()
            elif save == 'test':
                if val_r2[epoch] > best_val_r2:
                    best_val_r2 = val_r2[epoch]
                    torch.save(self.model.state_dict(), 'checkpoint/' + self.name + '.pt')
                    print('Save here')
                    file = open('log/{}.txt'.format(self.name), 'a+')
                    file.write('Save here' + '\n')
                    file.close()
            else:
                raise NotImplementedError()

            print(
                'Epoch:{:>3d} |Train_Loss:{:.6f} |R2:{:.6f}|Val_Loss:{:.6f} |R2:{:.6f} |Rate:{:.3f} |lr:{:.6f}'.format(
                    start_epoch + epoch + 1 + continued,
                    train_loss[epoch], train_r2[epoch], val_loss[epoch], val_r2[epoch], val_rate[epoch],
                    self.optimizer.state_dict()['param_groups'][0]['lr']))
            log = [start_epoch + epoch + 1 + continued, train_loss[epoch], train_r2[epoch], val_loss[epoch], val_r2[epoch],
                   val_rate[epoch],
                   self.optimizer.state_dict()['param_groups'][0]['lr']]
            self.train_log(log)
            path = 'checkpoint/' + self.name
            early_stopping(val_r2[epoch], self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("Done")
        self.write_log(train_loss, val_loss, train_r2, val_r2)

    def boost(self, threshold):
        for i in tqdm(range(len(self.train_f))):
            f = self.train_f[i]
            name = f.split('/')[-1]
            dataset = MyDataset(file_name=f, batch_size=self.Batch_size,
                                     pred_len=self.out_len, label_len=self.label_len, scaler=self.scaler)
            mse = []
            with torch.no_grad():
                for i in range(len(dataset)):
                    x, y, t = dataset(i)
                    pred = self.model(x, t)
                    mse.append( (torch.abs(pred - y) ).detach().cpu().numpy() < threshold )

            self.boost_index[name] = np.concatenate(mse)

            print(f'Drop {(~self.boost_index[name]).sum()} ({(~self.boost_index[name]).sum() / dataset.n *100:.3f}%) data in {name}')

    def boost_train_one_epoch(self, train_all=True, f=None):
        if not train_all:
            name = f.split('/')[-1]
            # boost_index = self.boost_index[name]
            self.dataset = MyDataset_b(file_name=f, batch_size=self.Batch_size, boost_index=self.boost_index[name],
                                     pred_len=self.out_len, label_len=self.label_len, scaler=self.scaler)
            loss, r2 = self.single_train()
        else:
            loss = np.empty((len(self.train_f),))
            r2 = np.empty((len(self.train_f),))
            conter = 0
            for f in self.train_f:
                name = f.split('/')[-1]
                # boost_index = self.boost_index[name]
                self.dataset = MyDataset_b(file_name=f, batch_size=self.Batch_size, boost_index=self.boost_index[name],
                                         pred_len=self.out_len, label_len=self.label_len, scaler=self.scaler)
                train_loss, tran_r2 = self.single_train()
                loss[conter] = train_loss
                r2[conter] = tran_r2
                conter += 1
                del (self.dataset)
            loss = loss.mean()
            r2 = r2.mean()
        return loss, r2

    def test(self, ic_name):
        dataset = MyDataset(ic_name, self.val_batch, self.out_len, label_len=self.label_len, scaler=self.scaler)
        r2 = np.empty((len(dataset),))
        test_loss = np.empty((len(dataset),))
        pred_list = []
        y_list = []

        self.model.eval()
        for i in range(len(dataset)):
            x, y, t = dataset(i)
            with torch.no_grad():
                pred = self.model(x, t)
                loss = torch.mean((pred - y) ** 2)
                       # F.binary_cross_entropy_with_logits(pred, torch.gt(y, 0).float())
                test_loss[i] = loss.item()
            r2_i = fr2(pred, y).cpu().detach().numpy()
            r2[i] = r2_i
            pred_list.append(pred.detach().cpu().numpy())
            y_list.append(y.detach().cpu().numpy())

        return r2, test_loss, pred_list, y_list

    def test_all(self):
        self.model.eval()
        t_val_loss = np.empty((len(self.test_f),))
        t_val_r2 = np.empty((len(self.test_f), 1))
        t_val_rate = np.empty((len(self.test_f), 1))
        conter = 0
        for f in self.test_f:
            print('predicting on' + f.split('/')[-1].split('.')[0])
            dataset = MyDataset(file_name=f, batch_size=self.val_batch,
                                pred_len=self.out_len, label_len=self.label_len, scaler=self.scaler)

            val_loss = np.empty((len(dataset),))
            val_r2 = np.empty((len(dataset),))
            val_rate = np.empty((len(dataset),))

            with torch.no_grad():
                for i in range(len(dataset)):
                    x, y, t = dataset(i)
                    pred = self.model(x, t)
                    loss = torch.mean((pred - y) ** 2)
                           # F.binary_cross_entropy_with_logits(pred, torch.gt(y, 0).float())
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



class MyDataset():
    def __init__(self, file_name, batch_size, pred_len=3, enc_seq_len=20, label_len=1, scaler=False):
        self.name = file_name
        if not scaler:
            self.__read_data__()
        else:
            self.__read_data_s()
        self.batch_size = batch_size
        self.n = self.y.shape[0]
        self.indexes = np.arange(self.n)
        # self.mask = list(range(15))  # [1, 3, 4, 5, 6, 7, 8, 9]# [0, 2, 10, 11, 12, 13, 14]
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
        codedtime = encode_time_(get_tick_date_time_(f['timestamp'][:]))
        codedtime = codedtime.reshape(len(codedtime), 1)
        onehot_encoder = OneHotEncoder(categories=categories, sparse=False)
        self.codedtime = onehot_encoder.fit_transform(codedtime)
        f.close()

    def __read_data_s(self):
        global is_scaler, scalers
        f = h5py.File(self.name, 'r')
        self.x = np.empty(shape=f['x'][:].shape, dtype=np.float32)
        if is_scaler:
            for i in range(15):
                self.x[:, :, i] = scalers[i].transform(f['x'][:, :, i])
        else:
            for i in range(15):
                scaler = StandardScaler(copy=False)
                self.x[:, :, i] = scaler.fit_transform(f['x'][:, :, i])
                scalers.append(scaler)
            is_scaler = True

        self.y = f['y'][:]
        codedtime = encode_time_(get_tick_date_time_(f['timestamp'][:]))
        codedtime = codedtime.reshape(len(codedtime), 1)
        onehot_encoder = OneHotEncoder(categories=categories, sparse=False)
        self.codedtime = onehot_encoder.fit_transform(codedtime)
        f.close()

    def __call__(self, i):
        batch_index = self.indexes[i * self.batch_size: (i + 1) * self.batch_size]
        # y_len = batch_index.shape[0]
        X = self.x[batch_index]
        Y = self.y[batch_index]
        T = self.codedtime[batch_index]
        X = torch.from_numpy(X).cuda()
        # X = torch.tensor(X).cuda()
        Y = torch.from_numpy(Y).cuda()
        T = torch.from_numpy(T).cuda().float()

        return X, Y, T

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __del__(self):
        del self.x, self.y, self.indexes


class MyDataset_b():
    def __init__(self, file_name, batch_size, boost_index, pred_len=3, enc_seq_len=20, label_len=1, scaler=False):
        self.name = file_name
        if not scaler:
            self.__read_data__(boost_index)
        else:
            self.__read_data_s(boost_index)
        self.batch_size = batch_size
        self.n = self.y.shape[0]
        self.indexes = np.arange(self.n)
        # self.mask = list(range(15))  # [1, 3, 4, 5, 6, 7, 8, 9]# [0, 2, 10, 11, 12, 13, 14]
        # self.enc_seq_len = enc_seq_len
        # self.label_len = label_len
        #         print(self.y.shape)
        #         self.ts = f['timestamp'][:]
        self.index = 0
        # self.shift = 10
        self.device = 'cuda'
        # self.pred_len = pred_len - 1

    def __read_data__(self, boost_index):
        f = h5py.File(self.name, 'r')
        self.x = f['x'][np.where(boost_index)[0]]
        self.y = f['y'][np.where(boost_index)[0]]
        codedtime = encode_time_(get_tick_date_time_(f['timestamp'][:]))
        codedtime = codedtime.reshape(len(codedtime), 1)
        onehot_encoder = OneHotEncoder(categories=categories, sparse=False)
        self.codedtime = onehot_encoder.fit_transform(codedtime)
        f.close()

    def __read_data_s(self, boost_index):
        global is_scaler, scalers
        f = h5py.File(self.name, 'r')
        self.x = np.empty(shape=f['x'][boost_index].shape, dtype=np.float32)
        if is_scaler:
            for i in range(15):
                self.x[:, :, i] = scalers[i].transform(f['x'][np.where(boost_index)[0]][:, i])
        else:
            for i in range(15):
                scaler = StandardScaler(copy=False)
                self.x[:, :, i] = scaler.fit_transform(f['x'][np.where(boost_index)[0]][:, i])
                scalers.append(scaler)
            is_scaler = True

        codedtime = encode_time_(get_tick_date_time_(f['timestamp'][:]))
        codedtime = codedtime.reshape(len(codedtime), 1)
        onehot_encoder = OneHotEncoder(categories=categories, sparse=False)
        self.codedtime = onehot_encoder.fit_transform(codedtime)

        self.y = f['y'][:]
        f.close()

    def __call__(self, i):
        batch_index = self.indexes[i * self.batch_size: (i + 1) * self.batch_size]
        # y_len = batch_index.shape[0]
        X = self.x[batch_index]
        Y = self.y[batch_index]
        T = self.codedtime[batch_index]
        X = torch.from_numpy(X).cuda()
        # X = torch.tensor(X).cuda()
        Y = torch.from_numpy(Y).cuda()
        T = torch.from_numpy(T).cuda().float()

        return X, Y, T

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
    def __init__(self, dim_val, dim_attn, input_size, n_layers, n_heads, dropout):
        super(AttentionRNN, self).__init__()
        self.n_layers = n_layers
        self.dim_val = dim_val

        self.fc1 = nn.Linear(input_size, dim_val)

        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.norm = nn.LayerNorm(dim_val)
        self.rnn = nn.LSTM(input_size=dim_val, hidden_size=dim_val, num_layers=n_layers, bidirectional=False,
                           dropout=dropout, batch_first=True)

        self.fc2 = nn.Linear(dim_val * 2 + 15, dim_val * 2)
        self.fc3 = nn.Linear(dim_val * 2, 1)

        self.fc4 = nn.Linear(15, 15)

    def forward(self, x, t):
        x = self.fc1(x)
        a = self.attn(x)
        x = self.norm(a + x)
        t = self.fc4(t)

        x, (hn, cn) = self.rnn(x)
        hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)

        x = torch.cat((hn, t), dim=1)
        x = self.fc3(F.elu(self.fc2(x)))
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


class AttentionGRU(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, n_layers, n_heads, dropout):
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


class GRU(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, n_layers, n_heads, dropout):
        super(GRU, self).__init__()
        self.n_layers = n_layers
        self.dim_val = dim_val

        self.rnn = nn.GRU(input_size=input_size, hidden_size=dim_val, num_layers=n_layers, bidirectional=False,
                          dropout=dropout, batch_first=True)

        self.fc1 = nn.Linear(dim_val * 2, dim_val * 2)
        self.fc2 = nn.Linear(dim_val * 2, 1)

        # self.fc3 = nn.Linear(15, 15)
        # self.relu = nn.ReLU()
        # self.fc4 = nn.Linear(dim_val * 2, 15)
        # self.fc5 = nn.Linear(15, 1)

        # self.fc3 = nn.Linear(dim_val * 2, dim_val * 2)
        # self.fc4 = nn.Linear(dim_val * 2, 1)

    def forward(self, x, t):
        # t = self.fc3(t)
        # t = self.relu(t)
        x, hn, = self.rnn(x)
        hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        x = self.fc2(F.elu(self.fc1(hn)))
        # x = torch.cat((x, t), dim=1)
        # x = F.elu(self.fc4(x))
        # x = self.fc5(x)
        # xx = self.fc4(F.elu(self.fc3(hn)))
        return x
