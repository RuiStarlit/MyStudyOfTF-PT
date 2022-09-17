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

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# from utils import *
from models import *
from utils1 import *
from models.model import Informer, InformerStack
from Mytools import EarlyStopping_R2

import warnings

warnings.filterwarnings('ignore')


def fr2(x, y):
    return 1 - torch.sum((x - y) ** 2) / (torch.sum((y - torch.mean(y)) ** 2) + 1e-4)


def frate(x, y):
    return torch.mean(torch.gt(x * y, 0).float())


is_scaler = False
scalers = []


class Train_Informer_mul():
    def __init__(self, enc_in, dec_in, c_out, seq_len, out_len, d_model, d_ff, n_heads,
                 e_layers, d_layers, label_len,
                 dropout, batch_size, val_batch, lr, device, train_f, test_f, scaler, decay, opt_schedule):
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
        # self.val_batch = val_batch
        self.scaler = scaler
        self.boost_index = {}
        self.clip_grad = True
        self.decay = decay
        self.opt_schedule = opt_schedule
        self.weight = 0.5

    def _build_model(self):
        model = Informer(enc_in=self.enc_in, dec_in=self.dec_in, c_out=self.c_out, out_len=self.out_len,
                         d_model=self.d_model, d_ff=self.d_ff, n_heads=self.n_heads, e_layers=self.e_layers,
                         d_layers=self.d_layers, label_len=self.label_len, dropout=self.dropout
                         )

        model.to(self.device)
        self.time = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("_%m-%d_%H:%M")
        self.name = 'Informer-' + 'mutilstep-' + str(self.out_len) + 's' + self.time
        self.model = model
        print(self.model)

    def _selct_optim(self, opt='adam'):
        if opt == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif opt == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        elif opt == 'adamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError()

    def _selct_scheduler(self, opt='plateau', patience=8, factor=0.8, min_lr=0.00001, epoch=50,
                         base_lr=0.0005, max_lr=0.005):
        if opt == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
                                                                        patience=patience, factor=factor, min_lr=min_lr)
        elif opt == 'onecycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.01, epochs=epoch,
                                                                 steps_per_epoch=9)
        elif opt == 'cyclic':
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=base_lr, max_lr=max_lr)
        else:
            raise NotImplementedError()

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
        f.write(f'Is scaler :{self.scaler}')
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
        # 对Dataloader返回的y值进行处理，提取出模型使用的时间序列部分和用于计算误差的对比部分

        batch_y = batch_y.float()
        # print('batch y', batch_y.shape)
        # decoder input
        dec_inp = torch.zeros([batch_y.shape[0], 10 + self.out_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:, :self.label_len], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        outputs = self.model(batch_x, dec_inp)
        batch_y = batch_y[:, -self.out_len-10:, :].to(self.device)

        return outputs, batch_y

    def single_train(self):
        self.model.train()
        train_loss = np.empty((len(self.dataset),))
        train_r2 = np.empty((len(self.dataset),))
        for i in range(len(self.dataset)):
            x, y = self.dataset(i)
            self.optimizer.zero_grad()
            pred, Y = self.process_one_batch(x, y)

            # print('pred',pred.shape)
            # print('Y',Y.shape)

            loss = self.weight * torch.mean((pred[:, :, 0] - pred[:, :, 0]) ** 2) + \
                   (1 - self.weight) * torch.mean((pred[:, -self.out_len:, 1] - Y[:, -self.out_len:, 1]))
            # F.binary_cross_entropy_with_logits(pred, torch.gt(Y, 0).float())

            train_loss[i] = loss.item()
            loss.backward()
            if self.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=50, norm_type=2)
            self.optimizer.step()

            if self.opt_schedule:
                if i % self.decay == 0:
                    self.scheduler.step(loss)

            r2 = fr2(pred[:, 0], Y[:, 0]).cpu().detach().numpy()
            train_r2[i] = r2

        train_loss = train_loss.mean()
        train_r2 = train_r2.mean()
        return train_loss, train_r2

    def train_one_epoch(self, train_all=True, f=None, bost=False):
        if not train_all:
            if bost:
                self.dataset = MyDataset(file_name='temp_train/' + f.split('/')[-1], batch_size=self.Batch_size,
                                         pred_len=self.out_len, label_len=self.label_len, scaler=self.scaler)
            else:
                self.dataset = MyDataset_p(file_name=f, batch_size=self.Batch_size,
                                         pred_len=self.out_len, label_len=self.label_len)
            loss, r2 = self.single_train()
        else:
            loss = np.empty((len(self.train_f),))
            r2 = np.empty((len(self.train_f),))
            conter = 0
            for f in self.train_f:
                if bost:
                    self.dataset = MyDataset(file_name='temp_train/' + f.split('/')[-1], batch_size=self.Batch_size,
                                             pred_len=self.out_len, label_len=self.label_len, scaler=self.scaler)
                else:
                    self.dataset = MyDataset_p(file_name=f, batch_size=self.Batch_size,
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
            print('predicting on' + f.split('/')[-1].split('.')[0])
            dataset = ValDataset_p(file_name=f, pred_len=self.out_len, label_len=self.label_len, device=self.device)
            with torch.no_grad():
                x, y = dataset()
                pred, Y = self.process_one_batch(x, y)
                pred = pred.squeeze(2)
                loss = torch.mean((pred - Y) ** 2) + \
                       F.binary_cross_entropy_with_logits(pred, torch.gt(Y, 0).float())
                val_loss = loss.item()
                val_r2 = fr2(pred[:, 0], Y[:, 0]).cpu().detach().numpy()
                val_rate = frate(pred[:, 0], Y[:, 0]).detach().cpu().numpy()
            del (dataset)

        else:
            t_val_loss = np.empty((len(self.test_f),))
            t_val_r2 = np.empty((len(self.test_f), 1))
            t_val_rate = np.empty((len(self.test_f), 1))
            conter = 0
            for f in self.test_f:
                dataset = ValDataset_p(file_name=f, pred_len=self.out_len, label_len=self.label_len, device=self.device)

                with torch.no_grad():
                    x, y = dataset()
                    pred, Y = self.process_one_batch(x, y)
                    pred = pred.squeeze(2)
                    loss = torch.mean((pred - Y) ** 2)
                    # F.binary_cross_entropy_with_logits(pred, torch.gt(Y, 0).float())
                    val_loss = loss.item()
                    val_r2 = fr2(pred[:, 0], Y[:, 0]).cpu().detach().numpy()
                    val_rate = frate(pred[:, 0], Y[:, 0]).detach().cpu().numpy()
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

    def warmup_train(self, warm_lr, warmup_step=5, train_all=False, f=None, val_all=True, testfile=None):
        # warm_lr 0.00001 -> 0.00005

        stored_lr = self.lr
        delta_lr = stored_lr - warm_lr
        self._set_lr(warm_lr)
        train_loss = np.empty((warmup_step,))
        train_r2 = np.empty((warmup_step,))
        val_loss = np.empty((warmup_step,))
        val_r2 = np.empty((warmup_step,))
        val_rate = np.empty((warmup_step,))
        print('Warm')
        for epoch in tqdm(range(warmup_step)):
            self.epoch = epoch
            loss, r2 = self.train_one_epoch(train_all, f)
            train_loss[epoch] = loss
            train_r2[epoch] = r2

            loss, r2, rate = self.val(val_all, testfile)

            val_loss[epoch] = loss
            val_r2[epoch] = r2
            val_rate[epoch] = rate
            # torch.save(self.model.state_dict(), 'checkpoint/' + self.name + '.pt')
            print(
                'Epoch:{:>3d} |Train_Loss:{:.6f} |R2:{:.6f}|Val_Loss:{:.6f} |R2:{:.6f} |Rate:{:.3f} |lr:{:.6f}'.format(
                    epoch + 1,
                    train_loss[epoch], train_r2[epoch], val_loss[epoch], val_r2[epoch], val_rate[epoch],
                    self.optimizer.state_dict()['param_groups'][0]['lr']))
            log = [epoch + 1, train_loss[epoch], train_r2[epoch], val_loss[epoch], val_r2[epoch],
                   val_rate[epoch],
                   self.optimizer.state_dict()['param_groups'][0]['lr']]
            self.train_log(log)
            self.lr += (1 / warmup_step) * delta_lr
            self._set_lr(self.lr)
        self._set_lr(stored_lr)
        print('Warm Up Done')

    def get_len_dataset(self, f=None):
        print(f'Batch size:{self.Batch_size}')
        if f is None:
            for file in self.train_f:
                dataset = MyDataset(file_name=file, batch_size=self.Batch_size,
                                    pred_len=self.out_len, label_len=self.label_len, scaler=self.scaler)
                name = file.split('/')[-1]
                print(f'Set:{name} | len:{len(dataset)}')
                del (dataset)
        else:
            dataset = MyDataset(file_name=f, batch_size=self.Batch_size,
                                pred_len=self.out_len, label_len=self.label_len, scaler=self.scaler)
            name = f.split('/')[-1]
            print(f'Set:{name} | len:{len(dataset)}')
            del (dataset)

    def train(self, epochs=200, train_all=True, f=None, val_all=False, testfile=None, save='train', continued=0,
              patience=20, bost=False):
        if bost:
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
            # if bost:
            #     # print('1')
            #     loss, r2 = self.boost_train_one_epoch(train_all, f)
            # else:
            # print('56')
            loss, r2 = self.train_one_epoch(train_all, f, bost)
            train_loss[epoch] = loss
            train_r2[epoch] = r2
            if not self.opt_schedule:
                self.scheduler.step(loss)

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
                    start_epoch + epoch + 1,
                    train_loss[epoch], train_r2[epoch], val_loss[epoch], val_r2[epoch], val_rate[epoch],
                    self.optimizer.state_dict()['param_groups'][0]['lr']))
            log = [start_epoch + epoch + 1, train_loss[epoch], train_r2[epoch], val_loss[epoch], val_r2[epoch],
                   val_rate[epoch],
                   self.optimizer.state_dict()['param_groups'][0]['lr']]
            self.train_log(log)
            self.lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            path = 'checkpoint/' + self.name
            early_stopping(val_r2[epoch], self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                file = open('log/{}.txt'.format(self.name), 'a+')
                file.write("Early stopping" + '\n')
                file.close()
                break

        print("Done")
        self.write_log(train_loss, val_loss, train_r2, val_r2)

    def boost(self, threshold, path):
        # 选择误差小于threshold的数据，写成h5py文件以便后续读取
        raise NotImplementedError()
        for i in tqdm(range(len(self.train_f))):
            f = self.train_f[i]
            name = f.split('/')[-1]
            dataset = MyDataset(file_name=f, batch_size=self.Batch_size,
                                pred_len=self.out_len, label_len=self.label_len, scaler=self.scaler)
            mse = []
            with torch.no_grad():
                for i in range(len(dataset)):
                    x, y = dataset(i)
                    pred, Y = self.process_one_batch(x, y)
                    pred = pred.squeeze(2)
                    mse.append((torch.abs(pred[:, 0] - Y[:, 0])).detach().cpu().numpy() < threshold)

            mse = np.concatenate(mse)

            with h5py.File(f, 'r') as old_f:
                x = old_f['x'][np.where(mse)[0]]
                y = old_f['y'][np.where(mse)[0]]
                ts = old_f['timestamp'][np.where(mse)[0]]

            with h5py.File('temp_train/' + name, 'w') as new_f:
                new_f.create_dataset('x', data=x)
                new_f.create_dataset('y', data=y)
                new_f.create_dataset('timestamp', data=ts)

            print(
                f'Drop {(~mse).sum()} ({(~mse).sum() / dataset.n * 100:.3f}%) data in {name}')
            file = open('log/{}.txt'.format(self.name), 'a+')
            file.write(
                f'Drop {(~mse).sum()} ({(~mse).sum() / dataset.n * 100:.3f}%) data in {name}')
            file.close()

    def test(self, ic_name):
        dataset = ValDataset_p(file_name=ic_name, pred_len=self.out_len, label_len=self.label_len, device=self.device)

        self.model.eval()
        x, y = dataset()
        with torch.no_grad():
            pred, Y = self.process_one_batch(x, y)
            pred = pred.squeeze(2)
            loss = torch.mean((pred - Y) ** 2)
            # F.binary_cross_entropy_with_logits(pred, torch.gt(Y, 0).float())
            test_loss = loss.item()
        pred = pred.squeeze(2)
        r2 = fr2(pred[:, 0], Y[:, 0]).cpu().detach().numpy()
        pred = pred[:, 0].detach().cpu().numpy()
        y = Y[:, 0].detach().cpu().numpy()

        return r2, test_loss, pred, y

    def test_all(self):
        self.model.eval()
        t_val_loss = np.empty((len(self.test_f),))
        t_val_r2 = np.empty((len(self.test_f), 1))
        t_val_rate = np.empty((len(self.test_f), 1))
        conter = 0
        for f in self.test_f:
            print('predicting on' + f.split('/')[-1].split('.')[0])
            dataset = ValDataset_p(file_name=f, pred_len=self.out_len, label_len=self.label_len, device=self.device)

            with torch.no_grad():
                x, y = dataset()
                pred, Y = self.process_one_batch(x, y)
                pred = pred.squeeze(2)
                loss = torch.mean((pred - Y) ** 2)
                # F.binary_cross_entropy_with_logits(pred, torch.gt(Y, 0).float())
                val_loss = loss.item()
                r2 = fr2(pred[:, 0], Y[:, 0]).cpu().detach().numpy()
                rate = frate(pred[:, 0], Y[:, 0]).detach().cpu().numpy()
                val_r2 = r2
                val_rate = rate
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
        f.close()

    def __call__(self, i):
        batch_index = self.indexes[i * self.batch_size: (i + 1) * self.batch_size]

        # 向过去取历史时间序列
        if i == 0:
            batch_index = self.indexes[0 + self.shift + self.label_len: self.batch_size]
            # temp = np.zeros((self.label_len + self.shift, 1))
            Y1 = self.y[batch_index]
            y_len = batch_index.shape[0]
            # temp = np.concatenate((temp, Y1))
            temp = self.y[0: self.batch_size]
            for j in range(self.label_len + self.shift):
                Y1 = np.hstack((temp[-1 - j - y_len: -1 - j], Y1))
        else:
            if i >= int(self.n / self.batch_size):
                batch_index = batch_index[:-self.pred_len]
            y_len = batch_index.shape[0]
            r_index = self.indexes[
                      i * self.batch_size - self.label_len - self.shift: (i + 1) * self.batch_size]
            temp = self.y[r_index]
            Y1 = self.y[batch_index]
            for j in range(self.label_len + self.shift):
                Y1 = np.hstack((temp[-1 - j - y_len: -1-j], Y1))

        # 向未来取趋势时间序列
        if i == 0:
            batch_index = self.indexes[0 + self.shift + self.label_len: self.batch_size]
            y_len = batch_index.shape[0]
        else:
            batch_index = self.indexes[i * self.batch_size: (i + 1) * self.batch_size]
            y_len = batch_index.shape[0]

        if i >= int(self.n / self.batch_size):
            # temp = np.full((self.pred_len, 1), self.y[-1, 0])
            temp = self.y[batch_index]
            batch_index = batch_index[:-self.pred_len]
            Y2 = self.y[batch_index]
            # temp = np.concatenate((Y2, temp))
            for j in range(self.pred_len):
                Y2 = np.hstack((Y2, temp[1 + j: j + y_len - 1]))

        else:
            r_index = self.indexes[i * self.batch_size: (i + 1) * self.batch_size + self.pred_len]
            temp = self.y[r_index]
            Y2 = self.y[batch_index]
            for j in range(self.pred_len):
                Y2 = np.hstack((Y2, temp[1 + j: 1 + j + y_len]))
        Y = np.hstack((Y1[:, :-1], Y2))  # size Batch, label_len+shift+1+pred_len-1, 1


        # 计算价格

        X = self.x[batch_index, -self.enc_seq_len:, :]
        Y = Y[:, :, np.newaxis]

        X = torch.from_numpy(X).to(self.device).float()
        Y = torch.from_numpy(Y)
        return X, Y

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __del__(self):
        del self.x, self.y, self.indexes


class MyDataset_p():
    def __init__(self, file_name, batch_size, pred_len=3, enc_seq_len=20, label_len=1, initpoint=1000):
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
        self.initpoint = initpoint

    def __read_data__(self):
        f = h5py.File(self.name, 'r')
        self.x = f['x'][:]
        self.y = f['y'][:]
        f.close()

    def __call__(self, i):
        batch_index = self.indexes[i * self.batch_size: (i + 1) * self.batch_size]

        # 向过去取历史时间序列
        if i == 0:
            batch_index = self.indexes[0 + self.shift + self.label_len: self.batch_size]
            # temp = np.zeros((self.label_len + self.shift, 1))
            Y1 = self.y[batch_index]
            y_len = batch_index.shape[0]
            # temp = np.concatenate((temp, Y1))
            temp = self.y[0: self.batch_size]
            for j in range(self.label_len + self.shift):
                Y1 = np.hstack((temp[-1 - j - y_len: -1 - j], Y1))
        else:
            if i >= int(self.n / self.batch_size):
                batch_index = batch_index[:-self.pred_len]
            y_len = batch_index.shape[0]
            r_index = self.indexes[
                      i * self.batch_size - self.label_len - self.shift: (i + 1) * self.batch_size]
            temp = self.y[r_index]
            Y1 = self.y[batch_index]
            for j in range(self.label_len + self.shift):
                Y1 = np.hstack((temp[-1 - j - y_len: -1-j], Y1))

        # 向未来取趋势时间序列
        if i == 0:
            batch_index = self.indexes[0 + self.shift + self.label_len: self.batch_size]
            y_len = batch_index.shape[0]
        else:
            batch_index = self.indexes[i * self.batch_size: (i + 1) * self.batch_size]
            y_len = batch_index.shape[0]

        if i >= int(self.n / self.batch_size):
            # temp = np.full((self.pred_len, 1), self.y[-1, 0])
            temp = self.y[batch_index]
            batch_index = batch_index[:-self.pred_len]
            Y2 = self.y[batch_index]
            # temp = np.concatenate((Y2, temp))
            for j in range(self.pred_len):
                Y2 = np.hstack((Y2, temp[1 + j: j + y_len - 1]))

        else:
            r_index = self.indexes[i * self.batch_size: (i + 1) * self.batch_size + self.pred_len]
            temp = self.y[r_index]
            Y2 = self.y[batch_index]
            for j in range(self.pred_len):
                Y2 = np.hstack((Y2, temp[1 + j: 1 + j + y_len]))
        Y = np.hstack((Y1[:, :-1], Y2))  # size Batch, label_len+shift+1+pred_len-1, 1
        pY = np.empty((Y.shape), dtype=np.float32)
        pY[:, 0] = self.initpoint
        for i in range(1, self.label_len + self.shift + 1 + self.pred_len):
            pY[:, i] = (Y[:, i] / 100 + 1) * pY[:, i - 1]

        # 计算价格

        X = self.x[batch_index, -self.enc_seq_len:, :]
        Y = Y[:, :, np.newaxis]
        pY = pY[:, :, np.newaxis]
        Y = np.concatenate((Y, pY), axis=2)

        X = torch.from_numpy(X).to(self.device).float()
        Y = torch.from_numpy(Y)
        return X, Y

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __del__(self):
        del self.x, self.y, self.indexes


class ValDataset():
    def __init__(self, file_name, pred_len=3, enc_seq_len=20, label_len=10, scaler=False, device='cuda'):
        self.name = file_name
        if not scaler:
            self.__read_data__()
        else:
            self.__read_data_s()
        self.n = self.y.shape[0]
        self.enc_seq_len = enc_seq_len
        self.label_len = label_len
        self.shift = 10
        self.device = device
        self.pred_len = pred_len - 1

    def __read_data__(self):
        f = h5py.File(self.name, 'r')
        self.x = f['x'][:]
        self.y = f['y'][:]
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
        f.close()

    def __call__(self):

        # 向过去取历史时间序列
        # temp = np.zeros((self.label_len + self.shift, 1))
        Y1 = self.y[0 + self.shift + self.label_len: -self.pred_len]
        y_len = Y1.shape[0]
        temp = self.y[0:-self.pred_len]
        # temp = np.concatenate((temp, Y1))
        for i in range(self.label_len + self.shift):
            Y1 = np.hstack((temp[-1 - i - y_len: -1 - i], Y1))

        # 向未来取趋势时间序列
        # temp = np.full((self.pred_len, 1), self.y[-1, 0])
        Y2 = self.y[0 + self.shift + self.label_len: -self.pred_len]
        temp = self.y[0 + self.shift + self.label_len:]
        # temp = np.concatenate((Y2, temp))
        for i in range(self.pred_len):
            Y2 = np.hstack((Y2, temp[1 + i: 1 + i + y_len]))

        Y = np.hstack((Y1[:, :-1], Y2))

        X = self.x[:, -self.enc_seq_len:, :]
        Y = Y[:, :, np.newaxis]

        X = torch.from_numpy(X).to(self.device).float()
        Y = torch.from_numpy(Y)
        return X, Y

    def __len__(self):
        return self.n

    def __del__(self):
        del self.x, self.y


class ValDataset_p():
    def __init__(self, file_name, pred_len=3, enc_seq_len=20, label_len=10, device='cuda', initpoint=1000):
        self.name = file_name
        self.__read_data__()
        self.n = self.y.shape[0]
        self.enc_seq_len = enc_seq_len
        self.label_len = label_len
        self.shift = 10
        self.device = device
        self.pred_len = pred_len - 1
        self.initpoint = initpoint

    def __read_data__(self):
        f = h5py.File(self.name, 'r')
        self.x = f['x'][:]
        self.y = f['y'][:]
        f.close()

    def __call__(self):

        # 向过去取历史时间序列
        # temp = np.zeros((self.label_len + self.shift, 1))
        Y1 = self.y[0 + self.shift + self.label_len: -self.pred_len]
        y_len = Y1.shape[0]
        temp = self.y[0:-self.pred_len]
        # temp = np.concatenate((temp, Y1))
        for i in range(self.label_len + self.shift):
            Y1 = np.hstack((temp[-1 - i - y_len: -1 - i], Y1))

        # 向未来取趋势时间序列
        # temp = np.full((self.pred_len, 1), self.y[-1, 0])
        Y2 = self.y[0 + self.shift + self.label_len: -self.pred_len]
        temp = self.y[0 + self.shift + self.label_len:]
        # temp = np.concatenate((Y2, temp))
        for i in range(self.pred_len):
            Y2 = np.hstack((Y2, temp[1 + i: 1 + i + y_len]))

        Y = np.hstack((Y1[:, :-1], Y2))
        pY = np.empty((Y.shape), dtype=np.float32)
        pY[:, 0] = self.initpoint
        for i in range(1, self.label_len + self.shift + 1 + self.pred_len):
            pY[:, i] = (Y[:, i] / 100 + 1) * pY[:, i - 1]

        X = self.x[:, -self.enc_seq_len:, :]
        Y = Y[:, :, np.newaxis]
        pY = pY[:, :, np.newaxis]
        Y = np.concatenate((Y, pY), axis=2)

        X = torch.from_numpy(X).to(self.device).float()
        Y = torch.from_numpy(Y)
        return X, Y

    def __len__(self):
        return self.n

    def __del__(self):
        del self.x, self.y


class Transformer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, dec_seq_len,
                 out_seq_len=1, n_encoder_layers=1, n_decoder_layers=1,
                 n_heads=1, dropout=0.1):
        super(Transformer, self).__init__()
        self.dec_seq_len = dec_seq_len

        # Initiate encoder and Decoder layers
        self.encs = nn.ModuleList()
        for i in range(n_encoder_layers):
            self.encs.append(EncoderLayer(dim_val, dim_attn, n_heads, dropout))

        self.decs = nn.ModuleList()
        for i in range(n_decoder_layers):
            self.decs.append(DecoderLayer(dim_val, dim_attn, n_heads, dropout))

        self.pos = PositionalEncoding(dim_val)

        # Dense layers for managing network inputs and outputs
        self.enc_input_fc = nn.Linear(input_size, dim_val)
        self.dec_input_fc = nn.Linear(input_size, dim_val)
        self.out_fc = nn.Linear(dec_seq_len * dim_val, out_seq_len)

    def forward(self, x, y):
        # encoder
        e = self.encs[0](self.pos(self.enc_input_fc(x)))
        #         print('e1',e.shape)
        for enc in self.encs[1:]:
            e = enc(e)

        #         print('e2',e.shape)
        # decoder

        d = self.decs[0](y, e)
        #         print('d1',d.shape)
        for dec in self.decs[1:]:
            d = dec(d, e)

        # output
        x = self.out_fc(d.flatten(start_dim=1))

        return x


class EncoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)

        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        a = self.attn(x)
        x = self.norm1(x + a)

        a = self.fc1(F.elu(self.fc2(x)))
        a = self.dropout(a)
        x = self.norm2(x + a)

        return x


class DecoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.attn2 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)

        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
        self.norm3 = nn.LayerNorm(dim_val)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc):
        a = self.attn1(x)
        x = self.norm1(a + x)

        a = self.attn2(x, kv=enc)
        x = self.norm2(a + x)

        a = self.fc1(F.elu(self.fc2(x)))
        a = self.dropout(a)

        x = self.norm3(x + a)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x
