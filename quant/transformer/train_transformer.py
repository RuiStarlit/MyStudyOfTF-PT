# -*- coding:utf-8 -*-
"""
Writer: RuiStarlit
File: train_transformer_direct
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
import platform
import h5py
import datetime
from sklearn import model_selection

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# from utils import *
from models import *
from utils1 import *
from Mytools import EarlyStopping_R2

import warnings

warnings.filterwarnings('ignore')


def fr2(x, y):
    return 1 - torch.sum((x - y) ** 2) / (torch.sum((y - torch.mean(y)) ** 2) + 1e-4)


def fr2_n(y_pred, y_true):
    return 1 - np.sum(np.square(y_pred - y_true)) / (np.sum(np.square(y_true - y_true.mean())) + 1e-4)


def frate(x, y):
    return torch.mean(torch.gt(x * y, 0).float())


is_scaler = False
scalers = []

is_print = False


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


class Train_transformer():
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
        self.val_batch = val_batch
        self.scaler = scaler
        self.boost_index = {}
        self.clip_grad = True
        self.decay = decay
        self.opt_schedule = opt_schedule
        self.weight = 1
        self.best_val_r2 = None
        self.noam = False
        self.val_every = False
        self.test_every =False

    def _build_model(self, opt=None):
        model = Transformer(dim_val=self.d_model, dim_attn=self.d_ff, enc_in=self.enc_in, dec_in=self.dec_in,
                            out_seq_len=self.out_len, n_encoder_layers=self.e_layers, n_decoder_layers=self.d_layers,
                            n_heads=self.n_heads, dropout=self.dropout, label_len=self.label_len)

        model.to(self.device)

        if opt == 'xavier':
            model.apply(weight_init)
            print('Using xavier initial')

        if platform.system() == 'Windows':
            self.time = (datetime.datetime.now()).strftime("_%m-%d_%H-%M")
        else:
            self.time = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("_%m-%d_%H-%M")
        self.name = 'transformer-' + str(self.out_len) + 's' + self.time
        self.model = model
        self._selct_criterion()
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
                         base_lr=0.0005, max_lr=0.005, step=1000):
        if opt == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
                                                                        patience=patience, factor=factor, min_lr=min_lr)
        elif opt == 'onecycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.01, epochs=epoch,
                                                                 steps_per_epoch=5)
        elif opt == 'cyclic':
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=base_lr, max_lr=max_lr,
                                                               step_size_up=step, step_size_down=step,
                                                               cycle_momentum=False)
        elif opt == 'noam':
            self.scheduler = NoamOpt(self.d_model, factor, step)
            self.noam = True
        else:
            raise NotImplementedError()

    def preview_noam(self):
        if self.noam is False:
            raise AttributeError('Only support Noam Schedule now')
        noam = self.scheduler
        plt.plot(np.arange(1, 20000), [[noam.rate(i)] for i in range(1, 20000)])
        plt.title('Preview of the lr  Noam')
        plt.show()

    def _set_lr(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print('Learning Rate is set to ' + str(lr))

    def _set_lr_noam(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _selct_criterion(self, criterion=None, beta=1):
        if criterion is None:
            self.criterion = nn.MSELoss()
        elif criterion == 'huber':
            if platform.system() == 'Windows':
                self.criterion = nn.SmoothL1Loss(beta=beta)
            else:
                self.criterion = nn.SmoothL1Loss()
        else:
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

    def save(self, name):
        torch.save(self.model.state_dict(), 'checkpoint/' + name + '.pt')
        print('Successfully save')

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

    def process_one_batch(self, batch_x, batch_y, ct):
        # global is_print
        # 对Dataloader返回的y值进行处理，提取出模型使用的时间序列部分和用于计算误差的对比部分

        # batch_y = batch_y
        # if is_print is False:
        #     print(batch_y.shape)
        # print('batch y', batch_y.shape)
        # decoder input
        dec_inp = batch_y[:, :self.label_len, :self.dec_in]
        dec_inp = torch.from_numpy(dec_inp).float().to(self.device)
        # if is_print is False:
        #     print(dec_inp.shape)
        # encoder - decoder
        outputs = self.model(batch_x, dec_inp)
        batch_y = torch.from_numpy(batch_y[:, -self.out_len:, :]).float().to(self.device)
        # if is_print is False:
        #     print(batch_y.shape)
        #     is_print = True
        # sys.exit('QAQ')

        return outputs, batch_y

    def single_train(self):
        self.model.train()
        train_loss = np.empty((len(self.dataset),))
        train_r2 = np.empty((len(self.dataset),))
        for i in range(len(self.dataset)):
            x, y, ct = self.dataset(i)
            self.optimizer.zero_grad()
            pred, Y = self.process_one_batch(x, y, ct)

            # print('pred',pred[:, :].shape)
            # print('Y', Y[:, :, 0].shape)
            # sys.exit('QAQ')

            loss = self.criterion(pred[:, :], Y[:, :, 0])

            # F.binary_cross_entropy_with_logits(pred, torch.gt(Y, 0).float())

            train_loss[i] = loss.item()
            loss.backward()
            if self.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            if self.noam is True:
                self.scheduler.step(self._set_lr_noam)
            self.optimizer.step()

            # if self.opt_schedule:
            #     if i % self.decay == 0:
            #         self.scheduler.step(loss)

            r2 = fr2(pred[:, -1], Y[:, -1, 0]).cpu().detach().numpy()
            train_r2[i] = r2

        train_loss = train_loss.mean()
        train_r2 = train_r2.mean()
        return train_loss, train_r2

    def train_one_epoch(self, train_all=True, f=None, bost=False, val_size=0.1):
        if not train_all:
            with h5py.File(f, 'r') as file:
                f_len = file['y'].shape[0]
            idx = np.arange(f_len)
            train_idx, valid_idx = model_selection.train_test_split(idx, shuffle=True, test_size=val_size,
                                                                    random_state=2021)
            if bost:
                self.dataset = MyDataset_p(file_name='temp_train/' + f.split('/')[-1], batch_size=self.Batch_size,
                                           label_len=self.label_len, index=train_idx)
            else:
                self.dataset = MyDataset_p(file_name=f, batch_size=self.Batch_size,
                                           label_len=self.label_len, index=train_idx)
            loss, r2 = self.single_train()
            val_loss, val_r2, val_rate = self.val(f, valid_idx)
        else:
            loss = np.empty((len(self.train_f),))
            r2 = np.empty((len(self.train_f),))
            val_loss = np.empty((len(self.train_f),))
            val_r2 = np.empty((len(self.train_f),))
            val_rate = np.empty((len(self.train_f),))
            conter = 0
            for f in self.train_f:
                with h5py.File(f, 'r') as file:
                    f_len = file['y'].shape[0]
                idx = np.arange(f_len)
                train_idx, valid_idx = model_selection.train_test_split(idx, shuffle=True, test_size=val_size,
                                                                        random_state=2021)
                if bost:
                    self.dataset = MyDataset_p(file_name='temp_train/' + f.split('/')[-1], batch_size=self.Batch_size,
                                               label_len=self.label_len, index=train_idx)
                else:
                    self.dataset = MyDataset_p(file_name=f, batch_size=self.Batch_size,
                                               label_len=self.label_len, index=train_idx)
                train_loss, train_r2 = self.single_train()
                loss[conter] = train_loss
                r2[conter] = train_r2
                del (self.dataset)

                val_loss_, val_r2_, val_rate_ = self.val(f, valid_idx)
                val_loss[conter] = val_loss_
                val_rate[conter] = val_rate_
                val_r2[conter] = val_r2_
                conter += 1
                if self.val_every is True:
                    print(
                        f'After training {f.split("/")[-1].split(".")[0]}  Train_loss:{train_loss:.6f} R2:{train_r2:.6f} '
                        f'| Val_Loss:{val_loss_:.6f} '
                        f'|R2:{val_r2_:.6f} |Rate:{val_rate_:.3f} |'
                        f'lr:{self.optimizer.state_dict()["param_groups"][0]["lr"]:.6f}')
                    file = open('log/{}.txt'.format(self.name), 'a+')
                    file.write(
                        f'After training {f.split("/")[-1].split(".")[0]}  Train_loss:{train_loss:.6f} R2:{train_r2:.6f} '
                        f'| Val_Loss:{val_loss_:.6f} '
                        f'|R2:{val_r2_:.6f} |Rate:{val_rate_:.3f} |'
                        f'lr:{self.optimizer.state_dict()["param_groups"][0]["lr"]:.6f}' + '\n')
                    file.close()
                if self.test_every is True:
                    loss_, r2_, rate_, _, _, _ = self.test_all()
                    print(f'Test_Loss:{loss_:.6f} |R2:{r2_:.6f} |Rate:{rate_:.3f} ')
                    file = open('log/{}.txt'.format(self.name), 'a+')
                    file.write(f'Test_Loss:{loss_:.6f} |R2:{r2_:.6f} |Rate:{rate_:.3f} ' + '\n')
                    file.close()

            val_loss = val_loss.mean()
            val_r2 = val_r2.mean()
            val_rate = val_rate.mean()
            loss = loss.mean()
            r2 = r2.mean()
        return loss, r2, val_loss, val_r2, val_rate

    def val(self, f, index):
        self.model.eval()
        dataset = MyDataset_p(file_name=f, batch_size=self.val_batch,
                              label_len=self.label_len, index=index)

        val_loss = np.empty((len(dataset),))
        val_rate = np.empty((len(dataset),))
        pred_list = []
        y_list = []
        with torch.no_grad():
            for i in range(len(dataset)):
                x, y, ct = dataset(i)

                pred, Y = self.process_one_batch(x, y, ct)

                loss = self.criterion(pred[:, :], Y[:, :, 0])
                # 0.4 * torch.mean(torch.square(pred[:, 10:, 0] - Y[:, 10:, 0]))
                # F.binary_cross_entropy_with_logits(pred, torch.gt(Y, 0).float())

                val_loss[i] = loss.item()
                val_rate[i] = frate(pred[:, -1], Y[:, -1, 0]).detach().cpu().numpy()
                pred_list.append(pred[:, -1].detach().cpu().numpy())
                y_list.append(Y[:, -1, 0].detach().cpu().numpy())
            pred = np.concatenate(pred_list)
            y = np.concatenate(y_list)
            val_loss = val_loss.mean()
            val_rate = val_rate.mean()
            val_r2 = fr2_n(pred, y)
        del (dataset)

        return val_loss, val_r2, val_rate

    def warmup_train(self, warm_lr, warmup_step=5, train_all=False, f=None, testfile=None):
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
            loss, r2, v_loss, v_r2, v_rate = self.train_one_epoch(train_all, f)
            train_loss[epoch] = loss
            train_r2[epoch] = r2

            val_loss[epoch] = v_loss
            val_r2[epoch] = v_r2
            val_rate[epoch] = v_rate
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
            self.lr += (1 / (warmup_step - 1)) * delta_lr
            self._set_lr(self.lr)
        self._set_lr(stored_lr)
        print('Warm Up Done')

    def get_len_dataset(self, f=None):
        print(f'Batch size:{self.Batch_size}')
        if f is None:
            for file in self.train_f:
                dataset = MyDataset_p(file_name=file, batch_size=self.Batch_size,
                                      label_len=self.label_len, index=None)
                name = file.split('/')[-1]
                print(f'Set:{name} | len:{len(dataset)}')
                del (dataset)
        else:
            dataset = MyDataset_p(file_name=f, batch_size=self.Batch_size,
                                  label_len=self.label_len, index=None)
            name = f.split('/')[-1]
            print(f'Set:{name} | len:{len(dataset)}')
            del (dataset)

    def train(self, epochs=200, train_all=True, f=None, save='train', continued=0, val_size=0.1, test_round=5,
              patience=20, bost=False):
        if bost:
            print('Training Mode: boost')
        best_train_r2 = float('-inf')
        if self.best_val_r2 is None:
            self.best_val_r2 = float('-inf')
        self.train_log_head()
        early_stopping = EarlyStopping_R2(patience=patience, verbose=True, val_r2=self.best_val_r2)

        train_loss = np.empty((epochs,))
        train_r2 = np.empty((epochs,))
        val_loss = np.empty((epochs,))
        val_r2 = np.empty((epochs,))
        val_rate = np.empty((epochs,))
        start_epoch = self.epoch + continued
        for epoch in tqdm(range(epochs)):
            self.epoch = start_epoch + epoch

            loss, r2, v_loss, v_r2, v_rate = self.train_one_epoch(train_all, f, bost, val_size)
            train_loss[epoch] = loss
            train_r2[epoch] = r2
            # if not self.opt_schedule:
            if self.noam is False:
                self.scheduler.step(loss)

            # if need_val:
            # else:
            #     loss, r2, rate = 0, 0, 0

            val_loss[epoch] = v_loss
            val_r2[epoch] = v_r2
            val_rate[epoch] = v_rate

            if save == 'train':
                if train_r2[epoch] > best_train_r2:
                    best_train_r2 = train_r2[epoch]
                    torch.save(self.model.state_dict(), 'checkpoint/' + self.name + '.pt')
                    print('Save here')
                    file = open('log/{}.txt'.format(self.name), 'a+')
                    file.write('Save here' + '\n')
                    file.close()
            elif save == 'test':
                if val_r2[epoch] > self.best_val_r2:
                    self.best_val_r2 = val_r2[epoch]
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

            if (epoch+1) % test_round == 0:
                loss, r2, rate, _, _, _ = self.test_all()
                print(f'Test_Loss:{loss:.6f} |R2:{r2:.6f} |Rate:{rate:.3f} ')
                file = open('log/{}.txt'.format(self.name), 'a+')
                file.write(f'Test_Loss:{loss:.6f} |R2:{r2:.6f} |Rate:{rate:.3f} ' + '\n')
                file.close()

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
        for i in tqdm(range(len(self.train_f))):
            f = self.train_f[i]
            name = f.split('/')[-1]
            dataset = MyDataset_p(file_name=f, batch_size=self.val_batch,
                                  label_len=self.label_len)
            mse = []
            with torch.no_grad():
                for j in range(len(dataset)):
                    x, y = dataset(j)
                    pred, Y = self.process_one_batch(x, y)

                    mse.append((torch.abs(pred[:, -1, 0] - Y[:, -1, 0])).detach().cpu().numpy() < threshold)

            mse = np.concatenate(mse)
            print('get mse')
            index = np.where(mse)[0]
            print('get index')

            with h5py.File(f, 'r') as old_f:
                x = old_f['x'][:]
                y = old_f['y'][:]
                ts = old_f['timestamp'][:]
            x = x[index]
            y = y[index]
            ts = ts[index]

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
        self.model.eval()
        print('predicting on' + ic_name.split('/')[-1].split('.')[0])
        dataset = MyDataset_p(file_name=ic_name, batch_size=self.val_batch,
                              label_len=self.label_len, index=None)

        test_loss = np.empty((len(dataset),))
        rate = np.empty((len(dataset),))
        pred_list = []
        y_list = []
        with torch.no_grad():
            for i in range(len(dataset)):
                x, y, ct = dataset(i)

                pred, Y = self.process_one_batch(x, y, ct)

                loss = self.criterion(pred[:, :], Y[:, :, 0])
                # 0.4 * torch.mean(torch.square(pred[:, 10:, 0] - Y[:, 10:, 0]))
                # F.binary_cross_entropy_with_logits(pred, torch.gt(Y, 0).float())

                test_loss[i] = loss.item()
                rate[i] = frate(pred[:, -1], Y[:, -1, 0]).detach().cpu().numpy()
                pred_list.append(pred[:, -1].detach().cpu().numpy())
                y_list.append(Y[:, -1, 0].detach().cpu().numpy())
            pred = np.concatenate(pred_list)
            y = np.concatenate(y_list)
            test_loss = test_loss.mean()
            rate = rate.mean()
            r2 = fr2_n(pred, y)
        del (dataset)

        return r2, test_loss, rate, pred, y

    def test_all(self):
        self.model.eval()
        t_val_loss = np.empty((len(self.test_f),))
        t_val_r2 = np.empty((len(self.test_f), 1))
        t_val_rate = np.empty((len(self.test_f), 1))
        conter = 0
        for f in self.test_f:
            dataset = MyDataset_p(file_name=f, batch_size=self.val_batch,
                                  label_len=self.label_len, index=None)
            val_loss = np.empty((len(dataset), ))
            val_rate = np.empty((len(dataset), ))
            pred_list = []
            y_list = []

            with torch.no_grad():
                for i in range(len(dataset)):
                    x, y, ct = dataset(i)
                    if y.shape[0] == 0:
                        # 数据集末尾的batch太小，数据缺失导致无法预测。实际业务情况下不需要对比实际值就没问题
                        # print('batch erro on', f, i)
                        val_loss[i] = val_loss[:i].mean()
                        val_rate[i] = val_rate[:i].mean()
                        break

                    pred, Y = self.process_one_batch(x, y, ct)

                    loss = self.criterion(pred[:, :], Y[:, :, 0])
                    # 0.4 * torch.mean(torch.square(pred[:, 10:, 0] - Y[:, 10:, 0]))
                    # F.binary_cross_entropy_with_logits(pred, torch.gt(Y, 0).float())

                    val_loss[i] = loss.item()
                    val_rate[i] = frate(pred[:, -1], Y[:, -1, 0]).detach().cpu().numpy()
                    pred_list.append(pred[:, -1].detach().cpu().numpy())
                    y_list.append(Y[:, -1, 0].detach().cpu().numpy())

                pred = np.concatenate(pred_list)
                y = np.concatenate(y_list)
                val_r2 = fr2_n(pred, y)
            t_val_loss[conter] = val_loss.mean()
            t_val_r2[conter] = val_r2
            t_val_rate[conter] = val_rate.mean()
            conter += 1
            del (dataset)
        val_loss = t_val_loss.mean()
        val_r2 = t_val_r2.mean()
        val_rate = t_val_rate.mean()
        return val_loss, val_r2, val_rate, t_val_loss, t_val_r2, t_val_rate


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup):
        # self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self, lr_func):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        lr_func(rate)
        self._rate = rate

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


class MyDataset_p():
    def __init__(self, file_name, batch_size, index, enc_seq_len=20, label_len=10):
        self.name = file_name
        self.__read_data__(index)
        self.batch_size = batch_size
        self.n = self.y.shape[0]
        self.indexes = np.arange(self.n)
        self.mask = list(range(15))  # [1, 3, 4, 5, 6, 7, 8, 9]# [0, 2, 10, 11, 12, 13, 14]
        self.enc_seq_len = enc_seq_len
        self.label_len = label_len
        # print(self.y.shape)
        # self.ts = f['timestamp'][:]
        self.index = 0
        self.shift = 9
        self.device = 'cuda'

    def __read_data__(self, index):
        f = h5py.File(self.name, 'r')
        x = f['x'][:]
        y = f['y'][:]
        ct = f['codedtime'][:]
        if index is None:
            self.x = x
            self.y = y
            self.ct = ct
        else:
            self.x = x[index]
            self.y = y[index]
            self.ct = ct[index]
        f.close()

    def __call__(self, i):
        batch_index = self.indexes[i * self.batch_size: (i + 1) * self.batch_size]

        X = self.x[batch_index, -self.enc_seq_len:, :]
        Y = self.y[batch_index, :, :]

        X = torch.from_numpy(X).to(self.device).float()
        # Y = torch.from_numpy(Y)
        return X, Y, self.ct[batch_index]

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __del__(self):
        del self.x, self.y, self.indexes
