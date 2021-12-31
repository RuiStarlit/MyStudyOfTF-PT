# -*- coding:utf-8 -*-
"""
Writer: RuiStarlit
File: main
Project: HW
Create Time: 2021-12-27
"""

import datetime
import matplotlib.pyplot as plt
import datetime

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from WeatherPredict.models import *
from WeatherPredict.Mytools import EarlyStopping
import sys
import warnings

warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
# device = 'cpu'

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))


class RmspeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        return torch.sqrt(torch.mean(torch.pow(((y_true - y_pred) / y_pred), 2)))


class Train():
    def __init__(self, enc_in, dec_in, c_out, seq_len, out_len, d_model, d_ff, n_heads,
                 e_layers, d_layers, label_len,
                 dropout, batch_size, val_batch, lr, mdevice, train_f, train_y):
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
        self.device = mdevice
        self.train_f = train_f
        self.train_y = train_y
        self.epoch = 0
        self.val_batch = val_batch
        self.clip_grad = True
        self.noam = False

    def _build_model(self, Model='transformer', opt=None, p=True):
        self.Model = Model
        if Model == 'transformer':
            model = Transformer(dim_val=self.d_model, dim_attn=self.d_ff, enc_in=self.enc_in, dec_in=self.dec_in,
                                out_seq_len=self.out_len, n_encoder_layers=self.e_layers, n_decoder_layers=self.d_layers,
                                n_heads=self.n_heads, dropout=self.dropout, label_len=self.label_len)
        elif Model == 'LSTM':
            model = LSTM(dim_val=self.d_model, dim_attn=self.d_ff, input_size=self.enc_in, n_layers=self.d_layers,
                         n_heads=self.n_heads, dropout=self.dropout)
        elif Model == 'AttnLSTM':
            model = AttentionLSTM(dim_val=self.d_model, dim_attn=self.d_ff, input_size=self.enc_in,
                                  n_layers=self.d_layers, n_heads=self.n_heads, dropout=self.dropout)
        else:
            raise NotImplementedError()

        model.to(self.device)

        if opt == 'xavier':
            model.apply(weight_init)
            print('Using xavier initial parameters')

        self.time = (datetime.datetime.now()).strftime("_%m-%d_%H-%M")
        self.name = self.Model+'-'+self.time
        self.model = model
        self.criterion = nn.MSELoss()
        if p :
            print(self.model)

    def _selct_optim(self, opt='adam'):
        if opt == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif opt == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        elif opt == 'adamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        elif opt == 'sgdm':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=0.01, momentum=0.9)
        else:
            raise NotImplementedError()

    def _selct_scheduler(self, opt='plateau', patience=8, factor=0.8, min_lr=0.00001, epoch=50,
                         base_lr=0.0005, max_lr=0.005, step=1000):
        if opt == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
                                                                        patience=patience, factor=factor, min_lr=min_lr)
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
            self.criterion = nn.SmoothL1Loss(beta=beta)
        elif criterion == 'rmspe':
            self.criterion = RmspeLoss().to(self.device)
        else:
            self.criterion = criterion

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        print("Load " + path + " Successfully")

    def train_log(self, log):
        f = open('log/{}.txt'.format(self.name), 'a+')
        epoch, avg_loss, val_loss, lr = log
        f.write('Epoch:{:>3d} |Train_Loss:{:.6f} |Val_Loss:{:.6f} |lr:{:.6f}\n'.format(
            epoch, avg_loss, val_loss, lr))
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
        f.close()

    def write_remarks(self, s):
        f = open('log/{}.txt'.format(self.name), 'a+')
        f.write(s + '\n')
        f.close()

    def TrainOneEpoch(self, train_idx):
        dataset = MyDataset(self.train_f, self.train_y, self.Batch_size, index=train_idx)
        self.model.train()
        train_loss = np.empty((len(dataset),))

        for i in range(len(dataset)):
            x, y = dataset(i)
            self.optimizer.zero_grad()
            if self.Model == 'transformer':
                dec, enc = x[:, :1, :], x[:, 1:, :]
                # print(dec.shape)
                # print(enc.shape)
                pred = self.model(enc, dec)
            # elif self.Model == 'LSTM':
            else:
                pred = self.model(x)
            # print('pred', pred.shape)
            # print('Y', y.shape)
            # sys.exit('QAQ')
            loss = self.criterion(pred, y)
            train_loss[i] = loss.item()
            loss.backward()
            if self.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            if self.noam is True:
                self.scheduler.step(self._set_lr_noam)
            self.optimizer.step()

        train_loss = train_loss.mean()
        return train_loss

    def val(self, val_idx):
        dataset = MyDataset(self.train_f, self.train_y, self.val_batch, index=val_idx)
        self.model.eval()
        val_loss = np.empty((len(dataset),))

        with torch.no_grad():
            for i in range(len(dataset)):
                x, y = dataset(i)
                self.optimizer.zero_grad()
                if self.Model == 'transformer':
                    dec, enc = x[:, :1, :], x[:, 1:, :]
                    pred = self.model(enc, dec)
                # elif self.Model == 'LSTM':
                else:
                    pred = self.model(x)
                loss = self.criterion(pred, y)
                val_loss[i] = loss.item()

        val_loss = val_loss.mean()
        return val_loss

    def test(self, test_idx):
        dataset = MyDataset(self.train_f, self.train_y, self.Batch_size, index=test_idx)
        self.model.eval()
        test_loss = np.empty((len(dataset),))
        pred_list = []
        y_list = []

        with torch.no_grad():
            for i in range(len(dataset)):
                x, y = dataset(i)
                self.optimizer.zero_grad()
                if self.Model == 'transformer':
                    dec, enc = x[:, :1, :], x[:, 1:, :]
                    pred = self.model(enc, dec)
                # elif self.Model == 'LSTM':
                else:
                    pred = self.model(x)
                loss = self.criterion(pred, y)
                test_loss[i] = loss.item()
                pred_list.append(pred.detach().cpu().numpy())
                y_list.append(y.detach().cpu().numpy())

        test_loss = test_loss.mean()
        pred = np.concatenate(pred_list)
        y = np.concatenate(y_list)
        return test_loss, pred, y

    def reset_dataset(self, train, y):
        self.train_f = train
        self.train_y = y
        print("Reset Dataset")

    def train(self, epochs, train_idx, val_idx, patience=10):
        self.train_log_head()
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        train_loss = np.zeros((epochs,))
        val_loss = np.zeros((epochs,))
        start_epoch = self.epoch
        best_loss = float('+inf')

        for epoch in tqdm(range(epochs)):
            self.epoch = start_epoch + epoch
            loss = self.TrainOneEpoch(train_idx=train_idx)
            train_loss[epoch] = loss
            if self.noam is False:
                self.scheduler.step(loss)
            loss = self.val(val_idx)
            val_loss[epoch] = loss

            if train_loss[epoch] < best_loss:
                best_loss = train_loss[epoch]
                torch.save(self.model.state_dict(), 'checkpoint/' + self.name + '.pt')
                print('Save here')
                file = open('log/{}.txt'.format(self.name), 'a+')
                file.write('Save here' + '\n')
                file.close()

            print('Epoch:{:>3d} |Train_Loss:{:.6f} |Val_Loss:{:.6f} |lr:{:.6f}'.format(
                    start_epoch + epoch + 1,
                    train_loss[epoch], val_loss[epoch],
                    self.optimizer.state_dict()['param_groups'][0]['lr']))
            log = [start_epoch + epoch + 1, train_loss[epoch], val_loss[epoch],
                   self.optimizer.state_dict()['param_groups'][0]['lr']]
            self.train_log(log)
            self.lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            path = 'checkpoint/' + self.name
            early_stopping(val_loss[epoch], self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                file = open('log/{}.txt'.format(self.name), 'a+')
                file.write("Early stopping" + '\n')
                file.close()
                break
        print("\nDone")
        return train_loss, val_loss

    def PlotLoss(self, train_loss, valid_loss, name=None):
        plt.figure()
        plt.plot(train_loss, label='Train Loss')
        plt.plot(valid_loss, label='Val Loss')
        plt.legend()
        if name is None:
            name = self.name
        plt.title(name)
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.show()

        min_train_loss = np.argmin(train_loss)
        min_val_loss = np.argmin(valid_loss)
        print(f'Min Train Loss:{train_loss[min_train_loss]:.6f} at {min_train_loss}|'
              f'Min Valid Loss:{valid_loss[min_val_loss]:.6f} at {min_val_loss}')

    def show_plot(self, index, delta, title):
        labels = ["History", "True Future", "Model Prediction"]
        marker = [".-", "rx", "go"]
        x = self.train_f[index]
        x = x[np.newaxis, :, :]
        # print(x.shape)
        y = self.train_y[index]
        self.model.eval()
        if self.Model == 'transformer':
            dec, enc = torch.from_numpy(x[:, :1, :]).float(), torch.from_numpy(x[:, 1:, :]).float()
            pred = self.model(enc, dec)
        # elif self.Model == 'LSTM':
        else:
            pred = self.model(torch.from_numpy(x).float())
        pred = pred.detach().cpu().numpy()[0]
        time_steps = list(range(-(x.shape[2]), 0))
        if delta:
            future = delta
        else:
            future = 0

        plt.figure()
        plt.title(title)
        plt.plot(time_steps, x[0, 0, :], marker[0], markersize=10, label=labels[0])
        plt.plot(future, y, marker[1], label=labels[1])
        plt.plot(future, pred, marker[2], label=labels[2])
        plt.legend()
        plt.xlim([time_steps[0], (future + 5) * 2])
        plt.xlabel("Time-Step")
        plt.show()
        return


class NoamOpt:
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup):
        # self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self, lr_func):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        lr_func(rate)
        self._rate = rate

    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


class MyDataset():
    """内存足够的情况下将数据读入内存比用DataLoader要快"""
    def __init__(self, data, y, batch_size, index, enc_seq_len=8, label_len=8):
        self.data = data
        self.y = y
        if index is not None:
            self.data = data[index]
            self.y = y[index]
        self.batch_size = batch_size
        self.n = self.y.shape[0]
        self.indexes = np.arange(self.n)
        self.enc_seq_len = enc_seq_len
        self.label_len = label_len
        self.index = 0
        self.device = device

    def __call__(self, i):
        batch_index = self.indexes[i * self.batch_size: (i + 1) * self.batch_size]
        x = self.data[batch_index, -self.enc_seq_len:, :]
        y = self.y[batch_index]
        x = torch.from_numpy(x).to(self.device).float()
        y = torch.from_numpy(y).to(self.device).float()
        return x, y

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __del__(self):
        del self.data, self.y, self.indexes
