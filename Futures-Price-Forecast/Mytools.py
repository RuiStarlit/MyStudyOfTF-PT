# -*- coding:utf-8 -*-
"""
Writer: RuiStarlit
File: Mytools
Project: informer
Create Time: 2021-11-10
"""
import torch
import numpy as np



class EarlyStopping_R2:
    def __init__(self, patience=10, verbose=False, delta=0, val_r2=-np.Inf):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = val_r2
        self.early_stop = False
        self.val_r2_max = val_r2
        self.delta = delta

    def __call__(self, val_r2, model, path):
        score = val_r2
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_r2, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_r2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_r2, model, path):
        if self.verbose:
            print(f'Validation R2 increased ({self.val_r2_max:.6f} --> {val_r2:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'-'+'checkpoint.pth')
        self.val_r2_max = val_r2


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss