# -*- coding:utf-8 -*-
"""
Author: RuiStarlit
File: test1
Project: LearningPyTorch
Create Time: 2021-07-07

"""
import torch


class Arg:
    def __init__(self,):
        self.gpu = 0
        self.num_users = 100
        self.dataset = 'cifar10'
        self.epochs = 6
        self.frac = 0.1
        self.num_classes = 10 if self.dataset == 'cifar10' else 100
        self.local_bs = 10
        self.local_ep = 20
        self.lr = 0.01  # learning rate
        self.optimizer = 'sgd'
        self.verbose = 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = 'ResNet18'


args = Arg(local_bs =10)
print(args.local_bs)