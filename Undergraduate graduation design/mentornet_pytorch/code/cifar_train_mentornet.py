from itertools import accumulate
import os
from this import d
import time
import numpy as np
import torch
import torch.nn as nn
from zmq import device
import utils
import argparse
import tqdm
import torchvision


parser = argparse.ArgumentParser(description='[Train Mentornet with Studentnet]')

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--data_dir',type=str, default='')
parser.add_argument('--dataset_name', type=str, default='cifar10')
parser.add_argument('--studentnet',type=str,default='resnet101')
parser.add_argument('--lr',type=float,default=0.1)
parser.add_argument('--lr_decay',type=float,default=0.1)
parser.add_argument('--trained_mentornet',type=str,default='')
parser.add_argument('--loss_p_percentile', type=float,default=0.7)
parser.add_argument('--burn_in_epoch',type=int, default=0)
parser.add_argument('--fixed_epoch_after_burn_in',type=bool,default=False)
parser.add_argument('--loss_moving_average_decay',type=float,default=0.5)


class Train():
    def __init__(self, **kwarg):
        self.batch_size = kwarg['batch_size']
        self.data_dit = kwarg['data_dir']
        self.dataset_name = kwarg['dataset_name']
        self.studentnet = kwarg['studentnet']
        self.lr = kwarg['lr']
        self.lr_decay = kwarg['lr_decay']
        self.trained_mentornet = kwarg['trained_mentornet']
        self.loss_p_percentile = kwarg['loss_p_percentile']
        self.burn_in_epoch = kwarg['burn_in_epoch']
        self.fixed_epoch_after_burn_in = kwarg['fixed_epoch_after_burn_in']
        self.loss_moving_average_decay = kwarg['loss_moving_average_decay']
        self.device = kwarg['device']
        self.example_dropout_rates = kwarg['example_dropout_rates']
        self._build_resnet_model()
    
    def _select_optim(self, opt='sgd'):
        if opt == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters, lr = self.lr)
        else:
            raise NotImplementedError()

    def _set_lr(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _build_resnet_model(self):
        if self.dataset_name == 'cifar10':
            kwarg = {'num_classes':10}
        else:
            kwarg = {'num_classes':100}
        self.model = torchvision.models.resnet101(pretrained=False,
         progress=True, kwargs=kwarg)
    
    def train(self, epochs, dataset, ckp=None):
        """Trains the mentornet with the student resnet model.

            Args:
                epoch:
                ckp: check point
        """
        
        if not ckp is None:
            self.model.load_state_dict(torch.load(ckp))

        for epoch in tqdm(range(epochs)):
            accumulate_loss = 0.0
            dropout_rates = utils.parse_dropout_rate_list(self.example_dropout_rates)
            loss_p_percentile = self.loss_p_percentile
            for batch in dataset:
                data, label = batch
                y = self.model(data.to(self.device))
                y = torch.softmax(y)
                self.optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(y, label)
                mloss = loss.reshpae(-1, 1)
                epoch_step = float(epoch) / float(epochs) * 100
                zero_labels = torch.zeros_like(mloss)

                v = utils.mentornet(
                    epoch_step,
                    loss,
                    zero_labels,
                    loss_p_percentile,
                    dropout_rates,
                    burn_in_epoch=self.burn_in_epoch,
                    fixed_epoch_after_burn_in=self.fixed_epoch_after_burn_in,
                    loss_moving_average_decay=self.loss_moving_average_decay).detach()
                
                data_util = utils.summarize_data_utilization(v, global_step,
                                                   self.batch_size)
                # L2 正则化
                l2_regularization = torch.tensor([0],dtype=torch.float32)
                for param in self.model.parameters():
                    l2_regularization += torch.norm(param, 2)
                weighted_loss = torch.mul(mloss, v)

                weighted_total_loss = weighted_loss + l2_regularization
                weighted_total_loss.backward()
                self.optimizer.step()
                accumulate_loss += weighted_total_loss.item()




                
                # loss.backward()
                # accumulate_loss += loss.item()
                # self.optimizer.step()
            
            



