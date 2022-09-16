"""Trains MentorNet. After trained, we update the MentorNet in the Algorithm.

Run the following command before running the python code.
export PYTHONPATH="$PYTHONPATH:$PWD/code/"
"""

import os
import time
import numpy as np
from mentornet_pytorch.code.utils import Mentornet_nn
import reader
import torch
import utils
import argparse
import tqdm


parser = argparse.ArgumentParser(description='[train Mentornet]')

parser.add_argument('--train_dir', type=str, default='')
parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--mini_batch_size', type=int, default=32)
parser.add_argument('--max_step_train', type=float, default=3e4)
parser.add_argument('--learning_rate', type=float, default=0.1)

arg = parser.parse_args()

class Train_mentornet():
    def __init__(self, epoch, input_features):
        self.epoch = epoch
        self.model = utils.Mentornet_nn(input_features)
        self.batch_size = arg.mini_batch_size

    def _select_optim(self, opt='adam'):
        if opt == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters, lr = self.lr)
        else:
            raise NotImplementedError()

    def _set_lr(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def train(self):
        """Runs the model on the given data."""
        if not os.path.exists(arg.train_dir):
            os.makedirs(arg.train_dir)
        
        print('Start loading the data')
        train_data = reader.DataSet(arg.data_path, 'tr')
        test_data = reader.DataSet(arg.data_path, 'ts')
        print('Finish loading the data')

        batch_size = self.batch_size
        epoch_size = int(np.floor(train_data.num_examples / self.batch_size))
        parameter_info = ['Hyper Parameter Info:']
        parameter_info.append('=' * 20)
        parameter_info.append('data_dir={}'.format(arg.data_path))
        parameter_info.append('#train_examples={}'.format(train_data.num_examples))
        parameter_info.append('#test_examples={}'.format(test_data.num_examples))
        parameter_info.append('is_binary_label={}'.format(train_data.is_binary_label))
        parameter_info.append(
            'mini_batch_size = {}\nstarter_learning_rate = {}'.format(
                batch_size, arg.learning_rate))
        parameter_info.append('#iterations per epoch = {}'.format(epoch_size))
        parameter_info.append('=' * 20)
        print(parameter_info)

        train_loss = np.empty((self.epoch,))
        for epoch in tqdm(range(self.epoch)):
            accumulated_loss = 0

            for i in epoch_size:
                data = train_data.next_batch(batch_size)
                v_truth = data[:, 4]
                x = data[:, 0:4]
                v = self.model(x)

                self.optimizer.zero_grad()
                if train_data.is_binary_label:
                    loss = torch.nn.BCELoss(v, v_truth)
                else:
                    loss = torch.nn.MSELoss(v, v_truth)
                accumulated_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                self.optimizer.step()
            lr = self.optimizer.state_dict()["param_groups"][0]["lr"]

            train_loss[epoch] = accumulated_loss / epoch_size
            #TODO: log by WB

                

    def eval_model_once(self, data_path, model_dir):
        """Evaluates the model.

        Args:
            data_path: path where the data is stored.
            model_dir: path where the model checkpoints are stored.

        Returns:
            average loss
        """
        self.model.load_state_dict(torch.load(model_dir))
        batch_size = self.batch_size
        test_data = reader.Dataset(data_path, 'ts')
        epoch_size = int(np.floor(train_data.num_examples / arg.mini_batch_size))
        accumulated_loss = 0
        with torch.no_grad():
            for i in tqdm(range(epoch_size)):
                data = test_data.next_batch(batch_size)
                v_truth = data[:, 4]
                x = data[:, 0:4]
                v = self.model(x)
                if test_data.is_binary_label:
                    loss = torch.nn.BCELoss(v, v_truth)
                else:
                    loss = torch.nn.MSELoss(v, v_truth)
                accumulated_loss += loss.item()
        accumulated_loss /= epoch_size

