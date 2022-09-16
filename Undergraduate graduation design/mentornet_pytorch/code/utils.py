"""Utility functions for training the MentorNet models."""

from turtle import forward
from unicodedata import bidirectional
import numpy as np
import torch
import torch.nn as nn

def summarize_data_utilization(v, global_step, batch_size, epsilon=0.001):
    """Summarizes the samples of non-zero weights during training.

  Args:
    v: a numpy arrar [batch_size, 1] represents the sample weights.
      0: loss, 1: loss difference to the moving average, 2: label and 3: epoch,
      where epoch is an integer between 0 and 99 (the first and the last epoch).
    tf_global_step: the tensor of the current global step.
    batch_size: an integer batch_size
    epsilon: the rounding error. If the weight is smaller than epsilon then set
      it to zero.
  Returns:
    data_util: a num of data utilization.
    """
    rounded_v = np.max(v-epsilon, 0)
    nonzero_v = np.count_nonzero(rounded_v)

    data_util = (nonzero_v) / batch_size / (global_step + 2)
    data_util = np.min(data_util, 1)

    return data_util


def parse_dropout_rate_list(str_list):
    """Parse a comma-separated string to a list.

  The format follows [dropout rate, epoch_num]+ and the result is a list of 100
  dropout rate.

  Args:
    str_list: the input string.
  Returns:
    result: the converted list
    """
    str_list = np.array(str_list)
    values = str_list[np.arange(0, len(str_list), 2)]
    indexes = str_list[np.arange(1, len(str_list), 2)]

    values = [float(t) for t in values]
    indexes = [int(t) for t in indexes]

    assert len(values) == len(indexes) and np.sum(indexes) == 100
    for t in values:
        assert t >= 0.0 and t <= 1.0

    result = []
    for t in range(len(str_list) // 2):
        result.extend([values[t]] * indexes[t])
    return result


class Mentornet_nn(nn.Module):
    def __init__(self,
                input_fetures,
                label_embedding_size=2,
                epoch_embedding=5,
                num_fc_nodes=20):
        self.label_embedding = nn.Linear(2, label_embedding_size)
        self.epoch_embedding = nn.Linear(100, epoch_embedding)
        self.lstm = nn.LSTM(input_size=input_fetures, hidden_size=1, batch_first=True,
                            bidirectional=True)
        feat_dim = label_embedding_size+epoch_embedding+2
        self.fc1 = nn.Linear(feat_dim, num_fc_nodes)
        self.fc2 = nn.Linear(num_fc_nodes, 1)

    def forward(self, x):
        losses = x[:, 0].reshape(-1, 1)
        loss_diff = x[:, 1].reshape(-1, 1)
        labels = x[:, 2].reshape(-1, 1).int()
        epochs = x[:, 3].reshape(-1, 1).int()
        epochs = torch.min(epochs, torch.ones_like(epochs, dtype=torch.int32)*99)

        if len(losses.shape) <= 1:
            num_steps = 1
        else:
            num_steps = int(losses.shape[1])
        
        label_inputs = self.label_embedding(labels)
        epoch_inputs = self.epoch_embedding(epochs)

        _, (h_t, final_cell_state) = self.lstm(x)
        print(h_t.shape)

        x = torch.concat([label_inputs, epoch_inputs, h], 1)
        x = self.fc1(x)
        x = nn.Tanh(x)
        x = self.fc2(x)
        return nn.Sigmoid(x)
    

def mentornet(model,
            epoch,
            loss,
            loss_p_percentile,
            example_dropout_rates,
            burn_in_epoch=18,
            fixed_epoch_after_burn_in=True,
            loss_moving_average_decay=0.9,
            debug=False):
    """The MentorNet to train with the StudentNet.
    Args:
    epoch: a tensor [batch_size, 1] representing the training percentage. Each
      epoch is an integer between 0 and 99.
    loss: a tensor [batch_size, 1] representing the sample loss.
    labels: a tensor [batch_size, 1] representing the label. Every label is set
      to 0 in the current version.
    loss_p_percentile: a 1-d tensor of size 100, where each element is the
      p-percentile at that epoch to compute the moving average.
    example_dropout_rates: a 1-d tensor of size 100, where each element is the
      dropout rate at that epoch. Dropping out means the probability of setting
      sample weights to zeros proposed in Liang, Junwei, et al. "Learning to
      Detect Concepts from Webly-Labeled Video Data." IJCAI. 2016.
    burn_in_epoch: the number of burn_in_epoch. In the first burn_in_epoch, all
      samples have 1.0 weights.
    fixed_epoch_after_burn_in: whether to fix the epoch after the burn-in.
    loss_moving_average_decay: the decay factor to compute the moving average.
    debug: whether to print the weight information for debugging purposes.

    Returns:
        v: [batch_size, 1] weight vector.
    """
    if not fixed_epoch_after_burn_in:
        cur_epoch = epoch
    else:
        cur_epoch = min(epoch, burn_in_epoch)
    
    if cur_epoch < (burn_in_epoch-1):
        upper_bound = np.ones(loss.shape)
    else:
        upper_bound = np.zeros(loss.shape)
    
    this_dropout_rate = example_dropout_rates[cur_epoch, :]
    this_percentile = loss_p_percentile[cur_epoch, :]

    percentile_loss = np.percentile(loss, this_percentile*100)
    
    loss_moving_avg = (1 - loss_moving_average_decay) * percentile_loss

    #TODO: log by WB

    ones = torch.ones((loss.shape[0], 1), dtype=torch.float32)
    epoch_vec = ones * cur_epoch
    loss_diff = loss - ones*loss_moving_avg

    input_data = torch.stack([loss, lossdiff, labels, epoch_vec], 1).squeeze()
    print(input_data.shape)

    v = model(input_data)
    v = torch.max(v, upper_bound)

    v_dropout = probabilistic_sample(v, this_dropout_rate, 'random')
    v_dropout = torch.from_numpy(v_dropout).reshape(-1, 1)

    if debug:
        pass
    return v_dropout


def probabilistic_sample(v, rate=0.5, mode='binary'):
    """Implement the sampling techniques.

  Args:
    v: [batch_size, 1] the weight column vector.
    rate: in [0,1]. 0 indicates using all samples and 1 indicates
      using zero samples.
    mode: a string. One of the following 1) actual returns the actual sampling;
      2) binary returns binary weights; 3) random performs random sampling.
  Returns:
    v: [batch_size, 1] weight vector.
    """
    assert rate >= 0 and rate <= 1
    epsilon = 1e-5
    p = np.copy(v)
    p = np.reshape(p, -1)
    if mode == 'random':
        ids = np.random.choice(p.shape[0], int(p.shape[0] * (1 - rate)), replace=False)
    else:
        # Avoid 1) all zero loss and 2) zero loss are never selected.
        p += epsilon
        p /= np.sum(p)
        ids = np.random.choice(
            p.shape[0], int(p.shape[0] * (1 - rate)), p=p, replace=False)
    result = np.zeros(v.shape, dtype=np.float32)
    if mode == 'binary':
        result[ids, 0] = 1
    else:
        result[ids, 0] = v[ids, 0]
    return result