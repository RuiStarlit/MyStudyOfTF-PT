import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F
import pickle
import argparse
from helper_functions_twitter import *

from tqdm.notebook import tqdm

import wandb
# import sys
from meta import *
from model import *
from noise_generator import *
from utils import *

parser = argparse.ArgumentParser(description='[Robust DL based on Meata Learning]')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--imbalanced_factor', type=int, default=None)
parser.add_argument('--corruption_type', type=str, default=None)
parser.add_argument('--corruption_ratio', type=float, default=0.)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--max_epoch', type=int, default=100)

device = "cuda" if torch.cuda.is_available() else "cpu"
parser.add_argument('--device', type=str, default=device)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--meta_net_hidden_size', type=int, default=80)
parser.add_argument('--meta_net_num_layers', type=int, default=00)

parser.add_argument('--lr', type=float, default=.1)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=.9)
parser.add_argument('--meta_lr', type=float, default=1e-3)
parser.add_argument('--meta_weight_decay', type=float, default=0.)
parser.add_argument('--loss_p', type=float, default=0.75)

parser.add_argument('--meta_method', type=str, default='MAML')
parser.add_argument('--innerepochs', type=int, default=1)
parser.add_argument('--interval', type=int, default=1)
parser.add_argument('--burn_in', type=int, default=10)


# parser.add_argument('--amp', type=bool, default=False)
parser.add_argument('--num_workers', type=int, default=0)


args = parser.parse_args(args=[])

window_size = 1

# note that we encode the tags with numbers for later convenience
tag_to_number = {
    u'N': 0, u'O': 1, u'S': 2, u'^': 3, u'Z': 4, u'L': 5, u'M': 6,
    u'V': 7, u'A': 8, u'R': 9, u'!': 10, u'D': 11, u'P': 12, u'&': 13, u'T': 14,
    u'X': 15, u'Y': 16, u'#': 17, u'@': 18, u'~': 19, u'U': 20, u'E': 21, u'$': 22,
    u',': 23, u'G': 24
}

embeddings = embeddings_to_dict('../data/Tweets/embeddings-twitter.txt')
vocab = embeddings.keys()

# we replace <s> with </s> since it has no embedding, and </s> is a better embedding than UNK
X_train, Y_train = data_to_mat('../data/Tweets/tweets-train.txt', vocab, tag_to_number, window_size=window_size,
                     start_symbol=u'</s>')
X_dev, Y_dev = data_to_mat('../data/Tweets/tweets-dev.txt', vocab, tag_to_number, window_size=window_size,
                         start_symbol=u'</s>')
X_test, Y_test = data_to_mat('../data/Tweets/tweets-devtest.txt', vocab, tag_to_number, window_size=window_size,
                             start_symbol=u'</s>')

def prepare_data(corruption_matrix, gold_fraction=0.5, merge_valset=True):
    np.random.seed(1)

    twitter_tweets = np.copy(X_train)
    twitter_labels = np.copy(Y_train)
    if merge_valset:
        twitter_tweets = np.concatenate([twitter_tweets, np.copy(X_dev)], axis=0)
        twitter_labels = np.concatenate([twitter_labels, np.copy(Y_dev)])

    indices = np.arange(len(twitter_labels))
    np.random.shuffle(indices)

    twitter_tweets = twitter_tweets[indices]
    twitter_labels = twitter_labels[indices].astype(np.long)

    num_gold = int(len(twitter_labels)*gold_fraction)
    num_silver = len(twitter_labels) - num_gold

    for i in range(num_silver):
        twitter_labels[i] = np.random.choice(num_classes, p=corruption_matrix[twitter_labels[i]])

    dataset = {'x': twitter_tweets, 'y': twitter_labels}
    gold = {'x': dataset['x'][num_silver:], 'y': dataset['y'][num_silver:]}

    return dataset, gold, num_gold, num_silver

def uniform_mix_C(mixing_ratio):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)

def flip_labels_C(corruption_prob):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(1)

    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C

reg_str = 5e-5
num_epochs = 20
num_classes = 25
hidden_size = 256
batch_size = 64
embedding_dimension = 50
example_size = (2*window_size + 1)*embedding_dimension
init_lr = 0.001
num_examples = Y_train.shape[0]
num_batches = num_examples//batch_size

class MetaDataset(torch.utils.data.Dataset):#需要继承data.Dataset
    def __init__(self, dataset, len, batch_size):
        self.dataset = dataset
        self.len = len
        self.index = 0
        self.batch_size = batch_size
    def __call__(self):
        idx = self.index*self.batch_size
        self.index += 1
        if self.index == self.len//self.batch_size:
            self.index = 0
        x_batch = to_embeds(self.dataset['x'][idx:idx+self.batch_size])
        y_batch = self.dataset['y'][idx:idx+self.batch_size]
        x, y = V(torch.from_numpy(x_batch).cuda()), V(torch.from_numpy(y_batch).cuda())
        return x, y

    def __len__(self):
        return self.len


# //////////////////////// defining graph ////////////////////////
class ThreeLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(example_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
            )

        self.init_weights()

    def init_weights(self):
        self.main[0].weight.data.normal_(0, 1/np.sqrt(example_size))
        self.main[0].bias.data.zero_()
        self.main[2].weight.data.normal_(0, 1/np.sqrt(256))
        self.main[2].bias.data.zero_()
        self.main[4].weight.data.normal_(0, 1/np.sqrt(256))
        self.main[4].bias.data.zero_()


    def forward(self, x):
        return self.main(x)


to_embeds = lambda x: word_list_to_embedding(x, embeddings, embedding_dimension)

def train(corruption_level=0, gold_fraction=0.02, get_C=uniform_mix_C):
    lr = 0.001
    # np.random.seed(1)
    # torch.manual_seed(1)
    # torch.cuda.manual_seed(1)
    net = ThreeLayerNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr, weight_decay=reg_str)
    meta_net = Metanet_multiplex(num_classes,1,80).to(device)
    meta_optimizer = torch.optim.Adam(meta_net.parameters(),lr = 1e-4)

    C = get_C(corruption_level)

    dataset, gold, num_gold, num_silver = prepare_data(C, gold_fraction)

    meta_dataset = MetaDataset(gold, num_gold, 64)

    num_examples = num_silver
    num_batches = num_examples//batch_size
    indices = np.arange(num_examples)

    bar = tqdm(range(num_epochs))
    for epoch in bar:
        # shuffle data indices every epoch
        np.random.shuffle(indices)
        correct = 0
        train_loss = 0
        if (epoch+1) % 6 == 0:
            lr = lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('[Main] set lr to',lr)

        net.train()
        for i in range(num_batches):
            offset = i * batch_size

            x_batch = to_embeds(dataset['x'][indices[offset:offset + batch_size]])
            y_batch = dataset['y'][indices[offset:offset + batch_size]]
            data, target = V(torch.from_numpy(x_batch).cuda()), V(torch.from_numpy(y_batch).cuda())
            
            pseudo_model = ThreeLayerNet().to(device)
            pseudo_y = pseudo_model(data)
            pseudo_loss = F.cross_entropy(pseudo_y, target.long(), reduction='none')
            pseudo_loss = pseudo_loss.reshape((-1, 1))
            meta_weight = meta_net(pseudo_loss.data, target, 0.7, epoch)
            pseudo_loss = torch.mean(meta_weight * pseudo_loss)
            pseudo_grads = torch.autograd.grad(
                pseudo_loss, pseudo_model.parameters(), create_graph=True
            )

            pseudo_optimizer = MetaSGD(pseudo_model, pseudo_model.parameters(), lr=lr)
            pseudo_optimizer.step(pseudo_grads)
            del pseudo_grads

            meta_x, meta_y = meta_dataset()
            

            meta_ouputs = pseudo_model(meta_x)
            meta_loss = F.cross_entropy(meta_ouputs, meta_y.long())

            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()

            outputs = net(data)
            _, pred = outputs.max(1)
            correct += pred.eq(target.data).sum().item()
            loss = F.cross_entropy(outputs, target.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()


        train_loss /= num_batches
        train_acc = correct / num_silver

        # //////////////////////// evaluate method ////////////////////////

        net.eval()
        with torch.no_grad():
            data, target = V(torch.from_numpy(to_embeds(X_test)).cuda()), \
                    V(torch.from_numpy(Y_test.astype(np.long)).cuda())
            output = net(data)
            test_loss = F.cross_entropy(output, target.long()).item()
    
        _, pred = output.max(1)
        correct = pred.eq(target.data).sum().item()

        test_acc = correct / len(Y_test)
        print(f'[Train:{epoch+1}] Train loss:{train_loss:.4f} acc:{train_acc:.4f}|',end='')
        print(f'Test loss:{test_loss:.4f} acc:{test_acc:.4f}')
        bar.set_description(f'Epoch:{epoch+1}|Test acc:{test_acc:.4f} loss:{test_loss:.4f}')
        bar.set_postfix({'Lr':optimizer.state_dict()["param_groups"][0]["lr"]})


    return test_acc




