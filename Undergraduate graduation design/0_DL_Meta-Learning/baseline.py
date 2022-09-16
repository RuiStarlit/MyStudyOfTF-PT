import argparse
# from multiprocessing import reduction
import torch.nn.functional as F
import torch
import torchvision
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from model import *
from utils import *
from noise_generator import *
from tqdm.notebook import tqdm
import copy
# import sys
from meta import *
import wandb


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


parser.add_argument('--lr', type=float, default=.1)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=.9)

parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--augment', type=int, default=3)


args = parser.parse_args(args=[])

print('Args:')
print(args)


class TrainNet():
    def __init__(self, args, fix=False):
        self.args = args
        if fix is True:
            # When running on my device, my device will crash
            reproduce(args.seed)
            set_cudnn(args.device)
        if self.args.dataset == 'cifar10':
            self.num_classes = 10
        elif self.args.dataset == 'cifar100':
            self.num_classes = 100
        else:
            raise NotImplementedError()
        self.device = args.device
        self.lr = args.lr
        self.resnet_option = 'A' # if self.num_classes == 10 else 'B'
        self._build_model()
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr = self.args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        
        
    def _set_lr(self, lr, opt='DL'):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print('Learning Rate is set to ' + str(lr))
    
    def _build_model(self, model='resnet32'):
        if model == 'resnet32':
            self.model = ResNet32(self.num_classes, option=self.resnet_option).to(self.device)
            # self.mdoel = torchvision.models.resnet34(
            #     True, {'num_classes':self.num_classes}
            #     ).to(self.device)
        elif model == 'resnet101':
            self.model = torchvision.models.resnet.resnet101(
                {'num_classes':self.num_classes}).to(self.device)
        elif model == 'resnet34':
            self.model = torchvision.models.resnet34(
                {'num_classes':self.num_classes}).to(self.device)
        elif model == 'wrn28':
            self.model = WideResNet(28, self.num_classes, 10, nc=3).to(self.device)
        else:
            raise NotImplementedError()
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    
    def save(self, name):
        torch.save(self.model.state_dict(), 'checkpoint/' + name + '.pt')
        print('Successfully save')
    
    def train(self, epochs=None):
        if epochs is None:
            epochs = self.args.max_epoch
        
        train_dataloader, meta_dataloader, test_dataloader, imbalanced_num_list = cifar_dataloader(
        seed=self.args.seed,
        dataset=self.args.dataset,
        num_meta_total=self.args.num_meta,
        imbalanced_factor=self.args.imbalanced_factor,
        corruption_type=self.args.corruption_type,
        corruption_ratio=self.args.corruption_ratio,
        batch_size=self.args.batch_size,
        num_workers=self.args.num_workers,
        augment= self.args.augment
    )
        if self.args.imbalanced_factor is not None:
            data = [[x, y] for (x, y) in zip(
                list(range(self.num_classes)), imbalanced_num_list)]
            table = wandb.Table(data=data, columns = ["class", "num"])
            wandb.log({"Imblanced List plot" : wandb.plot.line(table, "class", "num",
                title="Imblanced List")})
        
        bar = tqdm(range(epochs))
        for epoch in bar:
            # scheduler
            if (epoch) in self.args.milestone:
                self.lr = self.lr * 0.1
                self._set_lr(self.lr)
            correct = 0
            train_loss = 0
            for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.model.train()
                outputs = self.model(inputs)
                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()
                self.optimizer.zero_grad()
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss +=loss.item()
            
            train_loss /= len(train_dataloader)
            train_acc = correct / len(train_dataloader.dataset)

            self.model.eval()
            correct = 0
            test_loss = 0
            with torch.no_grad():
                for inputs, labels in test_dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    test_loss += self.criterion(outputs, labels).item()
                    _, pred = outputs.max(1)
                    correct += pred.eq(labels).sum().item()
            test_loss /= len(test_dataloader)
            test_acc = correct / len(test_dataloader.dataset)

            # print(f'Epoch: {epoch+1}| Train loss:{train_loss:.4f} acc:{train_acc:.4f}|'
            # f'Test loss:{test_loss:.4f} acc:{test_acc:.4f}'
            # )
            wandb.log({'Epoch': epoch+1, 'Train loss':train_loss, 'Train acc':train_acc,
            'Test loss':test_loss, 'Test acc':test_acc
            })
            bar.set_description(
                f'Epoch:{epoch+1}|Test acc:{test_acc:.2f} loss:{test_loss:.3f}') 
    def test(self, b = 1024):
        test_dataloader = cifar_testdataloader(self.args.dataset, batch_size=b)
        self.model.eval()
        correct = 0
        test_loss = 0
        confusion = np.zeros((self.num_classes, self.num_classes))
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                test_loss += self.criterion(outputs, labels).item()
                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()
                for i in range(labels.shape[0]):
                    confusion[labels[i].item()][pred[i].item()] += 1
        for i in range(self.num_classes):
            confusion[i] /= confusion[i].sum()

        test_loss /= len(test_dataloader)
        test_acc = correct / len(test_dataloader.dataset)
        print(f'[Test] Loss:{test_loss:.6f}| Acc:{test_acc}')
        return test_loss, test_acc, confusion

