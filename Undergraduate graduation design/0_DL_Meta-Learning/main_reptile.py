import argparse
import copy

# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
# from multiprocessing import reduction
import torch.nn.functional as F
import torchvision
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
parser.add_argument('--meta_net_hidden_size', type=int, default=128)
parser.add_argument('--meta_net_num_layers', type=int, default=1)

parser.add_argument('--lr', type=float, default=.1)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=.9)
parser.add_argument('--meta_lr', type=float, default=1e-3)
parser.add_argument('--meta_weight_decay', type=float, default=0.)
parser.add_argument('--inner_lr', type=float, default=.1)

parser.add_argument('--meta_method', type=str, default='g1+g2')
parser.add_argument('--innerepochs', type=int, default=4)
parser.add_argument('--interval', type=int, default=1)
parser.add_argument('--plot_interval', type=int, default=10)
parser.add_argument('--burn_in', type=int, default=10)

parser.add_argument('--threshlod', type=int, default=-1)
parser.add_argument('--semi', type=str, default='cutmix')   # cutmix, mixup, mixlabel
parser.add_argument('--semi_p', type=float, default=0.1)
parser.add_argument('--semi_beta', type=float, default=1.0)


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
        self.inner_lr= args.inner_lr
        self._build_model()
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr = self.args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        self.meta_optimizer = torch.optim.Adam(
            self.meta_net.parameters(),
            lr = args.meta_lr,
            weight_decay=args.meta_weight_decay
        )
        
        
    def _set_lr(self, lr, opt='DL'):
        if opt == 'DL':
            self.lr = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            print('Learning Rate is set to ' + str(lr))
        else:
            self.meta_lr = lr
            for param_group in self.meta_optimizer.param_groups:
                param_group['lr'] = lr
            print('Meta Net\'s Learning Rate is set to ' + str(lr))
    
    def _build_model(self, model='resnet32', meta_model='MLP'):
        if model == 'resnet32':
            self.model = ResNet32(self.num_classes).to(self.device)
        elif model == 'resnet101':
            self.model = torchvision.models.resnet.resnet101(
                kwargs={'num_classes':self.num_classes}).to(self.device)
        else:
            raise NotImplementedError()
        self.meta_net = Metanet_mlp(self.args.meta_net_num_layers,
            self.args.meta_net_hidden_size).to(self.device)
    
    def load(self, path, opt='DL'):
        if opt == 'DL':
            self.model.load_state_dict(torch.load(path))
        else:
            self.meta_net.load_state_dict(torch.load(path))
        print("success")
    
    def save(self, name, opt='all'):
        if opt == 'all':
            torch.save(self.model.state_dict(), 'checkpoint/' + name + '.pt')
            torch.save(self.meta_net.state_dict(), 'checkpoint/' + name + '-meta.pt')
        elif opt == 'meta':
            torch.save(self.meta_net.state_dict(), 'checkpoint/' + name + '-meta.pt')
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
    )
        innerepochs = self.args.innerepochs
        meta_dataloader_iter = iter(meta_dataloader)

        # plot the meta-weight
        y = np.empty((100))
        with torch.no_grad():
            for i in range(100):
                x = torch.tensor(i / 10).unsqueeze(0).to(self.device)
                mx = self.meta_net(x)
                y[i] = mx.item()
        data = [[x, y] for (x, y) in zip(list(np.arange(0,100)/10), y.tolist())]
        table = wandb.Table(data=data, columns = ["loss", "v"])
        wandb.log({"Meta-Weight" : wandb.plot.line(table, "loss", "v",
            title="Meta-Weight-init")})
        
        if self.args.imbalanced_factor is not None:
            data = [[x, y] for (x, y) in zip(list(range(self.num_classes)), imbalanced_num_list)]
            table = wandb.Table(data=data, columns = ["class", "num"])
            wandb.log({"Imblanced List plot" : wandb.plot.line(table, "class", "num",
                title="Imblanced List")})


        print('[Main] Training.... ')
        bar = tqdm(range(epochs))
        for epoch in bar:
            # scheduler
            if epoch >= 70 and epoch % 15 == 0:
                self.lr = self.lr / 10
                self._set_lr(self.lr)
            correct = 0
            train_loss = 0
            for iteration, (inputs, labels) in enumerate(train_dataloader):
                self.model.train()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # if (iteration+1) % self.args.interval == 0 and self.args.burn_in <= epoch:
                if (iteration+1) % self.args.interval == 0:
                    weights_before = copy.deepcopy(self.model.state_dict())
                    state_opt = self.optimizer.state_dict()
                    lookahead = Lookahead(self.optimizer, k=innerepochs-1, alpha=0.8) # 初始化Lookahead
                    for inner in range(innerepochs-1):
                        inputs, labels = totorch(inputs), totorch(labels)
                        pseudo_outputs = self.model(inputs)
                        loss = F.cross_entropy(pseudo_outputs, labels.long())
                        lookahead.zero_grad()
                        loss.backward()
                        lookahead.step()

                        # Inner gradients update! Can choose different methods vvv #
                        # for param in self.model.parameters():
                        #     param.data -= self.lr * param.grad.data

                        # pseudo_optimizer = MetaSGD(self.model, self.model.parameters(), 
                        #     lr=self.lr)
                        # pseudo_grads = torch.autograd.grad(
                        #     pseudo_loss, self.model.parameters(), create_graph=True)
                        # pseudo_optimizer.load_state_dict(self.optimizer.state_dict())
                        # pseudo_optimizer.step(pseudo_grads)
                        # del pseudo_grads

                        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ #
                    # model_weights_after = self.model.state_dict()
                    # # outerstepsize = outerstepsize0 * (1 - iteration / niterations) # linear schedule
                    # self.model.load_state_dict({name : 
                    #     weights_before[name] + \
                    #     (model_weights_after[name] - weights_before[name]) * self.inner_lr 
                    #     for name in weights_before})
                    
                    pseudo_model = ResNet32(self.num_classes).to(self.args.device)
                    pseudo_model.load_state_dict(self.model.state_dict())
                    pseudo_y = pseudo_model(inputs)
                    pseudo_loss = F.cross_entropy(pseudo_y, labels.long(), reduction='none')
                    pseudo_loss = pseudo_loss.reshape((-1, 1))
                    meta_weight = self.meta_net(pseudo_loss.data)
                    pseudo_loss = torch.mean(meta_weight * pseudo_loss)

                    pseudo_grads = torch.autograd.grad(
                    pseudo_loss, pseudo_model.parameters(), create_graph=True)

                    # Inner gradients update! Can choose different methods vvv #
                    # inner_update(self.model, pseudo_grads , self.lr)

                    pseudo_optimizer = MetaSGD(pseudo_model, pseudo_model.parameters(), 
                        lr=self.lr)
                    pseudo_optimizer.load_state_dict(self.optimizer.state_dict())
                    pseudo_optimizer.step(pseudo_grads)
                    del pseudo_grads

                    try:
                        meta_inputs, meta_labels = next(meta_dataloader_iter)
                    except StopIteration:
                        meta_dataloader_iter = iter(meta_dataloader)
                        meta_inputs, meta_labels = next(meta_dataloader_iter)

                    # Using Meta-Data to Update  vvvvvvvvvvvvvvvvvvvvvvvvvvvvv #
                    meta_inputs, meta_labels = meta_inputs.to(self.device), meta_labels.to(self.device)
                    meta_outputs = pseudo_model(meta_inputs)
                    meta_loss = self.criterion(meta_outputs, meta_labels.long())
                    meta_loss += 0.1*(meta_weight**2).sum()/2
                    self.meta_optimizer.zero_grad()
                    meta_loss.backward()
                    self.meta_optimizer.step()
                    # Meta-Net part^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   #
                    self.model.load_state_dict(weights_before)
                
                outputs = self.model(inputs)
                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()

                loss = F.cross_entropy(outputs, labels.long(), reduction='none') # default:mean
                loss_reshpae = loss.reshape((-1, 1))

                with torch.no_grad():
                    weight = self.meta_net(loss_reshpae)
                loss = torch.mean(weight * loss_reshpae)
                
                if self.args.threshold > 0 and self.args.burn_in <= epoch:
                    threshold = torch.mean(weight) * self.args.threshold
                    ind = torch.ge(weight,threshold)[:,0]
                    if self.args.semi == 'mixlabel':
                        mlabel = mix_label(labels, outputs)
                        loss = F.cross_entropy(outputs, mlabel, reduction='none')
                        loss = loss.reshape((-1, 1))
                        loss = torch.mean(weight * loss)
                        semi_loss = loss
                    elif self.args.semi == 'mixup':
                        semi_loss = mixup(
                            self.args.semi_beta, outputs, self.criterion, inputs, labels.long(), ind
                            )
                    elif self.args.semi == 'cutmix':
                        semi_loss = cutmix(
                            self.args.semi_beta, outputs, self.criterion, inputs, labels.long(), ind
                            )
                    else:
                        raise NotImplementedError()
                    loss = (1-self.args.semi_p)*loss + self.args.semi_p*semi_loss

                else:
                    pass
                train_loss += loss.item()

                if (iteration+1) % self.args.interval == 0:
                    self.optimizer.load_state_dict(state_opt)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # DNN part ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^      #

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
            bar.set_description(f'Epoch:{epoch+1}|Test acc:{test_acc:.2f} loss:{test_loss:.3f}') 


            if (epoch+1) % self.args.plot_interval == 0: # plot the meta-weight
                y = np.empty((100))
                with torch.no_grad():
                    for i in range(100):
                        x = torch.tensor(i / 10).unsqueeze(0).to(self.device)
                        mx = self.meta_net(x)
                        y[i] = mx.item()
                data = [[x, y] for (x, y) in zip(list(np.arange(0,100)/10), y.tolist())]
                table = wandb.Table(data=data, columns = ["loss", "v"])
                plot_name = 'Meta-Weight-'+str(epoch+1)+'Epoch'
                wandb.log({plot_name : wandb.plot.line(table, "loss", "v",
                    title=plot_name)})


def totorch(x):
    return torch.autograd.Variable(x)
