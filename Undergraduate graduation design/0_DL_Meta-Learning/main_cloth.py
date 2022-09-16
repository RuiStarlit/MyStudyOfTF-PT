import argparse
import copy
import datetime

# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
# from torch.cuda.amp import autocast, GradScaler
# from multiprocessing import reduction
import torch.nn.functional as F
import torchvision
from tdigest import TDigest
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import wandb
# import sys
from meta import *
from model import *
from noise_generator import *
from utils import *
from data_helper_clothing1m import prepare_data

parser = argparse.ArgumentParser(description='[Robust DL based on Meata Learning]')

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

parser.add_argument('--threshlod', type=int, default=-1)
parser.add_argument('--c_threshold', type=float, default=0.7)
parser.add_argument('--semi', type=str, default='mixlabel')   # cutmix, mixup, mixlabel
parser.add_argument('--semi_p', type=float, default=0.3)
parser.add_argument('--semi_beta', type=float, default=1.0)

# parser.add_argument('--amp', type=bool, default=False)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--augment', type=int, default=3)
parser.add_argument('--num_consistency', type=int, default=5)


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
        self.num_classes = 14

        self.device = args.device
        self.lr = args.lr
        self._build_model(self.args.model)
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
        # self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[30,60], gamma=0.1)
        # scheduler1 = lr_scheduler.LinearLR(self.optimizer, start_factor=0.01, total_iters=3)
        # scheduler2 = lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.2)
        # scheduler = lr_scheduler.SequentialLR(self.optimizer, schedulers=[scheduler1, scheduler2], milestones=[3])

        self.date = (datetime.datetime.now()).strftime("_%m-%d_%H-%M")
        self.name = 'CheckPoint-'+self.date
        
        
    def _set_lr(self, lr, opt='DL'):
        if opt == 'DL':
            self.lr = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            print('[Main]Learning Rate is set to ' + str(lr))
        else:
            self.meta_lr = lr
            for param_group in self.meta_optimizer.param_groups:
                param_group['lr'] = lr
            print('[Main]Meta Net\'s Learning Rate is set to ' + str(lr))
    
    def _build_model(self, model='resnet50', meta_model='MLP'):
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
        elif model == 'resnet50':
            self.model = torchvision.models.resnet50(
                pretrained=True,
                # num_classes=self.num_classes
            ).to(self.device)
        else:
            raise NotImplementedError()
        self.meta_net = Metanet_multiplex(self.num_classes,
        self.args.meta_net_num_layers,
        self.args.meta_net_hidden_size,
        ).to(self.device)
    
    def load(self, path, opt='all'):
        if opt =='all':
            checkpoint = torch.load(path)
            acc = checkpoint['acc']
            epoch = checkpoint['epoch']
            lr = checkpoint['lr']
            self.model.load_state_dict(checkpoint['model'])
            self.meta_net.load_state_dict(checkpoint['meta'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer'])
            self.lr = lr
            print(f'[Load] Acc:{acc:.4f}|Epoch:{epoch}|Lr:{lr}')
        elif opt == 'DL':
            self.model.load_state_dict(torch.load(path+'.pt'))
            print("[Load]success")
        else:
            self.meta_net.load_state_dict(torch.load(path+'-meta.pt'))
            print("[Load]success")
    
    def save(self, name, acc=0, epoch=0, opt='all', verbose=True):
        if opt == 'all':
            checkpoint = {
            'acc': acc,    
            'epoch': epoch,
            'lr': self.lr,
            'model': self.model.state_dict(),
            'meta':self.meta_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'meta_optimizer':self.meta_optimizer.state_dict(),
            }
            torch.save(checkpoint, 'checkpoint/' + name + '.pth.tar')

        elif opt == 'old':
            torch.save(self.model.state_dict(), 'checkpoint/' + name + '.pt')
            torch.save(self.meta_net.state_dict(), 'checkpoint/' + name + '-meta.pt')
        elif opt == 'meta':
            torch.save(self.meta_net.state_dict(), 'checkpoint/' + name + '-meta.pt')
        if verbose:
            print('[Save] Successfully')
    
    def train(self, epochs=None, ckp=0):
        if epochs is None:
            epochs = self.args.max_epoch

        meta_dataloader, train_dataloader, val_loader, test_dataloader, num_classes = prepare_data(
            batch_size=self.args.batch_size,
            num_workers= self.args.num_workers
        )


        innerepochs = self.args.innerepochs
        meta_dataloader_iter = DataIterator(meta_dataloader)

        # plot the meta-weight
        # y = np.empty((100))
        # with torch.no_grad():
        #     for i in range(100):
        #         x = torch.tensor(i / 10).unsqueeze(0).to(self.device)
        #         mx = self.meta_net(x)
        #         y[i] = mx.item()
        # data = [[x, y] for (x, y) in zip(list(np.arange(0,100)/10), y.tolist())]
        # table = wandb.Table(data=data, columns = ["loss", "v"])
        # wandb.log({"Meta-Weight" : wandb.plot.line(table, "loss", "v",
        #     title="Meta-Weight-init")})



        self.weight_digest1 = None
        self.weight_digest2 = None
        # TEMP vvvvvvvvvvvvvvv
        # if ckp != 0:
        #     count =  0
        #     weight_digest1 = TDigest()
        # TEMP ^^^^^^^^^^^^^^^

        iters = len(train_dataloader)

        print('[Main] Training.... ')
        bar = tqdm(range(epochs))
        for epoch in bar:
            epoch_p = epoch+ckp
            loss_p = self.args.loss_p
            # scheduler
            if epoch+ckp >= 5 :
                self.lr = self.lr * 0.1
                self._set_lr(self.lr)
            correct = 0
            train_loss = 0

            if self.weight_digest1 is not None:
                self.weight_digest2 = self.weight_digest1
                print('[Train:{}] The {}percentile of weight is {:6f}.'.format(
                    epoch+1+ckp, self.args.threshold*100,
                    self.weight_digest2.percentile(self.args.threshold*100)
                ))
            if self.args.threshold > 0 and self.args.burn_in <= epoch+ckp-1:
                # init TDigest Algorithm 
                # use weight infor from the previous epoch
                self.weight_digest1 = TDigest()
                # print(f'[Debug] digest reset')
            self.model.train()

            for iteration, (inputs, labels) in enumerate(train_dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # if (iteration+1) % self.args.interval == 0 and self.args.burn_in <= epoch:
                if (iteration+1) % self.args.interval == 0:

                    pseudo_model = torchvision.models.resnet50().to(self.device)
                    pseudo_model.load_state_dict(self.model.state_dict())
                    for inner in range(innerepochs):
                        pseudo_y = pseudo_model(inputs)
                        pseudo_loss = F.cross_entropy(pseudo_y, labels.long(), reduction='none')
                        pseudo_loss = pseudo_loss.reshape((-1, 1))
                        meta_weight = self.meta_net(pseudo_loss.data, labels, loss_p, epoch_p) # Tensor.data do not contain grad
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

                        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ #

                    meta_inputs, meta_labels = next(meta_dataloader_iter)

                    # Using Meta-Data to Update  vvvvvvvvvvvvvvvvvvvvvvvvvvvvv #
                    meta_inputs, meta_labels = meta_inputs.to(self.device), meta_labels.to(self.device)
                    meta_outputs = pseudo_model(meta_inputs)
                    meta_loss = self.criterion(meta_outputs, meta_labels.long())
                    
                    # l1_penalty_weight = torch.sum([torch.norm(w, 1) for w in meta_weight])
                    # meta_loss = meta_loss + 0.01 * l1_penalty_weight
                    
                    self.meta_optimizer.zero_grad()
                    meta_loss.backward()
                    self.meta_optimizer.step()
                    # Meta-Net part^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   #


                outputs = self.model(inputs)
                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()

                loss = F.cross_entropy(outputs, labels.long(), reduction='none') # default:mean
                loss_reshpae = loss.reshape((-1, 1))

                with torch.no_grad():
                    weight = self.meta_net(loss_reshpae, labels, loss_p, epoch_p)

                if self.args.threshold <= 0 or self.args.burn_in > epoch+ckp:
                    loss = torch.mean(weight * loss_reshpae)
                if self.args.threshold > 0 and self.args.burn_in <= epoch+ckp-1:
                    weight_np = weight.detach().cpu().numpy()
                    self.weight_digest1.batch_update(weight_np.reshape(-1))
                
                if self.args.threshold > 0 and self.args.burn_in <= epoch+ckp:
                    # threshold = self.weight_digest2.percentile(self.args.threshold*100)
                    threshold = self.args.threshold
                    c_threshold = self.args.c_threshold
                    c_ind = torch.ge(weight,c_threshold)[:,0]
                    n_ind = torch.le(weight,threshold)[:, 0]
                    # actually x_noise mean x_unimportant

                    print(f'\r[Debug] {c_ind.sum()/self.args.batch_size*100:>5.2f}%important|'
                    f'{(n_ind.sum())/self.args.batch_size*100:>5.2f}%unimportant|'
                    f'{(1-(c_ind.sum()+n_ind.sum())/self.args.batch_size)*100:>5.2f}%normal at this batch'
                    ,end = "")

                    weight[n_ind] = 0
                    flag_semi = False
                    if self.args.semi == 'mixlabel':
                        mlabel = mix_label(labels, outputs)
                        loss_mix = F.cross_entropy(outputs[n_ind], mlabel[n_ind], reduction='none')
                        loss_mix = loss_mix.reshape((-1, 1))
                        loss = torch.mean(weight * loss_reshpae) + torch.mean(self.args.semi_p*loss_mix)
                        flag_semi = True
                    elif self.args.semi == 'mixup':
                        if c_ind.sum() >=  n_ind.sum()/5 and n_ind.sum() > 0:
                            x_noise, y_noise, ym_clean, lam = mixup(
                                self.args.semi_beta, inputs, labels.long(), c_ind, n_ind,
                                self.device
                                )
                            output = self.model(x_noise)
                            semi_loss = lam * self.criterion(output, y_noise) + (1 - lam) * self.criterion(output, ym_clean)
                            loss = torch.mean(weight * loss_reshpae) + self.args.semi_p*semi_loss
                            flag_semi = True
                    elif self.args.semi == 'cutmix':
                        if c_ind.sum() >=  n_ind.sum()/5 and n_ind.sum() > 0:
                            x_noise, y_noise, ym_clean, lam = cutmix(
                                self.args.semi_beta, inputs, labels.long(), c_ind, n_ind,
                                self.device
                                )
                            output = self.model(x_noise)
                            semi_loss = lam * self.criterion(output, y_noise) + (1 - lam) * self.criterion(output, ym_clean)
                            loss = torch.mean(weight * loss_reshpae) + self.args.semi_p*semi_loss
                            flag_semi = True
                    elif self.args.semi =='consistency':
                        if c_ind.sum() >=  n_ind.sum()/5 and n_ind.sum() > 0:
                            m = self.args.num_consistency
                            x_noise = inputs[n_ind]
                            new_x = torch.empty((x_noise.shape[0]*m, x_noise.shape[1],x_noise.shape[2],x_noise.shape[3]))
                            for i in range(x_noise.shape[0]):
                                for j in range(m):
                                    new_x[i*m+j] = Randaugment(x_noise[i])
                            new_x = new_x.to(self.device)
                            # x_ = tensor2Image(x_noise[0])
                            # x1_ = tensor2Image(new_x[0])

                            output = self.model(new_x)
                            new_y = torch.empty((x_noise.shape[0], output.shape[1]))
                            for i in range(x_noise.shape[0]):
                                new_y[i] = sum([output[i*m+j] for j in range(m)]) / m
                            new_y = new_y.to(self.device)
                            _, self_pred = new_y.max(1)
                            semi_loss = self.criterion(outputs[n_ind], self_pred)
                            loss = torch.mean(weight * loss_reshpae) + self.args.semi_p*semi_loss
                            # return x_noise, new_x, output, outputs[n_ind], new_y, semi_loss # debug
                            flag_semi = True
                    else:
                        raise NotImplementedError()
                    # loss = (1-self.args.semi_p)*loss + self.args.semi_p*semi_loss
                    if not flag_semi:
                        loss = torch.mean(weight * loss_reshpae)

                else:
                    pass
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()
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
            
            print('\n')
            print(f'[Train:{epoch+1+ckp}] Train loss:{train_loss:.4f} acc:{train_acc:.4f}|',end='')
            print(f'Test loss:{test_loss:.4f} acc:{test_acc:.4f}')
            self.save(self.name, verbose=False, acc=test_acc, epoch=epoch+1+ckp)
            if epoch == 79:
                self.save(self.name+'e80', verbose=True, acc=test_acc, epoch=epoch+1+ckp)
            # print('[Main]Saving CheckPoint at '+ str(epoch+ckp+1) + ' Epoch')
            wandb.log({'Epoch': epoch+1+ckp, 'Train loss':train_loss, 'Train acc':train_acc,
            'Test loss':test_loss, 'Test acc':test_acc
            })
            bar.set_description(f'Epoch:{epoch+1+ckp}|Test acc:{test_acc:.4f} loss:{test_loss:.4f}')
            bar.set_postfix({'Lr':self.optimizer.state_dict()["param_groups"][0]["lr"]})
            if self.args.model == 'wrn28':
                torch.cuda.empty_cache()


            # if (epoch+1) % self.args.plot_interval == 0: # plot the meta-weight
            #     y = np.empty((100))
            #     with torch.no_grad():
            #         for i in range(100):
            #             x = torch.tensor(i / 10).unsqueeze(0).to(self.device)
            #             mx = self.meta_net(x)
            #             y[i] = mx.item()
            #     data = [[x, y] for (x, y) in zip(list(np.arange(0,100)/10), y.tolist())]
            #     table = wandb.Table(data=data, columns = ["loss", "v"])
            #     plot_name = 'Meta-Weight-'+str(epoch+1)+'Epoch'
            #     wandb.log({plot_name : wandb.plot.line(table, "loss", "v",
            #         title=plot_name)})


    def test(self, b = 512):
        test_dataloader = cifar_testdataloader(self.args.dataset, batch_size=b)
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
        print(f'[Test] Loss:{test_loss:.6f}| Acc:{test_acc}')
        return test_loss, test_acc
    
    def _get_weight(self):
        return self.weight_digest1


def totorch(x):
    return torch.autograd.Variable(x)
