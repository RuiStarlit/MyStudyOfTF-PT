# -*- coding:utf-8 -*-
"""
Author: RuiStarlit
File: ResNet_test
Project: LearningPyTorch
Create Time: 2021-07-11

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import datetime
import matplotlib as plt

PARAS_FN = 'cifar_resnet18_params.pkl'
data_path = '../data/cifar10'
loss_func = nn.CrossEntropyLoss()


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        # 残差块的第一个卷积
        # 通道数变换in->out，每一层（除第一层外）的第一个block
        # 图片尺寸变换：stride=2时，w-3+2 / 2 + 1 = w/2，w/2 * w/2
        # stride=1时尺寸不变，w-3+2 / 1 + 1 = w
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 残差块的第二个卷积
        # 通道数、图片尺寸均不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差块的shortcut
        # 如果残差块的输入输出通道数不同，则需要变换通道数及图片尺寸，以和residual部分相加
        # 输出：通道数*2 图片尺寸/2
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # 通道数相同，无需做变换，在forward中identity = x
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, ResBlock, args):
        super(ResNet18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, args.num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        # print(*layers)
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


best_acc = 0
# 记录准确率，显示曲线
global_train_acc = []
global_test_acc = []

def train(model, trainloader, optimizer, log_interval):
    model.train()
    start_time = datetime.datetime.now()

    train_loss = 0
    correct = 0
    size = len(trainloader.dataset)

    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to('cuda'), labels.to('cuda')
        # Compute prediction and loss
        pred = model(images)
        loss = loss_func(pred, labels)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * len(images)
            print(f"loss:{loss:>7f} [{current:>5d}/{size:>5d}")
            acc = 100.* correct / size
            global_train_acc.append(acc)

    print('One Epoch Spend:', datetime.datetime.now() - start_time)

def test(model, testloader):
    size = len(testloader.dataset)
    num_batcher = len(testloader)
    test_loss, correct =0, 0
    with torch.no_grad():
        for X, y in testloader:
            X, y = X.cuda(), y.cuda()
            pred = model(X)
            test_loss += loss_func(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batcher
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss:{test_loss:>8f} \n")
    acc = 100.*correct / size
    global_test_acc.append(acc)
    global best_acc
    if acc > best_acc:
        best_acc =acc


def show_acc_curv(ratio):
    # 训练准确率曲线的x、y
    train_x = list(range(len(global_train_acc)))
    train_y = global_train_acc

    # 测试准确率曲线的x、y
    # 每ratio个训练准确率对应一个测试准确率
    test_x = train_x[ratio - 1::ratio]
    test_y = global_test_acc

    plt.title('CIFAR10 RESNET34 ACC')

    plt.plot(train_x, train_y, color='green', label='training accuracy')
    plt.plot(test_x, test_y, color='red', label='testing accuracy')

    # 显示图例
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('accs')

    plt.show()


import argparse
from torchvision import datasets, transforms
def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFA10 ResNet18 Test')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument('--no-train', action='store_true', default=False,
                        help='If train the Model')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--num_classes', type=int, default=10, metavar='N',
                        help='number of classes (default: 10)')
    args = parser.parse_args()


    transform = torchvision.transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_data = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    test_data  = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transform)

    train_load = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_load = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)

    model = ResNet18(ResBlock, args).to('cuda')
    print(model)

    if args.no_train:
        model.load_state_dict(torch.load(PARAS_FN))
        test(model, test_load, 0)
        return

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    start_time = datetime.datetime.now()

    for epoch in range(1, args.epochs + 1):
        train(model, train_load, optimizer, args.log_interval)

        # 每个epoch结束后用测试集检查识别准确度
        test(model, test_load)

    end_time = datetime.datetime.now()

    global best_acc
    print('CIFAR10 pytorch ResNet34 Train: EPOCH:{}, BATCH_SZ:{}, LR:{}, ACC:{}'.format(args.epochs, args.batch_size,
                                                                                        args.lr, best_acc))
    print('train spend time: ', end_time - start_time)

    # 每训练一个迭代记录的训练准确率个数
    ratio = len(train_data) / args.batch_size / args.log_interval
    ratio = int(ratio)

    # 显示曲线
    show_acc_curv(ratio)

if __name__ == '__main__':
    main()