import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import numpy as np
import sys
import math


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet32(nn.Module):
    def __init__(self, num_classes=10, block=BasicBlock, num_blocks=[5, 5, 5], option='A'):
        super(ResNet32, self).__init__()
        self.in_planes = 16
        self.option = option

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        # self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, option=self.option))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class HiddenLayer(nn.Module):
    def __init__(self, input, output):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input, output)
        self.relu = nn.ReLU()
        self.act = nn.ELU()

    def forward(self, x):
        return self.act(self.fc(x))


class Metanet_mlp(nn.Module):
    def __init__(self, hidden_layers=1, hidden_size=128):
        super(Metanet_mlp, self).__init__()
        self.layer1 = HiddenLayer(1, hidden_size) # embedding
        self.layer_output = nn.Linear(hidden_size, 1)
        self.layer_mid = nn.Sequential(
            *[HiddenLayer(hidden_size, hidden_size) for _ in range(hidden_layers - 1)]
            )
    
    def forward(self, x):
        x = self.layer1(x)
        # x = self.layer_mid(x)
        x = self.layer_output(x)
        return torch.sigmoid(x)

class Metanet_label(nn.Module):
    def __init__(self, class_num, hidden_size = 128) -> None:
        super(Metanet_label, self).__init__()
        self.label_emb = nn.Linear(class_num, hidden_size//2)
        self.layer1 = nn.Linear(1, hidden_size//2)
        self.layer2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.class_num = class_num
    
    def forward(self, x, y):
        y = F.one_hot(y, num_classes=self.class_num).float()
        x = self.layer1(x)
        y = self.label_emb(y)
        x = torch.concat([x, y], dim=1)
        # x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return torch.sigmoid(x)
    

class Metanet_feature(nn.Module):
    def __init__(self, feature_num, hidden_size = 128) -> None:
        super(Metanet_feature, self).__init__()
        self.loss_emb = nn.Linear(1, hidden_size*3//4)
        self.feature_emb = nn.Linear(feature_num, hidden_size//4)
        self.layer2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x, y):
        x = self.loss_emb(x)
        y = self.feature_emb(y)
        x = torch.concat([x,y], dim=1)
        # x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return 2 * torch.sigmoid(x)

class Metanet_multiplex(nn.Module):
    def __init__(self, class_num, hidden_layers=0, hidden_size = 96) -> None:
        super(Metanet_multiplex, self).__init__()
        
        architecture = {
        '96':[32,32,16,16], # 96
        '160':[64,32,32,32], # 160
        '80':[32, 32, 16 ], # 80
        '144':[64, 64, 16 ], # 144
        }
        net = architecture[str(hidden_size)]
        hidden_size += 32
        self.label_emb = nn.Linear(class_num, net[0])
        self.loss_emb = nn.Linear(1, net[1])
        self.loss_diff_emb = nn.Linear(1, net[2])
        if len(net) == 4:
            self.epoch_emb = nn.Linear(1, net[3])
        else:
            self.epoch_emb = None
        self.class_num = class_num

        self.lstm = nn.LSTM(
            input_size = 1,
            hidden_size = 16,
            batch_first = True,
            bidirectional = True,
            num_layers = 1
        )

        self.layer_output = nn.Linear(hidden_size, 1)
        if hidden_size != 0:
            self.layer_mid = nn.Sequential(
                *[HiddenLayer(hidden_size, hidden_size) for _ in range(hidden_layers - 1)]
                )
            self.layer_output = nn.Sequential(
                self.layer_mid,
                self.layer_output
            )

        self.relu = nn.ReLU()
        # self.act = nn.CELU(0.075)
        self.act = nn.ELU()
    
    def forward(self, x, y, loss_p, epoch_p):
        loss_diff = x - np.percentile(x.detach().cpu().numpy(), loss_p)
        if self.epoch_emb is not None:
            epoch_p = torch.tensor(epoch_p).repeat(x.shape[0],1).cuda().float()
            epoch_p = self.epoch_emb(epoch_p)
        lstm_inputs = torch.permute(x , (1,0)).unsqueeze(2)
        lstm_outputs,(hn, cn) = self.lstm(lstm_inputs)
        loss_variance = lstm_outputs.squeeze(0)
        x = self.loss_emb(x)
        y = F.one_hot(y.long(), num_classes=self.class_num).float()
        y = self.label_emb(y)
        loss_diff = self.loss_diff_emb(loss_diff)

        if self.epoch_emb is not None:
            x = torch.concat([x, y, loss_diff, epoch_p,loss_variance], dim=1)
        else:
            x = torch.concat([x, y, loss_diff,loss_variance], dim=1)
        x = self.layer_output(x)
        
        return torch.sigmoid(x)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, option = 'B'))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, nc=1):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(nc, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 7)
        out = out.view(-1, self.nChannels)
        return self.fc(out)