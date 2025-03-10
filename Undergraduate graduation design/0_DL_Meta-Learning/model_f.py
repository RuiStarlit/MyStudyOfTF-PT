import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision


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
    def __init__(self, num_classes=10, block=BasicBlock, num_blocks=[5, 5, 5]):
        super(ResNet32, self).__init__()
        self.in_planes = 16

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
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        f = out
        out = self.linear(out)
        return out, f

class HiddenLayer(nn.Module):
    def __init__(self, input, output):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input, output)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


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
        self.label_emb = nn.Linear(class_num, hidden_size/2)
        self.layer1 = nn.Linear(1, hidden_size/2)
        self.layer2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x, y):
        y = F.one_hot(y)
        x = self.layer1(x)
        y = self.label_emb(y)
        x = torch.concat([x, y], dim=1)
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return torch.sigmoid(x)
    

class Metanet_feature(nn.Module):
    def __init__(self, feature_num, hidden_size = 128) -> None:
        super(Metanet_feature, self).__init__()
        self.loss_emb = nn.Linear(1, hidden_size - feature_num)
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x, y):
        x = self.loss_emb(x)
        x = torch.concat([x,y], dim=1)
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return torch.sigmoid(x)