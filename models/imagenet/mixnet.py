import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

__all__ = ['MixNet', 'mixnet105', 'mixnet121', 'mixnet141']


def mixnet105(**kwargs):
    model = MixNet(num_init_features=64, k1=32, k2=32, block_config=(6, 12, 20, 12), **kwargs)
    return model

def mixnet121(**kwargs):
    model = MixNet(num_init_features=80, k1=40, k2=40, block_config=(6, 12, 24, 16), **kwargs)
    return model

def mixnet141(**kwargs):
    model = MixNet(num_init_features=96, k1=48, k2=48, block_config=(6, 12, 30, 20), **kwargs)
    return model

class _MixLayer(nn.Sequential):
    def __init__(self, num_input_features, expansion, k1, k2, drop_rate):
        super(_MixLayer, self).__init__()
        if k1 > 0:
            self.bn1_1   = nn.BatchNorm2d(num_input_features)
            self.conv1_1 = nn.Conv2d(num_input_features, expansion * k1, kernel_size = 1, stride=1, bias = False)
            self.bn1_2   = nn.BatchNorm2d(expansion * k1)
            self.conv1_2 = nn.Conv2d(expansion * k1, k1, kernel_size = 3, stride=1, padding = 1, bias = False)

        if k2 > 0:
            self.bn2_1   = nn.BatchNorm2d(num_input_features)
            self.conv2_1 = nn.Conv2d(num_input_features, expansion * k2, kernel_size = 1, stride=1, bias = False)
            self.bn2_2   = nn.BatchNorm2d(expansion * k2)
            self.conv2_2 = nn.Conv2d(expansion * k2, k2, kernel_size = 3, stride=1, padding = 1, bias = False)
        
        self.drop_rate = drop_rate
        self.relu = nn.ReLU(inplace=True)
        self.k1 = k1
        self.k2 = k2

    def forward(self, x):
        if self.k1 > 0:
            inner_link = self.bn1_1(x)
            inner_link = self.relu(inner_link)
            inner_link = self.conv1_1(inner_link)
            inner_link = self.bn1_2(inner_link)
            inner_link = self.relu(inner_link)
            inner_link = self.conv1_2(inner_link)

        if self.k2 > 0:
            outer_link = self.bn2_1(x)
            outer_link = self.relu(outer_link)
            outer_link = self.conv2_1(outer_link)
            outer_link = self.bn2_2(outer_link)
            outer_link = self.relu(outer_link)
            outer_link = self.conv2_2(outer_link)

        if self.drop_rate > 0:
            inner_link = F.dropout(inner_link, p=self.drop_rate, training=self.training)
            outer_link = F.dropout(outer_link, p=self.drop_rate, training=self.training)

        c = x.size(1)
        if self.k1 > 0 and self.k1 < c:
            xl  = x[:, 0: c - self.k1, :, :]
            xr  = x[:, c - self.k1: c, :, :] + inner_link
            x   = torch.cat((xl, xr), 1)
        elif self.k1 == c:
            x   = x + inner_link

        if self.k2 > 0:
            out = torch.cat((x, outer_link), 1)
        else:
            out = x

        return out


class Block(nn.Sequential):
    def __init__(self, num_layers, num_input_features, expansion, k1, k2, drop_rate):
        super(Block, self).__init__()
        for i in range(num_layers):
            layer = _MixLayer(num_input_features + i * k2, expansion, k1, k2, drop_rate)
            self.add_module('mixlayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class MixNet(nn.Module):
    def __init__(self, block_config=(6, 12, 24, 16), num_init_features=64, expansion=4, k1=32, k2=32, drop_rate=0, num_classes=1000):

        super(MixNet, self).__init__()
        print('k1: ', k1, 'k2: ', k2)

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each block
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = Block(num_layers=num_layers, num_input_features=num_features, expansion=expansion, k1=k1, k2=k2, drop_rate=drop_rate)
            self.features.add_module('block%d' % (i + 1), block)
            num_features = num_features + num_layers * k2
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = max(num_features // 2, k1)

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out
