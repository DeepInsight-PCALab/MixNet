import torch
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['mixnet']


from torch.autograd import Variable

class Bottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, k1=12, k2=12, dropRate=0):
        super(Bottleneck, self).__init__()
        # inner link module
        if k1 > 0:
            planes = expansion * k1
            self.bn1_1   = nn.BatchNorm2d(inplanes)
            self.conv1_1 = nn.Conv2d(inplanes, planes, kernel_size = 1, bias = False)
            self.bn1_2   = nn.BatchNorm2d(planes)
            self.conv1_2 = nn.Conv2d(planes, k1, kernel_size = 3, padding = 1, bias = False)

        # outer link module
        if k2 > 0:
            planes = expansion * k2
            self.bn2_1 = nn.BatchNorm2d(inplanes)
            self.conv2_1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn2_2 = nn.BatchNorm2d(planes)
            self.conv2_2 = nn.Conv2d(planes, k2, kernel_size=3, padding=1, bias=False)

        self.dropRate = dropRate
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
        
        if self.dropRate > 0:
            inner_link = F.dropout(inner_link, p=self.dropRate, training=self.training)
            outer_link = F.dropout(outer_link, p=self.dropRate, training=self.training)

        
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

class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(inplanes)
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        out = F.avg_pool2d(out, 2)
        return out

class MixNet(nn.Module):

    def __init__(self, 
        depth=100, 
        unit=Bottleneck, 
        dropRate=0, 
        num_classes=10, 
        k1=12, 
        k2=12, 
        compressionRate=2):
        super(MixNet, self).__init__()

        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6

        self.k2 = k2
        self.k1 = k1
        self.dropRate = dropRate

        self.inplanes = max(k2 * 2, k1)
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        self.block1 = self._make_block(unit, n)
        self.trans1 = self._make_transition(compressionRate)
        self.block2 = self._make_block(unit, n)
        self.trans2 = self._make_transition(compressionRate)
        self.block3 = self._make_block(unit, n)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.inplanes, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_block(self, unit, unit_num):
        layers = []
        for i in range(unit_num):
            # Currently we fix the expansion ratio as the default value
            layers.append(unit(self.inplanes, k1=self.k1, k2=self.k2, dropRate=self.dropRate))
            self.inplanes += self.k2

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = max(int(math.floor(self.inplanes // compressionRate)), self.k1)
        self.inplanes = outplanes
        return Transition(inplanes, outplanes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x) 
        x = self.block3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def mixnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return MixNet(**kwargs)
