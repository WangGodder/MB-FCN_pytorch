# -*- coding: utf-8 -*-
# @Time    : 2018/11/29 21:02
# @Author  : Godder
# @File    : mb_fcn.py
########################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.functions import Detect, PriorBoxLayer
import math
import os


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out, inplace=True)
        return out


class MB_FCN(nn.Module):
    def __init__(self, phase, num_class, block, num_blocks, size, connections, strides):
        super(MB_FCN, self).__init__()
        self.phase = phase
        self.in_planes = 64
        self.num_class = num_class
        self.size = size
        self.priorboxs = PriorBoxLayer(size, size, stride=strides)
        self.priors = None
        self.connections = connections
        self.strides = strides

        # Resnet network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 修改stride 使得深层的感受野大小于浅层大小相同(只修改layer4)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1)
        out_channels = [64, 64 * 4, 128 * 4, 256 * 4, 512 * 4]

        # 下采样的pooling
        self.downspamle_maxpools = list()  # type: list
        self.downspamle_features = list()  # type: list
        # 用于上采样的反卷积层
        self.deconvs = list()  # type:list
        self.upsample_features = list()  # type: list
        upsample_padding = [(1, 1), (0, 1), (0, 5)]     # 用于转置卷积的padding参数，对应放大2倍，4倍，8倍
        self.out_channels = []
        for stride, connection in zip(strides, connections):
            pools = list()
            deconv = list()
            down_features = list()
            up_features = list()
            channels = 0
            for c in connection:
                channels += out_channels[c-1]
                current_stride = pow(2, c)
                if c == 5:
                    current_stride //= 2
                if pow(2, c) < stride:
                    pool = nn.MaxPool2d(kernel_size=3, stride=stride // pow(2, c), padding=1)  # type: nn.MaxPool2d
                    if c == 5:
                        pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # type: nn.MaxPool2d
                    pools.append(pool)
                    down_features.append(c)
                elif current_stride > stride:
                    upsample_time = current_stride // stride
                    padding_index = int(math.log(upsample_time, 2) - 1)
                    convT = nn.ConvTranspose2d(out_channels[c - 1], out_channels[c - 1], kernel_size=3, \
                                                stride=upsample_time, padding=upsample_padding[padding_index][0], \
                                                output_padding=upsample_padding[padding_index][1])
                    deconv.append(convT)
                    up_features.append(c)
            self.downspamle_maxpools.append(pools)
            self.downspamle_features.append(down_features)
            self.deconvs.append(deconv)
            self.upsample_features.append(up_features)
            self.out_channels.append(channels)

        # 用于提取位置信息和人脸信息的单层卷积序列
        loc = []
        conf = []
        for channels in self.out_channels:
            loc.append(nn.Conv2d(channels, 4, kernel_size=3, stride=1, padding=1))
            conf.append(nn.Conv2d(channels, 4, kernel_size=3, stride=1, padding=1))
        self.loc = nn.ModuleList(loc)
        self.conf = nn.ModuleList(conf)

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_class, 0, 750, 0.05, 0.3)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # [stride, 1, 1...(num_blocks-1)]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        source = list()         # 用来存储不同branch的特征
        loc = list()
        conf = list()

        # Resnet 提取特征
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)  # type: torch.Tensor

        x2 = self.layer1(x1)  # type: torch.Tensor
        x3 = self.layer2(x2)  # type: torch.Tensor
        x4 = self.layer3(x3)  # type: torch.Tensor
        x5 = self.layer4(x4)  # type: torch.Tensor
        features = [x1, x2, x3, x4, x5]

        # 各branch特征提取，合并放入source中
        for down_pools, down_features, deconv, up_features, connection in\
                zip(self.downspamle_maxpools, self.downspamle_features, self.deconvs, self.upsample_features, self.connections):
            connection_features = []
            i = 0
            j = 0
            for c in connection:
                feature = features[c - 1]
                if c in down_features:
                    feature = down_pools[i](feature)
                    i += 1
                elif c in up_features:
                    feature = deconv[j](feature)
                    j += 1
                connection_features.append(feature)
            source.append(torch.cat(connection_features, 1))

        # prior box提取
        prior_box = []
        for idx, f_layer in enumerate(source):  #type: int, torch.Tensor
            prior_box.append(self.priorboxs.forward(idx, f_layer.shape[3], f_layer.shape[2]))
        with torch.no_grad():
            self.priors = torch.cat([p for p in prior_box], 0)
        print(self.priors.shape, prior_box[0].shape)

        # 对feature map 进行信息提取
        for idx, (x, l , c) in enumerate(zip(source, self.loc, self.conf)):
            if idx == 0:    # 浅层信息提取
                tmp_conf = c(x) # type: torch.Tensor
                a, b, c, pos_conf = tmp_conf.chunk(4, 1)
                neg_conf = torch.cat([a, b, c], 1)  # type: torch.Tensor
                max_conf, _ = neg_conf.max(1)
                max_conf = max_conf.view_as(pos_conf)
                conf.append(torch.cat([max_conf, pos_conf], 1).permute(0, 2, 3, 1).contiguous())
            else:
                tmp_conf = c(x)  # type: torch.Tensor
                neg_conf, a, b, c = tmp_conf.chunk(4, 1)
                pos_conf = torch.cat([a, b, c], 1)  # type: torch.Tensor
                max_conf, _ = pos_conf.max(1)   # type: torch.Tensor, torch.Tensor
                max_conf = max_conf.view_as(neg_conf)
                conf.append(torch.cat([neg_conf, max_conf], 1).permute(0, 2, 3, 1).contiguous())
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())

        # 整理提取信息
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)  # 将每个loc中的tensor转为(batch_size, H*W*4),然后结合
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)  # （batch_size, H*W*2)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1, 2)),                         # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),  # (batch_size, H*W, 4)
                conf.view(conf.size(0), -1, 2),  # (batch_size, H*W, 2)
                self.priors,
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            pretrained_model = torch.load(base_file, map_location=lambda storage, loc: storage)
            model_dict = self.state_dict()
            pretrained_model = {k: v for k, v in pretrained_model.items() if k in model_dict}
            model_dict.update(pretrained_model)
            self.load_state_dict(model_dict)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def build_model(phase, size=640, num_class=2):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 640:
        print("Error: Sorry only 640 is supported currently!")
        return
    return MB_FCN(phase, num_class, Bottleneck, [3, 4, 6, 3], size, [[3, 4, 5], [4, 5]], [8, 16])
