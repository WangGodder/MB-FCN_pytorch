import torch
import math
import torchvision.models.resnet as resnet
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math


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


"""
    基于torhcvision的ResNet所检测的分支网络，仅修改分支卷积层的stride并删除后续的FC层，融合多个层的feature map作为输出
"""


class ResNet_Branch(nn.Module):

    def __init__(self, block, layers, connection, stride, num_classes=1000):
        self.in_planes = 64
        super(ResNet_Branch, self).__init__()
        sorted(connection)
        self.connection = connection
        self.stride = stride
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 修改stride 使得深层的感受野大小于浅层大小相同(只修改layer4)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        out_channels = [64, 64 * 4, 128 * 4, 256 * 4, 512 * 4]
        self.out_channels = 0

        # 下采样的pooling
        self.downspamle_maxpools = []  # type: list
        self.downspamle_features = []  # type: list
        for c in connection:
            if pow(2, c) < stride or stride == 32:
                pool = nn.MaxPool2d(kernel_size=3, stride=stride // pow(2, c), padding=1)  # type: nn.MaxPool2d
                if c == 5:
                    pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # type: nn.MaxPool2d
                self.downspamle_maxpools.append(pool)
                self.downspamle_features.append(c)
            else:
                break
            # pool = nn.MaxPool2d(kernel_size=3, stride=pow(2, pow_max - c), padding=1) # type: # nn.MaxPool2d
        # 用于上采样的反卷积层
        self.deconvs = []  # type:list
        self.upsample_features = []  # type: list
        upsample_padding = [(1, 1), (0, 1), (0, 5)]
        for c in connection:
            self.out_channels += out_channels[c-1]
            current_stride = pow(2, c)
            if c == 5:
                current_stride //= 2
            if current_stride > stride:
                self.upsample_features.append(c)
                upsample_time = current_stride // stride
                padding_index = int(math.log(upsample_time, 2) - 1)
                deconv = nn.ConvTranspose2d(out_channels[c - 1], out_channels[c - 1], kernel_size=3, \
                                            stride=upsample_time, padding=upsample_padding[padding_index][0], \
                                            output_padding=upsample_padding[padding_index][1])
                self.deconvs.append(deconv)
        # nn.ConvTranspose2d(2048, 2048, 3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # [stride, 1, 1...(num_blocks-1)]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    """
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)
 """

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)  # type: torch.Tensor

        x2 = self.layer1(x1)  # type: torch.Tensor
        x3 = self.layer2(x2)  # type: torch.Tensor
        x4 = self.layer3(x3)  # type: torch.Tensor
        x5 = self.layer4(x4)  # type: torch.Tensor

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        features = [x1, x2, x3, x4, x5]
        for f in features:
            print(f.size())
        connection_features = []
        i = 0
        j = 0
        for c in self.connection:
            feature = features[c - 1]
            if c in self.downspamle_features:
                feature = self.downspamle_maxpools[i](feature)
                i += 1
            elif c in self.upsample_features:
                feature = self.deconvs[j](feature)
                j += 1
            connection_features.append(feature)
            print(feature.shape)

        return torch.cat(connection_features, 1)


def resnet50_branch(connection, stride, pretrained=True, **kwargs):
    model = ResNet_Branch(Bottleneck, [3, 4, 6, 3], connection, stride, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet50']))
    return model


def resnet101_branch(connection, stride, pretrained=True, **kwargs):
    model = ResNet_Branch(Bottleneck, [3, 4, 23, 3], connection, stride, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet101']))
    return model
