# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : Godder

from model.ResNet_Branch import resnet50_branch
import torch
from mb_fcn import build_model
from dot import make_dot


if __name__ == '__main__':
    X = torch.randn(1, 3, 640, 640) # type: torch.Tensor
    # net = resnet50_branch([4,5], 16, False)
    net = build_model("train", size=640)
    # print(net)
    y = net(X)  # type: torch.Tensor
    # g = make_dot(y)
    # g.view(filename="test2")
    print(y.shape)
    # print(net.out_channels)
