from model import resnet50_branch
from model import FCN8s
from torch.autograd import Variable
from torchsummary import summary
from dot import make_dot
import torch
import sys


def view_fcn8s():
    model = FCN8s()
    x = Variable(torch.randn(1, 3, 224, 224))
    y = model(x)
    g = make_dot(y)
    g.view(filename="FCN8S")


def print_fcn8s(filename=''):
    model = FCN8s()
    summary(model.cuda(), (3, 224, 224))
    if filename != '':
        origin = sys.stdout
        sys.stdout = open(filename, 'w')
        summary(model.cuda(), (3, 224, 224))
        sys.stdout = origin


def view_resnet50_branch(connection):
    model = resnet50_branch(connection, pretrained=False)
    x = Variable(torch.randn(1, 3, 224, 224))
    y = model(x)
    g = make_dot(y)
    g.view(filename="resnet50_branch_with_connection_" + str(connection))


def print_resnet50_branch(connection, filename=''):
    model = resnet50_branch(connection, pretrained=False)
    summary(model.cuda(), (3, 224, 224))
    if filename != '':
        origin = sys.stdout
        sys.stdout = open(filename, 'w')
        summary(model.cuda(), (3, 224, 224))
        sys.stdout = origin


if __name__ == '__main__':
   # print_fcn8s("FCN8s.log")
    print_resnet50_branch([2,3,4,5])