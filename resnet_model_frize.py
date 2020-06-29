"""
Created on Tue Apr  2 14:21:30 2019

@author: aamir-mustafa
"""

import torch.nn as nn
import torch
import math
from resnet_model import *  # Imports the ResNet Moudle
import argparse


# parser = argparse.ArgumentParser("Softmax frize Training for CIFAR-10 Dataset")
# parser.add_argument('--filename', type=str, default='Models_Softmax/CIFAR10_Softmax.pth.tar') #gpu to be used

# args = parser.parse_args()
# state = {k: v for k, v in args._get_kwargs()}


class Frize_ResNet(nn.Module):

    def __init__(self, depth, num_classes=10, filename='Models_Softmax/CIFAR10_Softmax.pth.tar'):
        super(Frize_ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        model = resnet(num_classes=num_classes, depth=depth)
        model = nn.DataParallel(model).cuda()
        filename = filename
        checkpoint = torch.load(filename)
        # model.load_state_dict(checkpoint['state_dict'])
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        self.BackBone_model = model
        for p in self.parameters():
            p.requires_grad = False

        self.avgpool = nn.AvgPool2d(8)
        self.fcf256_15_1 = nn.Linear(256 * 64, 1024)
        self.fcf256_16_1 = nn.Linear(256 * 64, 1024)
        self.fcf256_17_1 = nn.Linear(256 * 64, 1024)
        self.fcf256_18_1 = nn.Linear(256 * 64, 1024)

        self.fcf256_15 = nn.Linear(1024, num_classes)
        self.fcf256_16 = nn.Linear(1024, num_classes)
        self.fcf256_17 = nn.Linear(1024, num_classes)
        self.fcf256_18 = nn.Linear(1024, num_classes)

    def forward(self, x):
        self.BackBone_model.eval()  # ################
        f128, f256, f1024, y, feature = self.BackBone_model(x)
        f256_1, f256_2, f256_3, f256_4, f256_5, f256_6, f256_7, f256_8, f256_9, f256_10, f256_11, f256_12, \
        f256_13, f256_14, f256_15, f256_16, f256_17, f256_18 = feature

        f256_15 = f256_15.view(f256_15.size(0), -1)
        f256_16 = f256_16.view(f256_16.size(0), -1)
        f256_17 = f256_17.view(f256_17.size(0), -1)
        f256_18 = f256_18.view(f256_18.size(0), -1)

        f256_15_1 = self.fcf256_15_1(f256_15)
        f256_16_1 = self.fcf256_16_1(f256_16)
        f256_17_1 = self.fcf256_17_1(f256_17)
        f256_18_1 = self.fcf256_18_1(f256_18)

        y256_15 = self.fcf256_15(f256_15_1)
        y256_16 = self.fcf256_16(f256_16_1)
        y256_17 = self.fcf256_17(f256_17_1)
        y256_18 = self.fcf256_18(f256_18_1)

        # m  ===  f128  f=feature
        # z  ===  f256  f=feature
        return y256_15, y256_16, y256_17, y256_18, y


def frize_resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return Frize_ResNet(**kwargs)
