#################################################################################################################
# @author code: Gautam Sharma                                                                                   #
#                                                                                                               #
# Original research work by : Alex Krizhevsky et al.                                                            #
# Title : ImageNet Classification with Deep Convolutional Neural Networks                                       #
# url : https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf   #
# Official pytorch implementation : https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py #
# Official pytorch model download url : https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth            #
#                                                                                                               #
#################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
@pytorch calculates the output dimensions using the following formula
url : https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
dim_out = [{H_in +  2*padding - dilation*(kernel_size-1) - 1}/stride + 1]

"""
class Alexnet(nn.module):
    def __int__(self, num_classes = 1000, default_bias=True):
        """

        :param num_classes: Original paper uses 1000 output classes. Can be changed to adopt different architecture.
        :param default_bias: Original paper initializes weights in each layer from a zero-mean Gaussian distribution \
        with standard deviation = 0.01. The biases in the second, fourth, and fifth convolutional layers as well as in \
        the fully-connected layers are initialized with constant 1.
        :return:
        """

        super(Alexnet, self).__init__()
        # convolutional layer (sees 32x32x3 image tensor)
        # input = 224x224x3 image
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.convolution = nn.Sequential(
            nn.Conv2d(3,96,kernel_size=11,stride=4),
            nn.Relu(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.Relu(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.Relu(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

        )

        self.fully_connected = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

        if default_bias:
            self.bias()

    def bias(self):
        """

        :return: Initializes the weights and biases as stated in the paper
        """
        for layer in self.net:
            if type(layer) == nn.Conv2d:
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """

        x = self.convolution(x)
        x = x.view(-1, 256*6*6)
        x = self.fully_connected(x)
        return x
