#################################################################################################################
# @author code: Gautam Sharma                                                                                   #
#                                                                                                               #
# Original research work by : Karen Simonyan∗ & Andrew Zisserman                                                #
# Title : VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION                                    #
# url : https://arxiv.org/pdf/1409.1556.pdf                                                                     #
# Official pytorch implementation : https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py     #
# Official pytorch model download url : https://download.pytorch.org/models/vgg11-bbd30ac9.pth                  #
#                                                                                                               #
#################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


# default weights = True sets the weights as mentioned in the paper
class VGG11(nn.Module):

    def __init__(self, input, num_classes=1000, default_weights = True):
        super(VGG11, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(64,128,kernel_size=3,stride=1),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.Conv2d(256 ,256, kernel_size=3, stride=1),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fully_connected = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(512 * 7 * 7), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

        for i,m in enumerate(self.modules()):
            print(i,m)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward (self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        x = self.convolution(x)
        x = x.view(-1, 512*7*7)
        x = self.fully_connected(x)
        return x

k = VGG11(1)
