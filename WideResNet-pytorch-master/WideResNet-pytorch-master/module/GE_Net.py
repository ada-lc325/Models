import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional


class GENet(nn.Module):
    """
    GENet: Gather-Excite Attention Module
    channels: 输入通道
    extent：空间压缩比例
    """

    def __init__(self,channels):

        super(GENet, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=15, stride=8, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        B, C, H, W = x.size()

        #Gather
        y=x.avg_pool

        #Excite
        y=y.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)

        gate=self.sigmoid(y)

        out = x*gate + x
        return out
