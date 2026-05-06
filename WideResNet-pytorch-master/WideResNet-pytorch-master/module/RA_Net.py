import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class PreActResidualUnit(nn.Module):
    """预激活残差单元（Pre-activation Residual Unit）"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(PreActResidualUnit, self).__init__()

        # 第一个卷积块
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)

        # 第二个卷积块
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

        # 下采样（如果需要）
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride,
                          bias=False)
            )

    def forward(self, x):
        identity = x

        # 预激活：BN -> ReLU -> Conv
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out