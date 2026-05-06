import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torchvision import models

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, args, in_ch, out_ch):
        super(UNetPlusPlus, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.args = args

        no_filters = [32, 64, 128, 256, 512]

        self.conv0_0 = DoubleConv(in_ch, no_filters[0])
        self.conv1_0 = DoubleConv(no_filters[0], no_filters[1])
        self.conv2_0 = DoubleConv(no_filters[1], no_filters[2])
        self.conv3_0 = DoubleConv(no_filters[2], no_filters[3])
        self.conv4_0 = DoubleConv(no_filters[3], no_filters[4])

        self.conv0_1 = DoubleConv(no_filters[0] + no_filters[1], no_filters[0])
        self.conv1_1 = DoubleConv(no_filters[1] + no_filters[2], no_filters[1])
        self.conv2_1 = DoubleConv(no_filters[2] + no_filters[3], no_filters[2])
        self.conv3_1 = DoubleConv(no_filters[3] + no_filters[4], no_filters[3])

        self.conv0_2 = DoubleConv(no_filters[0]*2 + no_filters[1] + no_filters[2], no_filters[0])
        self.conv1_2 = DoubleConv(no_filters[1]*2 + no_filters[2] + no_filters[3], no_filters[1])
        self.conv2_2 = DoubleConv(no_filters[2]*2 + no_filters[3] + no_filters[4], no_filters[2])
 
        self.conv0_3 = DoubleConv(no_filters[0]*3 + no_filters[1] + no_filters[2] + no_filters[3], no_filters[0])
        self.conv1_3 = DoubleConv(no_filters[1]*3 + no_filters[2] + no_filters[3] + no_filters[4], no_filters[1])
 
        self.conv0_4 = DoubleConv(no_filters[0]*4 + no_filters[1] + no_filters[2] + no_filters[3], no_filters[0])
    
        self.sigmoid = nn.Sigmoid()
        # 避免参数共享，每个输出层都需要单独定义
        if self.args.deepsupervision:
            self.final1 = nn.Conv2d(no_filters[0], out_ch, kernel_size=1)
            self.final2 = nn.Conv2d(no_filters[0], out_ch, kernel_size=1)
            self.final3 = nn.Conv2d(no_filters[0], out_ch, kernel_size=1)
            self.final4 = nn.Conv2d(no_filters[0], out_ch, kernel_size=1)
        else:
            self.final = nn.Conv2d(no_filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.args.deepsupervision:
            output1 = self.final1(x0_1)
            output1 = self.sigmoid(output1)
            output2 = self.final2(x0_2)
            output2 = self.sigmoid(output2)
            output3 = self.final3(x0_3)
            output3 = self.sigmoid(output3)
            output4 = self.final4(x0_4)
            output4 = self.sigmoid(output4)
            return [output1, output2, output3, output4]      
        else:
            output = self.final(x0_4)
            output = self.sigmoid(output)
            return output