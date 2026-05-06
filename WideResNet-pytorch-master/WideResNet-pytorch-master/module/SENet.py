import numpy as np
import torch
import torch.nn as nn

# Squeeze and Excitation Networks
class SELayer(nn.Module):
    """
    SELayer通常用在检测和分类的代码中，例如resnet， yolo等
    类的构造函数，申请SELayer时，回执行这个函数
    channel: 输入通道
    reduction：全连接层压缩比率
    """
    def __init__(self, channel=512, reduction=16):
        super(SELayer, self).__init__()

        # average pooling 平均池化 CxHxW-->Cx1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            #将通道数压缩为原来的1/reduction
            nn.ReLU(inplace=True),
            # Excitation激活，两个MLP学习通道间相关性
            nn.Linear(channel // reduction, channel, bias=False),
            # 将通道数恢复为原始大小
            nn.Sigmoid()
            # sigmoid激活函数实现特征重标定
        )

    def forward(self, x):
        b, c, _, _ = x.size() #batch size，channel，_是特征图高度和宽度维度
        y = self.avg_pool(x).view(b, c) #把四维重塑为(b, c)形状，准备输入全连接层
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    # y.expand_as: 将(c, 1, 1)的注意力权重扩展到与输入x相同的形状
    # x * y：对原始特征图的每个通道乘以v对应的权重（通道注意力）