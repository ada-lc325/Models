import torch
import torch.nn as nn
import torch.nn.functional as F

# CBAM注意力机制，通常用于分类和检测的神经网络，如resnet或者Unet
# 先提取通道的权重提取，再做空间的权重提取
class CBAM(nn.Module):
    # in_channels输入的模块的通道，reduction神经网路压缩率
    def __init__(self, in_channels, reductions=16, kernel_size=7):
        super(CBAM, self).__init__()

        # Dimension 通道的权重提取
        self.channel_avg_pool=nn.AdaptiveAvgPool2d(1) #[b,c,1,1]
        self.channel_max_pool=nn.AdaptiveMaxPool2d(1) #[b,c,1,1]

        self.channel_mlp=nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reductions, kernel_size=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels//reductions, in_channels, kernel_size=1, bias=False),
        )

        # Spatial 空间的权重提取
        self.spatial_conv = nn.Conv2d(in_channels=2,
                                      out_channels=1,
                                      kernel_size=kernel_size,
                                      padding=kernel_size//2,
                                      bias=False)

    def forward(self, x):
        # x [b,c,h,w]
        avg_pool = self.channel_avg_pool(x) #[b,c,1,1]
        max_pool = self.channel_max_pool(x) #[b,c,1,1]

        avg_out = self.channel_mlp(avg_pool) #[b,c,1,1]
        max_out = self.channel_max_pool(max_pool) #[b,c,1,1]

        channel_att = F.sigmoid(avg_out+max_out) #[b,c,1,1]

        x = channel_att * x  #[b,c,h,w]

        max_pool_spatial,_ = torch.max(x, dim=1,keepdim=True) #[b,1,h,w]
        # _, 用来接收函数返回的“我不需要的那一部分
        # 最大空间池化会变成channel=1的特征图
        avg_pool_spatial = torch.mean(x, dim=1,keepdim=True) #[b,1,h,w]

        spatial_input = torch.cat(max_pool_spatial+avg_pool_spatial, dim=1) #[b,2,h,w]
        spatial_out = self.spatial_conv(spatial_input)

        spatial_att = F.sigmoid(spatial_out)
        x = spatial_att * x #[b,c,h,w]

        return x

