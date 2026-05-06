import math
import torch
import torch.nn as nn

#Efficient Channel Attention模块

class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        c1：channel of feature map
        input features with shape (N, C, H, W)
        gamma, b: parameters of mapping functions
    """

    def __init__(self, c1, gamma=2, b=1):
        super().__init__()

        #为了得到卷积核的尺寸，k是由通道数决定的，来判断特征图的映射范围
        k = int(abs((math.log2(c1) + b)/gamma)) #c1: 64  k=3
        k = k if k%2==0 else k+1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k, padding=(k-1)//2, bias=False)

        self.sigmoid = nn.Sigmoid()

def forward(self, x):
    # original size of x= [b, c, h, w]
    y = self.avg_pool(x) #after average pooling, [b,c,1,1]
    #去掉最后一个维度，把倒数第一和第二个维度交换
    y = y.squeeze(-1).transpose(-1,-2) #[b,1,c]
    y = self.conv(y) #[b,1,c]
    y = y.transpose(-1,-2).unsqueeze(-1) #还原并加维度
    y = self.sigmoid(y) #[b,c,1,1]
    return x*y.expand_as(x) #把y的维度扩展到x，然后和x进行点乘

if __name__ == '__main__':
    model = ECA(3)
    input_tensor = torch.randn(4,3,224,224)
    result = model(input_tensor)

    print(input_tensor)
    print(result)