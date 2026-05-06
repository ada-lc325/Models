from SENet import SELayer
from FCANet import FcaBottleneck
import AlexNet
import torch.nn as nn

class SEFCANet(nn.Module):
    def __init__(self, inplanes=32, planes=8):
        super(SEFCANet, self).__init__()

        self.SELayer = SELayer(channel=512, reduction=16)
        self.FcaBottleneck = FcaBottleneck(inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 reduction=16)

def forward(self, x):
    x1 = self.SELayer(x)
    x2 = self.FcaBottleneck(x)
    return x1 + x2
