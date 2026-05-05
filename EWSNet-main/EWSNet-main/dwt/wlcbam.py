from torch import nn
from .dwt_module import *
from .dwtlayer import *
from .cbam import CBAM

class wa_module(nn.Module):
    def __init__(self, device, channel_in, wavename='haar'):
        super(wa_module, self).__init__()
        self.dev = device
        self.dwt = DWT_2D(wavename=wavename, device=self.dev)
        self.softmax = nn.Softmax2d()
        self.cbam = CBAM(channel_in=channel_in)

    @staticmethod
    def get_module_name():
        return "wa"

    def forward(self, input):
        assert len(input.shape) == 4
        LL, LH, HL, HH = self.dwt(input)
        output = LL
        LL = self.cbam(LL)
        x_high = self.softmax(torch.add(LH, HL))
        AttMap = torch.mul(output, x_high)
        output = torch.add(output, AttMap)
        return output, LL