import math
import torch
import torch.nn as nn

class ADCD_Net(nn.Module):
    def __init__(self):
        super(ADCD_Net, self).__init__()

# semantic perceptual self-supervision
class SPS(nn.Module):
