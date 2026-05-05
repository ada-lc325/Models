import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

# Fixed Sobel kernels defined once at module level (never recreated)
_SOBEL_DEFS = [
    [[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]],   # horizontal
    [[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]],   # vertical
    [[ 2,  1,  0], [ 1,  0, -1], [ 0, -1, -2]],   # diagonal ↘
    [[ 0, -1, -2], [ 1,  0, -1], [ 2,  1,  0]],   # diagonal ↙
]


class edge_for_loss(nn.Module):
    """Applies 4 fixed Sobel kernels via F.conv2d — no nn.Conv2d instantiation per call."""

    def __init__(self, device):
        super(edge_for_loss, self).__init__()
        # Register as non-trainable buffers: moved to correct device automatically,
        # never tracked by autograd → no memory leak
        for i, k in enumerate(_SOBEL_DEFS):
            self.register_buffer(
                f'k{i}',
                torch.tensor(k, dtype=torch.float32).view(1, 1, 3, 3)
            )

    def _apply_kernel(self, x, kernel):
        C = x.shape[1]
        # Depthwise conv: same fixed filter per channel, no gradients through weight
        w = kernel.to(dtype=x.dtype).expand(C, 1, 3, 3).contiguous()
        return torch.abs(F.conv2d(x, w, padding=1, groups=C))

    def forward(self, x):
        e0 = self._apply_kernel(x, self.k0)
        e1 = self._apply_kernel(x, self.k1)
        e2 = self._apply_kernel(x, self.k2)
        e3 = self._apply_kernel(x, self.k3)
        return e0 + e1 + e2 + e3, e0, e1, e2, e3


# ── Kept for compatibility with gt_edge / img_edge (not used in training) ────

def Gedge_map(im, device):
    """Standalone version — uses F.conv2d, no nn.Conv2d allocation."""
    edges = []
    for k in _SOBEL_DEFS:
        C = im.shape[1]
        w = torch.tensor(k, dtype=im.dtype, device=device).view(1, 1, 3, 3)
        w = w.expand(C, 1, 3, 3).contiguous()
        edges.append(torch.abs(F.conv2d(im, w, padding=1, groups=C)))
    sobel_out = edges[0] + edges[1] + edges[2] + edges[3]
    return sobel_out, edges[0], edges[1], edges[2], edges[3]


class gt_edge(nn.Module):
    def __init__(self, device):
        super(gt_edge, self).__init__()
        self.sig = nn.Sigmoid()
        self.dev = device

    def forward(self, x):
        result, _, _, _, _ = self.sig(Gedge_map(x, device=self.dev))
        result[result <= 0.5] = 0
        result[result > 0.5] = 1
        return result


class img_edge(nn.Module):
    def __init__(self, device):
        super(img_edge, self).__init__()
        self.dev = device

    def forward(self, x):
        result, _, _, _, _ = Gedge_map(x, self.dev)
        return result


if __name__ == '__main__':
    import cv2
    from torchvision.transforms import ToTensor
    from torchvision.utils import save_image
    img = cv2.imread('data/train/img/Image_01L.jpg', 0)
    img = cv2.resize(img, (512, 512))
    img = ToTensor()(img)
    img = torch.unsqueeze(img, 0)
    all_e, e1, e2, e3, e4 = edge_for_loss(device='cpu')(img)
    save_image(all_e, 'new1.png')
