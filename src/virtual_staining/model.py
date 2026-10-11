"""A compact single-input, single-output U-Net for fluorescence regression."""

import torch
from torch import nn
from torch.nn import functional as F


def _block(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU())


class UNet(nn.Module):
    def __init__(self, width: int = 16):
        super().__init__()
        if width < 1:
            raise ValueError("width must be positive")
        self.encoder1 = _block(1, width)
        self.encoder2 = _block(width, width * 2)
        self.encoder3 = _block(width * 2, width * 4)
        self.bottom = _block(width * 4, width * 8)
        self.decoder3 = _block(width * 12, width * 4)
        self.decoder2 = _block(width * 6, width * 2)
        self.decoder1 = _block(width * 3, width)
        self.output = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 1 or min(x.shape[-2:]) < 8:
            raise ValueError("expected NCHW with one channel and spatial dimensions at least eight")
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        x = self.bottom(F.max_pool2d(e3, 2))
        for skip, decoder in ((e3, self.decoder3), (e2, self.decoder2), (e1, self.decoder1)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((x, skip), dim=1))
        return self.output(x)
