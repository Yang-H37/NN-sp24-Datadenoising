from pytorch_msssim import SSIM as _SSIM
from torch import nn


class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.ssim = _SSIM(data_range=1, size_average=True, win_size=7, channel=3)

    def forward(self, x, y):
        ssim_loss = 1 - self.ssim(x, y)

        return ssim_loss