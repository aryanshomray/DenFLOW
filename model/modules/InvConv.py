import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import utils


class Invertible1x1Conv(nn.Module):

    def __init__(self, config):
        super(Invertible1x1Conv, self).__init__()
        num_channels = config['in_channels']
        self.shape = [num_channels, num_channels]
        wt_init = np.linalg.qr(np.random.random(self.shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(wt_init)))

    def param_setter(self, inp, reverse):

        num_pix = utils.num_pixels(inp)
        logdet = torch.slogdet(self.weight)[1] * num_pix

        if not reverse:
            wt = self.weight
        else:
            wt = torch.inverse(self.weight.double()).float()

        wt = wt.view(self.shape[0], self.shape[1], 1, 1)

        return wt, logdet

    def forward(self, inp: torch.Tensor, logdet=None, reverse: bool = False):
        isinstance(inp, torch.Tensor)
        isinstance(logdet, torch.Tensor)

        wt, _logdet = self.param_setter(inp, reverse)
        out = F.conv2d(inp, wt)
        if not reverse:
            if logdet is not None:
                logdet += _logdet
        else:
            if logdet is not None:
                logdet -= _logdet
        return out, logdet


