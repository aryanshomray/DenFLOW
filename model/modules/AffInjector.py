import torch
import torch.nn as nn
from . import utils


def get_logdet(inp):
    return utils.sum(torch.log(inp))


class AffineInjector(nn.Module):

    def __init__(self, config):
        super(AffineInjector, self).__init__()
        self.eps = 1e-6
        self.net = utils.get_net(config['condEncoderOutChannels'], config['in_channels'] * 2, 64, 10)

    def forward(self, inp, logdet=None, reverse=False, img_ft=None):
        img_ft = self.net(img_ft)
        scale, shift = utils.split(img_ft, "cross")
        scale = torch.sigmoid(scale + 2.0) + self.eps
        if not reverse:
            inp += shift
            inp *= scale
            if logdet is not None:
                logdet += get_logdet(scale)
        else:
            inp /= scale
            inp -= scale
            if logdet is not None:
                logdet -= get_logdet(scale)
        output = inp
        return output, logdet
