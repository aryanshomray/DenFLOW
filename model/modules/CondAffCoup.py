import torch
import torch.nn as nn
from . import utils


def get_logdet(inp):
    return utils.sum(torch.log(inp))


class CondAffineCoupling(nn.Module):

    def __init__(self, config):
        super(CondAffineCoupling, self).__init__()
        self.eps = 1e-4
        self.net = utils.get_net(config['condEncoderOutChannels'] + config['in_channels'] // 2,
                                 2 * (config['in_channels'] - config['in_channels'] // 2), 64, 10)

    def forward(self, inp, logdet=None, reverse=False, img_ft=None):
        z2, z1 = utils.split(inp, "cross")
        z = torch.cat([z1, img_ft], dim=1)
        out = self.net(z)
        scale, shift = utils.split(out, "cross")
        scale = torch.sigmoid(scale + 2.0) + self.eps
        if not reverse:
            z2 += shift
            z2 *= scale
            if logdet is not None:
                logdet += get_logdet(scale)
        else:
            z2 /= scale
            z2 -= shift
            if logdet is not None:
                logdet -= get_logdet(scale)

        z = torch.cat([z1, z2], dim=1)
        output = z
        return output, logdet
