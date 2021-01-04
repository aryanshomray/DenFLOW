import torch
from .InvConv import Invertible1x1Conv
from .ActNorm import ActNorm2d
import torch.nn as nn


class Transition(nn.Module):

    def __init__(self, config):
        super(Transition, self).__init__()

        self.actNorm = ActNorm2d(config)
        self.InvConv = Invertible1x1Conv(config)

    def forward(self, inp, logdet=None, reverse=False):
        if not reverse:
            # ActNorm Layer
            output, logdet = self.actNorm(inp, logdet, reverse)

            # Invertible Conv Layer (Permutation)
            output, logdet = self.InvConv(output, logdet, reverse)

        else:
            # Invertible Conv Layer (Permutation)
            output, logdet = self.InvConv(inp, logdet, reverse)

            # ActNorm Layer
            output, logdet = self.actNorm(output, logdet, reverse)

        return output, logdet
