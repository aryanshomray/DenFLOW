import torch
import numpy as np
from .modules import utils
log2pi = np.log(2 * np.pi)


def calc_prior(output: torch.tensor):
    prior = -0.5 * ((output ** 2) + torch.tensor(log2pi))
    prior = torch.sum(prior)
    return prior


def nll_loss(output):
    output, logdet = output
    B, C, H, W = output.shape
    N = C * H * W
    prior = calc_prior(output)
    loss = -(prior + logdet)
    loss = loss/(N * np.log(2.0))
    return loss
