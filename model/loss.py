import torch
import numpy as np
from .modules import utils
log2pi = np.log(2 * np.pi)


def calc_prior(output: torch.tensor):
    prior = -0.5 * ((output ** 2) + torch.tensor(log2pi))
    prior = torch.sum(prior, dim=[1, 2, 3])
    return prior


def nll_loss(output):
    output, logdet = output
    num_pixels = utils.num_pixels(output)
    prior = calc_prior(output)
    loss = prior + logdet
    loss = -torch.sum(loss)
    loss = loss/num_pixels
    return loss
