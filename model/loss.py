import torch
import numpy as np

log2pi = np.log(2 * np.pi)


def calc_prior(output: torch.tensor):
    prior = -0.5 * ((output ** 2) + torch.tensor(log2pi))
    prior = torch.sum(prior, dim=[1, 2, 3])
    return prior


def nll_loss(output):
    output, logdet = output
    prior = calc_prior(output)
    loss = prior + logdet
    loss = torch.mean(loss)
    return loss
