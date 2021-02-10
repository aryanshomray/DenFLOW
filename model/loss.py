import torch
import numpy as np
log2pi = np.log(2 * np.pi)


def calc_prior(output: torch.tensor):
    prior = -0.5 * ((output ** 2) + torch.tensor(log2pi))
    prior = torch.sum(prior)
    return prior


def nll_loss(output):
    log_p, logdet, _ = output
    # B, C, H, W = log_p.shape
    # print(output.shape)
    N = 3 * 128 * 128
    logdet = logdet.mean()
    loss = logdet + log_p + (-torch.log(torch.tensor(256.0)) * N)
    loss /= torch.log(torch.tensor(2.0)) * N
    loss = -loss.mean()

    # prior = calc_prior(output)
    # loss = -(prior + logdet)
    # loss = loss/(N * np.log(2.0))
    return loss
