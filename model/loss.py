import torch
from math import pi


def calc_prior(output: torch.tensor):
    output = output.view(-1).contiguous()
    prior = torch.matmul(output, output.T)
    prior += torch.log(torch.tensor(2 * pi))
    prior *= -0.5
    print(prior)
    return prior


def nll_loss(output):
    output, logdet = output
    prior = calc_prior(output)
    return prior + logdet
