import torch.nn as nn
from .Transition import Transition
from .AffInjector import AffineInjector
from .CondAffCoup import CondAffineCoupling


class ConditionalFlowStep(nn.Module):

    def __init__(self, config):
        super(ConditionalFlowStep, self).__init__()
        self.transition = Transition(config)
        self.affineInject = AffineInjector(config)
        self.condAddCoup = CondAffineCoupling(config)

    def forward(self, inp, logdet=None, reverse=False, img_ft=None):

        if not reverse:
            # Actnorm and 1x1 Conv
            output, logdet = self.transition(inp, logdet, reverse)

            # Affine Injector
            output, logdet = self.affineInject(output, logdet, reverse, img_ft)

            # Conditional Affine Coupling
            output, logdet = self.condAddCoup(output, logdet, reverse, img_ft)

        else:
            # Conditional Affine Coupling
            output, logdet = self.condAddCoup(inp, logdet, reverse, img_ft)

            # Affine Injector
            output, logdet = self.affineInject(output, logdet, reverse, img_ft)

            # Actnorm and 1x1 Conv
            output, logdet = self.transition(output, logdet, reverse)

        return output, logdet
