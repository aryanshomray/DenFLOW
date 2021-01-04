import torch
import torch.nn as nn
from .Transition import Transition
from .CondFlowStep import ConditionalFlowStep


class Flow(nn.Module):

    def __init__(self, config):
        super(Flow, self).__init__()
        self.transition = Transition(config)
        self.CondFlowStep = nn.ModuleList(
            [ConditionalFlowStep(config) for _ in range(config['flow_steps'])]
        )

    def forward(self, clean, img_ft, reverse=False):
        logdet = torch.zeros_like(clean[:, 0, 0, 0])

        if not reverse:
            output, logdet = self.transition(clean, logdet, reverse)
            for i, module in enumerate(self.CondFlowStep):
                output, logdet = module(output, logdet, reverse, img_ft)
        else:
            output = clean
            for i, module in enumerate(reversed(self.CondFlowStep)):
                output, logdet = module(output, logdet, reverse, img_ft)
            output, logdet = self.transition(output, logdet, reverse)

        return output, logdet
