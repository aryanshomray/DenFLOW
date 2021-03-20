from nflows import distributions
import numpy as np
from nflows.flows.base import Flow
from denflow import getTransform
from glow import getGLOW


class DenFlow(Flow):

    def __init__(self, config):
        num_channels = config['num_channels']
        crop_size = config['crop_size']
        self.N = num_channels * crop_size * crop_size
        transform = getTransform(config)
        distribution = distributions.StandardNormal(shape=[num_channels * crop_size * crop_size])
        super().__init__(transform, distribution)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class Glow(Flow):

    def __init__(self, config):
        num_channels = config['num_channels']
        crop_size = config['crop_size']
        self.N = num_channels * crop_size * crop_size
        transform = getGLOW(config)
        distribution = distributions.StandardNormal(shape=[num_channels * crop_size * crop_size])
        super().__init__(transform, distribution)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
