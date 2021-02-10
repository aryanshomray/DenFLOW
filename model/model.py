from nflows import transforms, distributions, flows
from nflows.transforms.base import Transform
import numpy as np
from nflows.flows.base import Flow


class AffineInjector(Transform):

    def __init__(self, num_channels, condChannels):
        super(AffineInjector, self).__init__()

    def forward(self, inputs, context=None):
        pass

    def inverse(self, inputs, context=None):
        pass


class ConditionalAffineCoupling(Transform):

    def __init__(self, num_channels, condChannels):
        super(ConditionalAffineCoupling, self).__init__()

    def forward(self, inputs, context=None):
        pass

    def inverse(self, inputs, context=None):
        pass


def getTransition(num_channels):
    return transforms.CompositeTransform([
        transforms.ActNorm(num_channels),
        transforms.OneByOneConvolution(num_channels)
    ])


def getConditionalFlowStep(num_channels, condChannels):
    return transforms.CompositeTransform([
        getTransition(num_channels),
        AffineInjector(num_channels, condChannels),
        ConditionalAffineCoupling(num_channels, condChannels)
    ])


def getScaleLevel(num_channels, condChannels, numFlowStep):
    z = [getConditionalFlowStep(num_channels, condChannels)]*numFlowStep
    return transforms.CompositeTransform([
        getTransition(num_channels),
        *z,
    ])


def getTransform(config):
    num_channels = config['num_channels']
    condChannels = config['condChannels']
    numFlowStep = config['numFlowStep']
    num_scales = config['num_scales']
    scale_transform = [getScaleLevel(num_channels, condChannels, numFlowStep)]*num_scales
    return transforms.MultiscaleCompositeTransform(scale_transform)


class DenFlow(Flow):

    def __init__(self, config):
        num_channels = config['num_channels']
        crop_size = config['crop_size']
        self.N = num_channels * crop_size * crop_size

        self.squeeze = transforms.SqueezeTransform()



        transform = transforms.CompositeTransform([
            transforms.OneByOneConvolution(num_channels),
            transforms.OneByOneConvolution(num_channels),
            transforms.OneByOneConvolution(num_channels),
            transforms.OneByOneConvolution(num_channels),
            transforms.OneByOneConvolution(num_channels)
        ])
        distribution = distributions.StandardNormal(shape=[num_channels, crop_size, crop_size])
        super().__init__(transform, distribution)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)