from .nflows import transforms, distributions, flows
from .nflows.transforms.coupling import CouplingTransform
from .nflows.transforms.base import Transform
import numpy as np
from .nflows.flows.base import Flow
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out


def getNetwork():
    pass


class AffineInjector(Transform):

    def __init__(self, num_channels, condChannels):
        super(AffineInjector, self).__init__()
        mask = [1] * num_channels
        mask[::2] = -1
        mask = torch.as_tensor(mask)
        if mask.dim() != 1:
            raise ValueError("Mask must be a 1-dim tensor.")
        if mask.numel() <= 0:
            raise ValueError("Mask can't be empty.")

    def _transform_dim_multiplier(self):
        pass

    def _coupling_transform_forward(self, inputs, transform_params):
        pass

    def _coupling_transform_inverse(self, inputs, transform_params):
        pass


class ConditionalAffineCoupling(CouplingTransform):

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


def getScaleLevel(num_channels, condChannels, numFlowStep, squeeze=False):
    z = [getConditionalFlowStep(num_channels, condChannels)] * numFlowStep
    if not squeeze:
        return transforms.CompositeTransform([
            getTransition(num_channels),
            *z,
        ])
    else:
        return transforms.CompositeTransform([
            transforms.SqueezeTransform(),
            getTransition(num_channels),
            *z
        ])


def getTransform(config):
    num_channels = config['num_channels']
    condChannels = config['condChannels']
    numFlowStep = config['numFlowStep']
    num_scales = config['num_scales']
    scale_transform = [getScaleLevel(num_channels * 4, condChannels, numFlowStep)] * (num_scales - 1)
    return transforms.MultiscaleCompositeTransform([
        getScaleLevel(num_channels, condChannels, numFlowStep, squeeze=True),
        *scale_transform
    ])


class DenFlow(Flow):

    def __init__(self, config):
        num_channels = config['num_channels']
        crop_size = config['crop_size']
        self.N = num_channels * crop_size * crop_size

        self.squeeze = transforms.SqueezeTransform()

        transform = getTransform(config)
        distribution = distributions.StandardNormal(shape=[num_channels, crop_size, crop_size])
        super().__init__(transform, distribution)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


def getGlowStep(num_channels):
    mask = [1] * num_channels
    mask[::2] = [-1]*(len(mask[::2]))

    def getNet(in_channel, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(64, out_channels),
        )

    return transforms.CompositeTransform([
        transforms.ActNorm(num_channels),
        transforms.OneByOneConvolution(num_channels),
        transforms.coupling.AffineCouplingTransform(mask, getNet)
    ])


def getGlowScale(num_channels, num_flow):
    z = [getGlowStep(num_channels)] * num_flow
    return transforms.CompositeTransform([
        transforms.SqueezeTransform(),
        *z
    ])


def getGLOW(config):
    # print(config)
    num_channels = config['num_channels'] * 4
    num_flow = config['num_flow']
    num_scale = config['num_scale']
    crop_size = config['crop_size'] // 2
    transform = transforms.MultiscaleCompositeTransform(num_scale)
    for i in range(num_scale):
        next_input = transform.add_transform(getGlowScale(num_channels, num_flow),
                                             [num_channels, crop_size, crop_size])
        num_channels *= 2
        crop_size //= 2

    return transform


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
