import torch
import torch.nn as nn
import torch.nn.functional as F
from nflows import transforms
import numpy as np
from torchvision.transforms.functional import resize
from nflows.transforms.base import Transform


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, inputs):
        out = F.pad(inputs, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out


class Resent(nn.Module):

    def __init__(self, num_channels):
        super(Resent, self).__init__()
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, 1, 1)

    def forward(self, x):
        inputs = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        output = x + inputs
        return output


class MultiscaleCompositeTransform(Transform):
    """A multiscale composite transform as described in the RealNVP paper.
    Splits the outputs along the given dimension after every transform, outputs one half, and
    passes the other half to further transforms. No splitting is done before the last transform.
    Note: Inputs could be of arbitrary shape, but outputs will always be flattened.
    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    def __init__(self, num_transforms, net, split_dim=1):
        """Constructor.
        Args:
            num_transforms: int, total number of transforms to be added.
            split_dim: dimension along which to split.
        """
        if split_dim < 0:
            raise TypeError("Split dimension must be a positive integer.")

        super().__init__()
        self._transforms = nn.ModuleList()
        self._output_shapes = []
        self._num_transforms = num_transforms
        self._split_dim = split_dim
        self.net = net

    def add_transform(self, transform, transform_output_shape):
        """Add a transform. Must be called exactly `num_transforms` times.
        Parameters:
            transform: the `Transform` object to be added.
            transform_output_shape: tuple, shape of transform's outputs, excl. the first batch
                dimension.
        Returns:
            Input shape for the next transform, or None if adding the last transform.
        """
        assert len(self._transforms) <= self._num_transforms

        if len(self._transforms) == self._num_transforms:
            raise RuntimeError(
                "Adding more than {} transforms is not allowed.".format(
                    self._num_transforms
                )
            )

        if (self._split_dim - 1) >= len(transform_output_shape):
            raise ValueError("No split_dim in output shape")

        if transform_output_shape[self._split_dim - 1] < 2:
            raise ValueError(
                "Size of dimension {} must be at least 2.".format(self._split_dim)
            )

        self._transforms.append(transform)

        if len(self._transforms) != self._num_transforms:  # Unless last transform.
            output_shape = list(transform_output_shape)
            output_shape[self._split_dim - 1] = (
                                                        output_shape[self._split_dim - 1] + 1
                                                ) // 2
            output_shape = tuple(output_shape)

            hidden_shape = list(transform_output_shape)
            hidden_shape[self._split_dim - 1] = hidden_shape[self._split_dim - 1] // 2
            hidden_shape = tuple(hidden_shape)
        else:
            # No splitting for last transform.
            output_shape = transform_output_shape
            hidden_shape = None

        self._output_shapes.append(output_shape)
        return hidden_shape

    def forward(self, inputs, context=None):
        context = self.net(context)
        if self._split_dim >= inputs.dim():
            raise ValueError("No split_dim in inputs.")
        if self._num_transforms != len(self._transforms):
            raise RuntimeError(
                "Expecting exactly {} transform(s) "
                "to be added.".format(self._num_transforms)
            )

        batch_size = inputs.shape[0]

        def cascade():
            hiddens = inputs

            for i, transform in enumerate(self._transforms[:-1]):
                transform_outputs, logabsdet = transform(hiddens, context)
                outputs, hiddens = torch.chunk(
                    transform_outputs, chunks=2, dim=self._split_dim
                )
                assert outputs.shape[1:] == self._output_shapes[i]
                yield outputs, logabsdet

            # Don't do the splitting for the last transform.
            outputs, logabsdet = self._transforms[-1](hiddens, context)
            yield outputs, logabsdet

        all_outputs = []
        total_logabsdet = inputs.new_zeros(batch_size)

        for outputs, logabsdet in cascade():
            all_outputs.append(outputs.reshape(batch_size, -1))
            total_logabsdet += logabsdet

        all_outputs = torch.cat(all_outputs, dim=-1)
        return all_outputs, total_logabsdet

    def inverse(self, inputs, context=None):
        context = self.net(context)
        if inputs.dim() != 2:
            raise ValueError("Expecting NxD inputs")
        if self._num_transforms != len(self._transforms):
            raise RuntimeError(
                "Expecting exactly {} transform(s) "
                "to be added.".format(self._num_transforms)
            )

        batch_size = inputs.shape[0]

        rev_inv_transforms = [transform.inverse for transform in self._transforms[::-1]]

        split_indices = np.cumsum([np.prod(shape) for shape in self._output_shapes])
        split_indices = np.insert(split_indices, 0, 0)

        split_inputs = []
        for i in range(len(self._output_shapes)):
            flat_input = inputs[:, split_indices[i]: split_indices[i + 1]]
            split_inputs.append(flat_input.view(-1, *self._output_shapes[i]))
        rev_split_inputs = split_inputs[::-1]

        total_logabsdet = inputs.new_zeros(batch_size)

        # We don't do the splitting for the last (here first) transform.
        hiddens, logabsdet = rev_inv_transforms[0](rev_split_inputs[0], context)
        total_logabsdet += logabsdet

        for inv_transform, input_chunk in zip(
                rev_inv_transforms[1:], rev_split_inputs[1:]
        ):
            tmp_concat_inputs = torch.cat([input_chunk, hiddens], dim=self._split_dim)
            hiddens, logabsdet = inv_transform(tmp_concat_inputs, context)
            total_logabsdet += logabsdet

        outputs = hiddens

        return outputs, total_logabsdet


def getResnet(in_channels, out_channels, hidden_channels, num_blocks):
    net = [nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)]
    for _ in range(num_blocks):
        net.append(Resent(hidden_channels))
    net.append(nn.Conv2d(hidden_channels, out_channels, 3, 1, 1))
    return nn.Sequential(*net)


class AffineInjector(Transform):

    def __init__(self, num_channels, condChannels, netHiddenChannels, numResBlocks):
        super(AffineInjector, self).__init__()
        self.num_channels = num_channels
        self.condChannels = condChannels
        self.net = getResnet(condChannels, num_channels * 2, netHiddenChannels, numResBlocks)

    def forward(self, inputs, context=None):
        if context is None:
            raise Exception("Context should not be None.")
        B, C, H, W = inputs.shape
        context = resize(context, [H, W])
        context = self.net(context)
        scale, shift = torch.chunk(context, 2, 1)
        scale = torch.exp(scale)
        output = scale * inputs + shift
        logabsdet = torch.sum(torch.log(scale), dim=[1, 2, 3])
        return output, logabsdet

    def inverse(self, inputs, context=None):
        if context is None:
            raise Exception("Context should not be None.")
        B, C, H, W = inputs.shape
        context = resize(context, [H, W])
        context = self.net(context)
        scale, shift = torch.chunk(context, 2, 1)
        scale = torch.exp(-scale)
        output = scale * (inputs - shift)
        logabsdet = torch.sum(torch.log(scale), dim=[1, 2, 3])
        return output, logabsdet


class ConditionalAffineCoupling(Transform):

    def __init__(self, num_channels, condChannels, netHiddenChannels, numResBlocks, reverse=False):
        super(ConditionalAffineCoupling, self).__init__()
        self.num_channels = num_channels
        self.CondChannels = condChannels
        self.net = getResnet(self.CondChannels + self.num_channels // 2, self.num_channels, netHiddenChannels,
                             numResBlocks)
        self.reverse = reverse

    def forward(self, inputs, context=None):
        if context is None:
            raise Exception("Context should not be None.")
        inputs_1, inputs_2 = torch.chunk(context, 2, 1)
        if self.reverse:
            inputs_1, inputs_2 = inputs_2, inputs_1
        outputs_1 = inputs_1
        B, C, H, W = inputs.shape
        context = resize(context, [H, W])
        context = self.net(torch.cat([context, inputs_1], dim=1))
        scale, shift = torch.chunk(context, 2, 1)
        scale = torch.exp(scale)
        outputs_2 = scale * inputs_2 + shift
        logabsdet = torch.sum(torch.log(scale), dim=[1, 2, 3])
        outputs = torch.cat([outputs_1, outputs_2], dim=1)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if context is None:
            raise Exception("Context should not be None.")
        inputs_1, inputs_2 = torch.chunk(context, 2, 1)
        if self.reverse:
            inputs_1, inputs_2 = inputs_2, inputs_1
        B, C, H, W = inputs.shape
        context = resize(context, [H, W])
        context = self.net(torch.cat([context, inputs_1], dim=1))
        scale, shift = torch.chunk(context, 2, 1)
        scale = torch.exp(-scale)
        outputs_1 = inputs_1
        outputs_2 = scale * (inputs_2 - shift)
        outputs = torch.cat([outputs_1, outputs_2], dim=1)
        logabsdet = torch.sum(torch.log(scale), dim=[1, 2, 3])
        return outputs, logabsdet


def getTransition(num_channels):
    return transforms.CompositeTransform([
        transforms.ActNorm(num_channels),
        transforms.OneByOneConvolution(num_channels)
    ])


def getConditionalFlowStep(num_channels, condChannels, netHiddenChannels, numResBlocks):
    return transforms.CompositeTransform([
        getTransition(num_channels),
        AffineInjector(num_channels, condChannels, netHiddenChannels, numResBlocks),
        ConditionalAffineCoupling(num_channels, condChannels, netHiddenChannels, numResBlocks)
    ])


def getScaleLevel(num_channels, condChannels, numFlowStep, netHiddenChannels, numResBlocks):
    z = [getConditionalFlowStep(num_channels, condChannels, netHiddenChannels, numResBlocks)
         for _ in range(numFlowStep)]

    return transforms.CompositeTransform([
        transforms.SqueezeTransform(),
        getTransition(num_channels),
        *z
    ])


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, in_channels=3, nf=64, gc=32, condChannels=160, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv = nn.Conv2d(in_channels, nf, 3, 1, 1, bias=bias)
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, condChannels, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x = self.lrelu(self.conv(x))
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        return self.conv5(torch.cat((x, x1, x2, x3, x4), 1))


def getTransform(config):
    num_channels = config['num_channels'] * 4
    condChannels = config['condChannels']
    numFlowStep = config['numFlowStep']
    num_scales = config['num_scales']
    crop_size = config['crop_size'] // 2
    netHiddenChannels = config['netHiddenChannels']
    numResBlocks = config['numResBlocks']
    net = ResidualDenseBlock_5C()
    transform = MultiscaleCompositeTransform(num_scales, net)
    for i in range(num_scales):
        next_input = transform.add_transform(getScaleLevel(num_channels, condChannels, numFlowStep, netHiddenChannels,
                                                           numResBlocks),
                                             [num_channels, crop_size, crop_size])
        num_channels *= 2
        crop_size //= 2
    return transform
