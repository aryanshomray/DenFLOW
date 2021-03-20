import torch
import torch.nn as nn
import torch.nn.functional as F
from nflows import transforms


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


class depthwiseConv(nn.Module):
    def __init__(self, num_features, kernel_size=2):
        super(depthwiseConv, self).__init__()
        self.kernel_size = kernel_size
        self.num_features = num_features

        self.register_buffer("W", nn.Parameter(torch.ones([num_features, 1, kernel_size, kernel_size])))

    def forward(self, x):
        x = F.conv2d(x, self.W, stride=self.kernel_size, groups=self.num_features)
        return x


class alphaConvolution(nn.Module):

    def __init__(self, num_features, crop_size, kernel_size=2):
        super(alphaConvolution, self).__init__()
        self.kernel_size = kernel_size
        self.depthConv = depthwiseConv(num_features, kernel_size)
        self.up = nn.Upsample(scale_factor=kernel_size, mode='nearest')
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        H = W = 64 // (num_features // 3)
        print(num_features, H, W)
        self.register_parameter('shift', nn.Parameter(torch.randn([1, num_features, H, W])))
        self.register_parameter('scale', nn.Parameter(torch.randn([1, num_features, H, W])))

    def forward(self, inputs, context=None):

        inputs = self.shift * inputs + self.scale
        x = inputs
        x = self.depthConv(x)
        x = self.up(x)
        x /= self.kernel_size * self.kernel_size
        output = inputs + x
        logabsdet = torch.slogdet(self.shift * 2)[1].sum().view(1)
        return output, logabsdet

    def inverse(self, inputs, context=None):
        x = inputs
        x = self.depthConv(x)
        x = self.up(x)
        x /= self.kernel_size * self.kernel_size * 2
        x *= -1
        output = x + inputs
        output = (output - self.scale) / self.shift
        logabsdet = -torch.slogdet(self.shift * 2)[1].sum().view(1)
        return output, logabsdet


class Net(nn.Module):

    def __init__(self, in_channel, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(64, out_channels),
        )

    def forward(self, inp, context=None):
        return self.net(inp)


def getGlowStep(num_channels):
    mask = [1] * num_channels
    mask[::2] = [-1] * (len(mask[::2]))

    def getNet(in_channel, out_channels):
        return Net(in_channel, out_channels)

    return transforms.CompositeTransform([
        transforms.ActNorm(num_channels),
        transforms.OneByOneConvolution(num_channels),
        transforms.coupling.AffineCouplingTransform(mask, getNet)
    ])


def getGlowScale(num_channels, num_flow):
    z = [getGlowStep(num_channels) for _ in range(num_flow)]
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
