import torch
import torch.nn as nn
from . import utils


class _ActNorm(nn.Module):

    def __init__(self, config):
        super(_ActNorm, self).__init__()
        self.num_features = config["in_channels"]
        self.size = [1, self.num_features, 1, 1]
        self.register_parameter("bias", nn.Parameter(torch.zeros(*self.size)))
        self.register_parameter("scale", nn.Parameter(torch.zeros(*self.size)))
        self._initialized = False

    def check_dims(self, inp: torch.tensor) -> None:
        raise NotImplementedError

    def init_param(self, inp: torch.Tensor):
        self.check_dims(inp)

        assert inp.device == self.bias.device

        with torch.no_grad():
            bias = utils.mean(inp.clone(), dim=[0, 2, 3], keepdim=True) * -1.0
            vars: torch.Tensor = utils.mean((inp.clone() + bias)**2, dim=[0, 2, 3], keepdim=True)
            scale = -torch.log((torch.sqrt(vars) + torch.tensor(1e-6)))
            self.bias.data.copy_(bias.data)
            self.scale.data.copy_(scale.data)

    def forward(self, inp, logdet=None, reverse=False):
        if not self._initialized and not reverse:
            self.init_param(inp)
            self._initialized = True
        self.check_dims(inp)

        if not reverse:
            inp += self.bias
            inp *= torch.exp(self.scale)
        else:
            inp *= torch.exp(-self.scale)
            inp -= self.bias

        _logdet = utils.sum(self.scale) * utils.num_pixels(inp)

        if reverse is True:
            _logdet *= -1.0

        logdet += _logdet
        return inp, logdet


class ActNorm2d(_ActNorm):
    def __init__(self, num_features):
        super(ActNorm2d, self).__init__(num_features)

    def check_dims(self, inp: torch.tensor) -> None:
        assert len(inp.size()) == 4
        assert inp.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(
                self.num_features, inp.size()))

