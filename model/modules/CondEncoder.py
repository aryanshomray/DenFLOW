import torch.nn as nn
from .Blocks import RRDB_Block, EDSR_backbone


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.net = config["condEncoder"]

        if self.net == "UNET":
            pass
        elif self.net == "EDSR":
            self.encoder = EDSR_backbone(config)
        elif self.net == "EfficientNet":
            pass
        elif self.net == "RRDB":
            self.encoder = RRDB_Block(config)

    def forward(self, inp):
        if self.net == "UNET":
            pass
        elif self.net == "EDSR":
            return self.encoder(inp)
        elif self.net == "EfficientNet":
            return self.encoder(inp)
        elif self.net == "RRDB":
            return self.encoder(inp)
