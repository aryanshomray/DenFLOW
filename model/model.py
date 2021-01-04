from base import BaseModel
from . import modules
import torch


class DenFlow(BaseModel):

    def __init__(self, config):
        super(DenFlow, self).__init__()
        self.config = config
        if self.config["data"] == "sRGB":
            self.in_channels = 3
        self.condEncoder = modules.Encoder(config)

        self.network = modules.Flow(config)

    def forward(self, inp, conditional, reverse=False):
        img_ft = self.condEncoder(conditional)
        if not reverse:
            output, logdet = self.network(inp, img_ft, reverse)
        else:
            output, logdet = self.network(inp, img_ft, reverse)
        return output, logdet
