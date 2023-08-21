from typing import Any
import torch
import torch.nn as nn

class EBO:
    def __init__(self, temperature):
        self.temperature = temperature

    @torch.no_grad()
    def confidence_score(self, net: nn.Module, data: Any):
        output = net.pred(data)
        conf = self.temperature * torch.logsumexp(output / self.temperature, dim=1)
        return conf