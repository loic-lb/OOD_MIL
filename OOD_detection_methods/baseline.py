from typing import Any
import torch
import torch.nn as nn


class Baseline:
    def __init__(self, temperature=1.0):
        self.temperature = temperature

    @torch.no_grad()
    def confidence_score(self, net: nn.Module, data: Any):
        output = net.pred(data)
        output = output / self.temperature
        score = torch.softmax(output, dim=1)
        conf = torch.max(score, dim=1)[0]
        return conf