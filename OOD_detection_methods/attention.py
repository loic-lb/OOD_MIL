from typing import Any
import torch
import torch.nn as nn


class AttentionScore:
    def __init__(self):
        pass

    @torch.no_grad()
    def confidence_score(self, net: nn.Module, data: Any):
        _, attention_weights = net(data)
        conf = torch.max(attention_weights, dim=1)[0] / attention_weights.std(dim=1)
        return conf
