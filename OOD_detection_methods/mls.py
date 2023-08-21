from typing import Any
import torch
import torch.nn as nn

class MLS:
    def __init__(self):
        pass
    
    @torch.no_grad()
    def confidence_score(self, net: nn.Module, data: Any):
        output = net.pred(data)
        conf = torch.max(output, dim=1)[0]
        return conf