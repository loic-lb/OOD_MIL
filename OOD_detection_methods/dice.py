from typing import Any
import torch
import torch.nn as nn
import numpy as np

class DICE:
    def __init__(self, p):
        self.p = p
        self.mean_act = None
        self.masked_w = None
        self.thresh = None
        self.device="cuda"

    def setup(self, net: nn.Module, train_loader_ID: Any):
        activation_log = []
        net.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader_ID):
                data = data.to(self.device)
                features = net(data, return_features=True)[-1]
                activation_log.append(features.detach().cpu().numpy())
            activation_log = np.concatenate(activation_log, axis=0)
            self.mean_act = activation_log.mean(axis=0)

    def compute_mask(self, w):
        contribution = self.mean_act[None, :] * w
        self.thresh = np.percentile(contribution, self.p)
        mask = torch.Tensor((contribution > self.thresh))
        self.masked_w = (mask * w).to(self.device)
    
    @torch.no_grad()
    def confidence_score(self, net: nn.Module, data: Any):
        fc_weight, fc_bias = net.get_classifier_params()
        fc_bias = torch.Tensor(fc_bias).to(self.device)
        if self.masked_w is None:
            self.compute_mask(fc_weight)
        features = net(data, return_features=True)[-1]
        output = (features[:, None, :] * self.masked_w).sum(-1) + fc_bias
        conf = torch.logsumexp(output, dim=1)
        return conf