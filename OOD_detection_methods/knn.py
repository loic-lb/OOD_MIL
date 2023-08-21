from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10


class KNN:
    def __init__(self, K):
        self.K = K
        self.activation_log = None
        self.device = "cuda"

    def setup(self, net: nn.Module, train_loader_ID):
        activation_log = []
        net.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader_ID):
                data = data.to(self.device)
                features = net(data, return_features=True)[-1]
                activation_log.append(
                    normalizer(features.data.cpu().numpy()))

        self.activation_log = np.concatenate(activation_log, axis=0)
        self.index = faiss.IndexFlatL2(features.shape[1])
        self.index.add(self.activation_log)

    @torch.no_grad()
    def confidence_score(self, net: nn.Module, data: Any):
        features = net(data, return_features=True)[-1]
        features_normed = normalizer(features.data.cpu().numpy())
        D, _ = self.index.search(
            features_normed,
            self.K,
        )
        kth_dist = -D[:, -1]
        return torch.from_numpy(kth_dist)
