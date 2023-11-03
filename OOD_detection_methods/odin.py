from typing import Any
import torch
import torch.nn as nn


class ODIN:
    def __init__(self, temperature, noise):
        self.temperature = temperature
        self.noise = noise
        self.input_std = None

    def setup(self, input_std: Any, dataset_name: str, device="cuda"):
        if "mnist" in dataset_name:
            self.input_std = torch.FloatTensor(input_std).view(1, 1, 1, 1).to(device)
        else:
            self.input_std = torch.FloatTensor(input_std).view(1, 3, 1, 1).to(device)

    def confidence_score(self, net: nn.Module, data: Any):
        data.requires_grad = True
        output = net.pred(data)

        criterion = nn.CrossEntropyLoss()

        labels = output.detach().argmax(axis=1)

        output = output / self.temperature
        
        loss = criterion(output, labels)
        loss.backward()

        gradient = torch.ge(data.grad.data.detach(), 0)
        gradient = (gradient.float() - 0.5) * 2

        gradient = gradient / self.input_std 

        pertubated_data = torch.add(data.detach(), gradient, alpha=-self.noise)
        output = net.pred(pertubated_data)
        output = output / self.temperature

        nnOutput = output.detach()
        nnOutput = nnOutput - nnOutput.max(dim=1, keepdim=True).values
        nnOutput = torch.exp(nnOutput) / torch.exp(nnOutput).sum(dim=1, keepdim=True)
        conf = torch.max(nnOutput, dim=1)[0]

        return conf