import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A


class GatedAttentionMNIST(nn.Module):
    def __init__(self, n_classes):
        super(GatedAttentionMNIST, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1
        self.n_classes = n_classes

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, self.n_classes),
        )

    def forward(self, x, return_features=False):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_logits = self.classifier(M)

        if return_features:
            return Y_logits, A, M
        else:
            return Y_logits, A

    def pred(self, x):
        return self.forward(x)[0]

    def get_classifier_params(self):
        return self.classifier[0].weight.cpu().detach().numpy(), self.classifier[0].bias.cpu().detach().numpy()


class GatedAttentionCIFAR(nn.Module):
    def __init__(self, n_classes, freeze_tile_embedding=True):
        super(GatedAttentionCIFAR, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1
        self.freeze_tile_embedding = freeze_tile_embedding

        if self.freeze_tile_embedding:
            self.feature_extractor_part1 = models.resnet50(pretrained=True)
            self.feature_extractor_part1.fc = nn.Identity()
            self.feature_extractor_part1.requires_grad_(False)

        else:
            self.feature_extractor_part1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
            )

        self.feature_extractor_part2 = nn.Sequential(nn.Linear(2048, self.L),
                                                     nn.ReLU())

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, n_classes),
        )

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        if self.freeze_tile_embedding:
            self.feature_extractor_part1.eval()

    def forward(self, x, return_features=False):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        if not self.freeze_tile_embedding:
            H = H.view(-1, 128 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_logits = self.classifier(M)

        if return_features:
            return Y_logits, A, M
        else:
            return Y_logits, A

    def pred(self, x):
        return self.forward(x)[0]

    def get_classifier_params(self):
        return self.classifier[0].weight.cpu().detach().numpy(), self.classifier[0].bias.cpu().detach().numpy()


class AverageMIL(nn.Module):
    def __init__(self, n_classes, freeze_tile_embedding=True):
        super(AverageMIL, self).__init__()
        self.L = 500

        self.freeze_tile_embedding = freeze_tile_embedding

        if self.freeze_tile_embedding:
            self.feature_extractor_part1 = models.resnet50(pretrained=True)
            self.feature_extractor_part1.fc = nn.Identity()
            self.feature_extractor_part1.requires_grad_(False)

        else:
            self.feature_extractor_part1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
            )

        self.feature_extractor_part2 = nn.Sequential(nn.Linear(2048, self.L),
                                                     nn.ReLU())

        self.classifier = nn.Sequential(nn.Linear(self.L, n_classes))

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        if self.freeze_tile_embedding:
            self.feature_extractor_part1.eval()

    def forward(self, x, return_features=False):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        if not self.freeze_tile_embedding:
            H = H.view(-1, 128 * 4 * 4)
        H = self.feature_extractor_part2(H)
        H = torch.mean(H, dim=0, keepdim=True)
        Y_logits = self.classifier(H)
        if return_features:
            return Y_logits, H
        else:
            return Y_logits

    def pred(self, x):
        return self.forward(x)

    def get_classifier_params(self):
        return self.classifier[0].weight.cpu().detach().numpy(), self.classifier[0].bias.cpu().detach().numpy()


class MaxMIL(nn.Module):
    def __init__(self, n_classes, freeze_tile_embedding=True):
        super(MaxMIL, self).__init__()
        self.L = 500
        self.D = 128

        self.freeze_tile_embedding = freeze_tile_embedding

        if self.freeze_tile_embedding:
            self.feature_extractor_part1 = models.resnet50(pretrained=True)
            self.feature_extractor_part1.fc = nn.Identity()
            self.feature_extractor_part1.requires_grad_(False)

        else:
            self.feature_extractor_part1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
            )

        self.feature_extractor_part2 = nn.Sequential(nn.Linear(2048, self.L),
                                                     nn.ReLU())

        self.classifier = nn.Sequential(nn.Linear(self.L, n_classes))

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        if self.freeze_tile_embedding:
            self.feature_extractor_part1.eval()

    def forward(self, x, return_features=False):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        if not self.freeze_tile_embedding:
            H = H.view(-1, 128 * 4 * 4)
        H = self.feature_extractor_part2(H)
        H = self.classifier(H)
        Y_logits = torch.max(H, dim=0, keepdim=True)[0]
        if return_features:
            return Y_logits, H
        else:
            return Y_logits
