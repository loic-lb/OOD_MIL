import torch
import numpy as np
import random
import sklearn.metrics as metrics


def apply_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def measure_perf(losses, ground_truths, predicted_classes, probas, n_classes):
    loss = np.mean(losses)
    bac = metrics.balanced_accuracy_score(ground_truths, predicted_classes)
    f1 = metrics.f1_score(ground_truths, predicted_classes, average="weighted")
    if n_classes == 2:
        auc = metrics.roc_auc_score(ground_truths, probas[:, 1])
    else:
        auc = metrics.roc_auc_score(ground_truths, probas, multi_class='ovr')
    return loss, bac, f1, auc
