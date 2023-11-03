from __future__ import print_function

import os
import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
import itertools
import torch.utils.data as data_utils
import yaml
from functools import partial

from datasets import DatasetBags, DatasetBagsHisto
from model import Attention, GatedAttentionMNIST, GatedAttentionCIFAR, AverageMIL, MaxMIL
from metrics_OOD_detection import compute_all_metrics
from utils import apply_random_seed, seed_worker, measure_perf

from OOD_detection_methods import baseline, odin, ebo, mls, dice, knn

OOD_detection_methods_dict = {"baseline": baseline.Baseline, "odin": odin.ODIN, "ebo": ebo.EBO, "mls": mls.MLS,
                              "dice": dice.DICE, "knn": knn.KNN}


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--params_ID_training', type=str, default=None,
                    help='Parameters config file used to train model on ID data')
parser.add_argument('--dataset_OOD', choices=['fashion-mnist', 'kmnist', 'textures',
                                              'svhn', 'places365', 'iciar'], default='mnist',
                    help='dataset to use: mnist, fashion-mnist or kmnist (default: mnist)')
parser.add_argument('--target_class_id_OOD', type=int, nargs='+', default=[9], metavar='T',
                    help='bags have a positive labels if they contain at least one target_class_id'
                         'for OOD dataset (should be the same size as target_class_id_ID)')
parser.add_argument('--neg_class_id_OOD', type=int, nargs='+', default=None, metavar='N',
                    help='class id for negative instances (default: None, consider all classes except target_class_id)')
parser.add_argument('--ood_detection_method', type=str, default='baseline', choices=['baseline', 'odin', 'ebo', 'mls',
                                                                                     'dice', 'knn'],
                    help='Choose ood detection method')
parser.add_argument('--ood_detection_method_params', type=str, default='./OOD_detection_methods/params.yml',
                    help='Parameters for ood detection method')

args = parser.parse_args()
with open(args.params_ID_training, 'r') as f:
    params_ID_training = yaml.load(f, Loader=yaml.FullLoader)
    args = argparse.Namespace(**params_ID_training, **vars(args))

assert len(args.target_class_id) == len(args.target_class_id_OOD)
save_location = os.path.join(args.experiment, f'{args.dataset}_{args.target_class_id}_ID')
assert os.path.exists(save_location), 'There is no found experiment for the given ID dataset'
args.cuda = not args.no_cuda and torch.cuda.is_available()

apply_random_seed(args.seed)

print('Load Test Set')
g = torch.Generator()
g.manual_seed(args.seed)
loader_kwargs = {'num_workers': 0, 'pin_memory': True, 'generator': g, "worker_init_fn": seed_worker}

dataset_fct_ID = partial(DatasetBagsHisto, test_size=args.test_size_histo) if args.dataset == 'pcam' else DatasetBags

if args.ood_detection_method in ["dice", "knn"]:
    train_loader_ID = data_utils.DataLoader(dataset_fct_ID(
        dataset=args.dataset,
        target_class_id=args.target_class_id,
        neg_class_id=args.neg_class_id,
        mean_bag_length=args.mean_bag_length,
        var_bag_length=args.var_bag_length,
        num_bag=args.num_bags_train,
        seed=args.seed,
        mode="train",
        perf_aug=True),
        batch_size=1,
        shuffle=True,
        **loader_kwargs)
    
test_loader_ID = data_utils.DataLoader(dataset_fct_ID(
        dataset=args.dataset,
        target_class_id=args.target_class_id,
        neg_class_id=args.neg_class_id,
        mean_bag_length=args.mean_bag_length,
        var_bag_length=args.var_bag_length,
        num_bag=args.num_bags_test,
        seed=args.seed,
        mode="test"),
        batch_size=1,
        shuffle=True,
        **loader_kwargs)

resize_size_dict = {"mnist": (28, 28),
                    "cifar10": (32, 32),
                    "pcam": (96, 96)}

dataset_fct_OOD = partial(DatasetBagsHisto, test_size=args.test_size_histo) if args.dataset_OOD == 'iciar'\
    else DatasetBags


test_loader_OOD = data_utils.DataLoader(dataset_fct_OOD(
        dataset=args.dataset_OOD,
        target_class_id=args.target_class_id_OOD,
        neg_class_id=args.neg_class_id_OOD,
        mean_bag_length=args.mean_bag_length,
        var_bag_length=args.var_bag_length,
        num_bag=args.num_bags_test,
        seed=args.seed,
        mode="test",
        resize_size=resize_size_dict[args.dataset]),
        batch_size=1,
        shuffle=True,
        **loader_kwargs)

print('Init Model')
n_classes = len(args.target_class_id) + 1
if args.model == 'attention':
    model = Attention()
elif args.model == 'gated_attention':
    if 'mnist' in args.dataset:
        model = GatedAttentionMNIST(n_classes=n_classes)
    else:
        model = GatedAttentionCIFAR(n_classes=n_classes, freeze_tile_embedding=args.freeze_tile_embedding)
elif args.model == 'average':
    model = AverageMIL(n_classes=n_classes, freeze_tile_embedding=args.freeze_tile_embedding)
elif args.model == 'max':
    model = MaxMIL(n_classes=n_classes, freeze_tile_embedding=args.freeze_tile_embedding)
if args.cuda:
    model.cuda()

loss_function = torch.nn.CrossEntropyLoss()

with open(args.ood_detection_method_params) as file:
    config = yaml.safe_load(file)
    
ood_detection_method = OOD_detection_methods_dict[args.ood_detection_method](**config[args.ood_detection_method])
    

def visualize_test_samples(data, attention_weights, bag_level, conf, batch_idx, mean, std, save_location):
    images = [(data[0][i]).squeeze().detach().cpu() for i in range(data[0].shape[0])]
    if 'mnist' not in args.dataset_OOD:
        mean = torch.FloatTensor(mean).view(3, 1, 1)
        std = torch.FloatTensor(std).view(3, 1, 1)
    else:
        mean = torch.FloatTensor(mean).view(1, 1, 1)
        std = torch.FloatTensor(std).view(1, 1, 1)
    images = [(image * std + mean).permute(1, 2, 0).numpy() for image in images]
    attention_weights = [round(a_w, 3) for a_w in attention_weights[0].detach().cpu().numpy()]
    row_count = 3
    col_count = 8

    fig, axes = plt.subplots(nrows=row_count, ncols=col_count, figsize=(40, 20))
    fig.suptitle(f'True Bag Label {bag_level[0]}, Predicted Bag Label: {bag_level[1]}, '
                 f'Confidence score: {round(conf.item(), 3)}', fontsize=30)
    for idx, (i, j) in enumerate(itertools.product(range(row_count), range(col_count))):
        if idx > len(images) - 1:
            axes[i, j].set_visible(False)
            continue
        axes[i, j].axis("off")
        axes[i, j].text(15, 26, str(attention_weights[idx]), color='red',
                        bbox=dict(facecolor='none', edgecolor='red'), fontsize=28)
        axes[i, j].imshow(images[idx], aspect="auto")

    plt.subplots_adjust(wspace=.05, hspace=.05)
    save_images_locations = os.path.join(save_location, f'images_{args.model}_freeze_tile_embedding_'
                                                        f'{args.freeze_tile_embedding}_ood_detection_method_'
                                                        f'{args.ood_detection_method}')
    if os.path.isdir(save_images_locations) is False:
        os.mkdir(save_images_locations)
    plt.savefig(os.path.join(save_images_locations, f'test_sample_{batch_idx}.png'))


def test(save_location, OOD=False):
    model.eval()
    losses = []
    proba_predictions = []
    ground_truths = []
    attention_weights_list = []
    conf_scores = []
    if OOD:
        test_loader = test_loader_OOD
    else:
        test_loader = test_loader_ID
    for batch_idx, (data, label) in enumerate(test_loader):
        bag_label = label[0].long()
        instance_labels = label[1]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        if "attention" in args.model:
            pred, attention_weights = model(data)
            attention_weights_list.extend(attention_weights.detach().cpu().numpy())
        else:
            pred = model(data)
        loss = loss_function(pred, bag_label)
        pred = torch.softmax(pred, dim=-1)
        conf = ood_detection_method.confidence_score(model, data)

        losses.append(loss.detach().cpu().numpy())
        proba_predictions.extend(pred.detach().cpu().numpy())
        ground_truths.extend(bag_label.detach().cpu().numpy())
        conf_scores.extend(conf.detach().cpu().numpy())

        if batch_idx < 10 and OOD and "attention" in args.model:  # plot bag labels and instance labels for first 10 bags
            _, predicted_label = torch.max(pred, axis=1)
            bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0]))
            instance_level = list(zip(instance_labels.numpy()[0].tolist(),
                                      np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))
            visualize_test_samples(data, attention_weights, bag_level, conf, batch_idx, mean=test_loader.dataset.mean,
                                   std=test_loader.dataset.std, save_location=save_location)
            print('\nTrue Bag Label, Predicted Bag Label: {}\n'
                  'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))

    proba_predictions = np.array(proba_predictions)
    predicted_classes = np.argmax(proba_predictions, axis=1)

    test_loss, test_bac, test_f1, test_auc = measure_perf(losses, ground_truths,
                                                          predicted_classes, proba_predictions, n_classes)

    print(f'test_loss={test_loss:.3f}', f' test_bac={test_bac:.3f}',
          f'    test_f1={test_f1:.3f}', f'    test_auc={test_auc:.3f}')

    conf = np.array(conf_scores)
    pred = predicted_classes
    if OOD:
        label = -1 * np.ones_like(ground_truths)
    else:
        label = ground_truths

    return conf, pred, label


if __name__ == "__main__":
    print('Start Testing')
    model_file_name = f'{args.model}_freeze_tile_embedding_{args.freeze_tile_embedding}.pt'
    model.load_state_dict(torch.load(os.path.join(save_location, model_file_name)))
    if args.ood_detection_method in ["dice", "knn"]:
        ood_detection_method.setup(model, train_loader_ID)
    elif args.ood_detection_method == "odin":
        ood_detection_method.setup(test_loader_ID.dataset.std, dataset_name=args.dataset)
    save_results_locations = os.path.join(save_location, f'{args.dataset_OOD}_OOD')
    if os.path.isdir(save_results_locations) is False:
        os.mkdir(save_results_locations)
    id_conf, id_pred, id_gt = test(save_results_locations)
    ood_conf, ood_pred, ood_gt = test(save_results_locations, OOD=True)
    pred = np.concatenate([id_pred, ood_pred])
    conf = np.concatenate([id_conf, ood_conf])
    label = np.concatenate([id_gt, ood_gt])
    ood_detection_metrics = compute_all_metrics(conf, label, pred, plot_roc_curve=True,
                                                save_path=save_results_locations)
    filename = f'{args.model}_freeze_tile_embedding_{args.freeze_tile_embedding}_' \
               f'ood_detection_method_{args.ood_detection_method}.txt'

    with open(os.path.join(save_results_locations, filename), "a") as f:
        print(f"OOD detection FPR at 95% TPR: {ood_detection_metrics[0]}\n"
              f"OOD detection AUCROC: {ood_detection_metrics[1]},\n"
              f"OOD detection AUCPR ID: {ood_detection_metrics[2]},\n"
              f"OOD detection AUCPR OOD: {ood_detection_metrics[3]},\n", file=f)