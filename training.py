from __future__ import print_function

import os
import yaml
import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
import itertools
import torch.utils.data as data_utils
import torch.optim as optim
from functools import partial
from pathlib import Path

from datasets import DatasetBags, DatasetBagsHisto
from model import Attention, GatedAttentionMNIST, GatedAttentionCIFAR, AverageMIL, MaxMIL
from utils import apply_random_seed, seed_worker, measure_perf

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--experiment', type=str, default='test', metavar='E',
                    help='experiment name (default: test)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',  # 40
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'pcam'],
                    default='mnist', help='dataset to use: mnist, fashion-mnist or kmnist (default: mnist)')
parser.add_argument('--target_class_id', type=int, nargs='+', default=[9], metavar='T',
                    help='bags have a positive labels if they contain at least one target_class_id (default: 9)')
parser.add_argument('--neg_class_id', type=int, nargs='+', default=None, metavar='N',
                    help='class id for negative instances (default: None, consider all classes except target_class_id)')
parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, nargs='+', default=[200], metavar='NTrain',  # 9000
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, nargs='+', default=[50], metavar='NTest',  # 370
                    help='number of bags in test set')
parser.add_argument('--val_size', type=float, default=0.2, metavar='vs',
                    help='size of validation set')
parser.add_argument('--test_size_histo', type=float, default=0.2, metavar='ts',
                    help='test size for histology datasets')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='gated_attention', choices=["gated_attention", "average", "max"],
                    help='Choose b/w attention and gated_attention')
parser.add_argument('--freeze_tile_embedding', action='store_true', default=False, help='Freeze tile embedding')
parser.add_argument('--params_config', type=str, default=None,
                    help='Parameters config file (to reproduce results)')

args = parser.parse_args()
if args.params_config is not None:
    with open(args.params_config, 'r') as f:
        params_config = yaml.load(f, Loader=yaml.FullLoader)
    args = argparse.Namespace(**params_config)

args.cuda = not args.no_cuda and torch.cuda.is_available()

Path(args.experiment).mkdir(parents=True, exist_ok=True)

apply_random_seed(args.seed)

print('Load Train and Test Set')

g = torch.Generator()
g.manual_seed(args.seed)
loader_kwargs = {'num_workers': 0, 'pin_memory': True, 'generator': g, "worker_init_fn": seed_worker}

dataset_fct = partial(DatasetBagsHisto, test_size=args.test_size_histo) if args.dataset == 'pcam' else DatasetBags

train_dataset = dataset_fct(dataset=args.dataset,
                            target_class_id=args.target_class_id,
                            neg_class_id=args.neg_class_id,
                            mean_bag_length=args.mean_bag_length,
                            var_bag_length=args.var_bag_length,
                            num_bag=args.num_bags_train,
                            seed=args.seed,
                            mode="train",
                            perf_aug=True)

val_dataset = dataset_fct(dataset=args.dataset,
                          target_class_id=args.target_class_id,
                          neg_class_id=args.neg_class_id,
                          mean_bag_length=args.mean_bag_length,
                          var_bag_length=args.var_bag_length,
                          num_bag=[int(num_bag * args.val_size) for num_bag in args.num_bags_train],
                          seed=args.seed+1, # to not sample the same bags as in the training set
                          mode="val")

train_loader = data_utils.DataLoader(train_dataset,
                                     batch_size=1,
                                     shuffle=True,
                                     **loader_kwargs)

val_loader = data_utils.DataLoader(val_dataset,
                                   batch_size=1,
                                   shuffle=True,
                                   **loader_kwargs)

test_dataset = dataset_fct(dataset=args.dataset,
                           target_class_id=args.target_class_id,
                           neg_class_id=args.neg_class_id,
                           mean_bag_length=args.mean_bag_length,
                           var_bag_length=args.var_bag_length,
                           num_bag=args.num_bags_test,
                           seed=args.seed,
                           mode="test")

test_loader = data_utils.DataLoader(test_dataset,
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

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

loss_function = torch.nn.CrossEntropyLoss()


def train():
    losses = []
    proba_predictions = []
    ground_truths = []
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label[0].long()
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        if "attention" in args.model:
            pred, attention_weights = model(data)
        else:
            pred = model(data)
        loss = loss_function(pred, bag_label)
        pred = torch.softmax(pred, dim=-1)
        # backward pass
        loss.backward()
        # step
        optimizer.step()

        losses.append(loss.detach().cpu().numpy())
        proba_predictions.extend(pred.detach().cpu().numpy())
        ground_truths.extend(bag_label.detach().cpu().numpy())

    proba_predictions = np.array(proba_predictions)
    predicted_classes = np.argmax(proba_predictions, axis=1)

    return losses, proba_predictions, predicted_classes, ground_truths


def val():
    losses = []
    proba_predictions = []
    ground_truths = []
    model.eval()
    for batch_idx, (data, label) in enumerate(val_loader):
        bag_label = label[0].long()
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        # calculate loss and metrics
        if "attention" in args.model:
            pred, attention_weights = model(data)
        else:
            pred = model(data)
        loss = loss_function(pred, bag_label)
        pred = torch.softmax(pred, dim=-1)

        losses.append(loss.detach().cpu().numpy())
        proba_predictions.extend(pred.detach().cpu().numpy())
        ground_truths.extend(bag_label.detach().cpu().numpy())

    proba_predictions = np.array(proba_predictions)
    predicted_classes = np.argmax(proba_predictions, axis=1)

    return losses, proba_predictions, predicted_classes, ground_truths


def visualize_test_samples(data, attention_weights, bag_level, pred, batch_idx, mean, std, save_location):
    images = [(data[0][i]).squeeze().detach().cpu() for i in range(data[0].shape[0])]
    if args.dataset != 'mnist':
        mean = torch.FloatTensor(mean).view(3, 1, 1)
        std = torch.FloatTensor(std).view(3, 1, 1)
    else:
        mean = torch.FloatTensor(mean).view(1, 1, 1)
        std = torch.FloatTensor(std).view(1, 1, 1)
    images = [(image * std + mean).permute(1, 2, 0).numpy() for image in images]
    attention_weights = [round(a_w, 3) for a_w in attention_weights[0].detach().cpu().numpy()]
    pred = pred[0].detach().cpu().numpy()
    row_count = 3
    col_count = 8

    fig, axes = plt.subplots(nrows=row_count, ncols=col_count, figsize=(40, 20))
    fig.suptitle(f'True Bag Label {bag_level[0]}, Predicted Bag Label: {bag_level[1]}, '
                 f'Maximum softmax probability: {round(pred.item(), 3)}', fontsize=30)
    for idx, (i, j) in enumerate(itertools.product(range(row_count), range(col_count))):
        if idx > len(images) - 1:
            axes[i, j].set_visible(False)
            continue
        axes[i, j].axis("off")
        axes[i, j].text(15, 26, str(attention_weights[idx]), color='red',
                        bbox=dict(facecolor='none', edgecolor='red'), fontsize=28)
        if 'mnist' in args.dataset:
            axes[i, j].imshow(images[idx], aspect="auto")
        else:
            axes[i, j].imshow(images[idx], aspect="auto")

    plt.subplots_adjust(wspace=.05, hspace=.05)
    save_images_locations = os.path.join(save_location, f'images_{args.model}_'
                                                        f'freeze_tile_embedding_{args.freeze_tile_embedding}')
    if os.path.isdir(save_images_locations) is False:
        os.mkdir(save_images_locations)
    plt.savefig(os.path.join(save_images_locations, f'test_sample_{batch_idx}.png'))


def test(save_location):
    model.eval()
    losses = []
    proba_predictions = []
    ground_truths = []
    for batch_idx, (data, label) in enumerate(test_loader):
        bag_label = label[0].long()
        instance_labels = label[1]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        if "attention" in args.model:
            pred, attention_weights = model(data)
        else:
            pred = model(data)
        loss = loss_function(pred, bag_label)
        pred = torch.softmax(pred, dim=-1)

        losses.append(loss.detach().cpu().numpy())
        proba_predictions.extend(pred.detach().cpu().numpy())
        ground_truths.extend(bag_label.detach().cpu().numpy())

        if batch_idx < 10 and "attention" in args.model:  # plot bag labels and instance labels for first 10 bags
            pred, predicted_label = torch.max(pred, axis=1)
            bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0]))
            instance_level = list(zip(instance_labels.numpy()[0].tolist(),
                                      np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))
            visualize_test_samples(data, attention_weights, bag_level, pred, batch_idx, mean=test_dataset.mean,
                                   std=test_dataset.std, save_location=save_location)
            print('\nTrue Bag Label, Predicted Bag Label: {}\n'
                  'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))

    proba_predictions = np.array(proba_predictions)
    predicted_classes = np.argmax(proba_predictions, axis=1)

    test_loss, test_bac, test_f1, test_auc = measure_perf(losses, ground_truths,
                                                          predicted_classes, proba_predictions, n_classes)
    return test_loss, test_bac, test_f1, test_auc


if __name__ == "__main__":
    print('Start Training')
    best_val_auc = 0
    save_location = os.path.join(args.experiment, f'{args.dataset}_{args.target_class_id}_ID')
    if os.path.isdir(save_location) is False:
        os.mkdir(save_location)
    filename = f'{args.model}_freeze_tile_embedding_{args.freeze_tile_embedding}'
    for epoch in range(1, args.epochs + 1):
        train_losses, train_probas, train_predicted_classes, train_ground_truths = train()
        train_loss, train_bac, train_f1, train_auc = measure_perf(train_losses, train_ground_truths,
                                                                  train_predicted_classes,
                                                                  train_probas, n_classes)
        print('Epoch', f'{epoch:3d}/{args.epochs}', f'    train_loss={train_loss:.3f}',
              f' train_bac={train_bac:.3f}', f'    train_f1={train_f1:.3f}', f'    train_auc={train_auc:.3f}')
        print('Start Validation')
        val_losses, val_probas, val_predicted_classes, val_ground_truths = val()
        val_loss, val_bac, val_f1, val_auc = measure_perf(val_losses, val_ground_truths, val_predicted_classes,
                                                          val_probas, n_classes)
        print('Epoch', f'{epoch:3d}/{args.epochs}', f'    val_loss={val_loss:.3f}',
              f' val_bac={val_bac:.3f}', f'    val_f1={val_f1:.3f}', f'    val_auc={val_auc:.3f}')
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            print('Saving Model')
            model_file_name = f'{filename}.pt'
            torch.save(model.state_dict(), os.path.join(save_location, model_file_name))
    model_file_name = f'{filename}.pt'
    print('Start Testing')
    model.load_state_dict(torch.load(os.path.join(save_location, model_file_name)))
    test_loss, test_bac, test_f1, test_auc = test(save_location)
    results_file_name = f'{filename}.txt'
    with open(os.path.join(save_location, results_file_name), "a") as f:
        print(f"Testing auc score: {test_auc},\n"
              f"Testing f1 score: {test_f1},\n"
              f"Testing bac score: {test_bac},\n",
              file=f)
    if args.params_config is None:
        print("Saving parameters for reproducibility")
        with open(os.path.join(save_location, f'{filename}.yml'), "a") as f:
            yaml.dump(args.__dict__, f)
