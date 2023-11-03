import os
import h5py
import numpy as np
import torch
import torch.utils.data as data_utils
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

dataset_dict = {"mnist": (datasets.MNIST, (60000, 10000), [0.1307], [0.3081]),
                "fashion-mnist": (datasets.FashionMNIST, (60000, 10000), [0.1307], [0.3081]),
                "kmnist": (datasets.KMNIST, (60000, 10000), [0.1307], [0.3081]),
                "cifar10": (datasets.CIFAR10, (50000, 10000), [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
                "textures": (datasets.DTD, (1880, 1880), [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
                "svhn": (datasets.SVHN, (73257, 26032), [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
                "places365": (datasets.Places365, (1803460, 36500), [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
                "iciar": (datasets.ImageFolder, "ICIAR2018_BACH_Challenge/Photos",
                          [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                "pcam": (datasets.ImageFolder, "pcam", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}


class DatasetBags(data_utils.Dataset):

    def __init__(self, dataset="mnist", target_class_id=None, neg_class_id=None, mean_bag_length=10, var_bag_length=2,
                 num_bag=None, seed=1, mode="train", resize_size=None, ratio_positive=0.4, perf_aug=False):
        if num_bag is None:
            num_bag = [250]
        if target_class_id is None:
            target_class_id = [9]
        self.target_class_id = np.array(target_class_id)  # set of target class id (if several classes, then it is a
        # multi-class multi-instance problem)
        self.neg_class_id = neg_class_id if neg_class_id is None else np.array(neg_class_id)  # set of negative class id
        self.mean_bag_length = mean_bag_length  # mean of the bag length
        self.var_bag_length = var_bag_length  # variance of the bag length
        self.num_bag_per_class = self._get_num_bag_per_class(num_bag)  # number of bags per target_class_id
        self.mode = mode  # train, val or test mode for selecting which split of the dataset to use
        self.dataset_name = dataset  # name of the dataset
        self.resize_size = resize_size  # if not None, resize images to this size
        self.ratio_positive = ratio_positive  # maximum ratio of positive bags in the dataset
        self.perf_aug = perf_aug  # if True, perform data augmentation on the bags

        self.r = np.random.RandomState(seed)

        self.mean, self.std = dataset_dict[self.dataset_name][2], dataset_dict[self.dataset_name][3]

        self.train_transforms = transforms.Compose([transforms.RandomRotation(180),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomVerticalFlip()])
        # transforms.ColorJitter(brightness=0.5, contrast=0.5,
        #                       saturation=0.5, hue=0.5)])

        if self.resize_size is not None:
            self.transform_ops = transforms.Compose([transforms.Resize(self.resize_size),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(self.mean, self.std)])
        else:
            self.transform_ops = transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize(self.mean, self.std)])

        self.bags_list, self.labels_list = self._create_bags()

    def _get_num_bag_per_class(self, num_bag):
        """

        :param num_bag: a list of integers describing the number of bags by self.target_class_id
        (in case of a single integer, the same number of bags is created for each class)
        :return: an array describing the number of bags by class (including negative class)
        """
        if len(num_bag) == 1:
            return np.array(num_bag * (len(self.target_class_id) + 1))
        elif len(num_bag) == len(self.target_class_id) + 1:
            return np.array(num_bag)
        else:
            raise ValueError("num_bag must be either a list of a single integer"
                             "or a list of length len(target_class_id)+1")

    def _data_to_bags(self, all_imgs, all_labels):
        """

        :param all_imgs: tensor containing all the images from the dataset
        :param all_labels: tensor containing all the labels from the dataset
        :return: bag version of the dataset with variable number of instances per bag
        and variable number of positive instances per bag
        """
        if len(set(all_labels.numpy()).intersection(set(self.target_class_id))) != len(self.target_class_id):
            raise ValueError("Not all target classes are present in the dataset")

        bags_list = []
        labels_list = []

        for i in range(len(self.num_bag_per_class)):
            for j in range(self.num_bag_per_class[i]):
                bag_length = int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
                if bag_length < 1:
                    bag_length = 1
                if self.neg_class_id is None:
                    neg_class_indices = np.where(np.logical_not(np.isin(all_labels, self.target_class_id)))[0]
                else:
                    neg_class_indices = np.where(np.isin(all_labels, self.neg_class_id))[0]
                if i == 0:
                    indices = torch.LongTensor(self.r.choice(neg_class_indices, bag_length, replace=False))
                    labels_in_bag = all_labels[indices]
                    labels_in_bag[:] = 0
                else:
                    pos_class_indices = np.where(all_labels == self.target_class_id[i - 1])[0]
                    nb_pos = self.r.randint(1, np.ceil(bag_length * self.ratio_positive) + 1)
                    indices_pos = torch.LongTensor(self.r.choice(pos_class_indices, nb_pos, replace=False))
                    indices_neg = torch.LongTensor(self.r.choice(neg_class_indices, bag_length - nb_pos, replace=False))
                    indices = torch.cat((indices_pos, indices_neg), dim=0)
                    indices = indices[self.r.permutation(len(indices))]

                    labels_in_bag = all_labels[indices]
                    labels_in_bag[labels_in_bag != self.target_class_id[i - 1]] = 0
                    labels_in_bag[labels_in_bag == self.target_class_id[i - 1]] = i

                bags_list.append(all_imgs[indices])
                labels_list.append(labels_in_bag)

        return bags_list, labels_list

    def _create_bags(self):
        """

        :return: bag version of the dataset with variable number of instances per bag
        and variable number of positive instances per bag
        """
        dataset_fct = dataset_dict[self.dataset_name][0]
        if self.perf_aug:
            transform_ops = transforms.Compose([self.train_transforms, self.transform_ops])
        else:
            transform_ops = self.transform_ops
        print(transform_ops)
        if self.dataset_name in ["textures", "svhn"]:
            dataset = dataset_fct('./datasets',
                                  split="train" if self.mode != "test" else "test",
                                  download=True,
                                  transform=transform_ops)
        elif self.dataset_name == "places365":
            dataset = dataset_fct('./datasets',
                                  split="train-standard" if self.mode != "test" else "val",
                                  small=True,
                                  download=False,
                                  transform=transform_ops)
        else:
            dataset = dataset_fct('./datasets',
                                  train=True if self.mode != "test" else False,
                                  download=True,
                                  transform=transform_ops)

        loader = data_utils.DataLoader(dataset,
                                       batch_size=len(dataset),
                                       shuffle=False)

        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        return self._data_to_bags(all_imgs, all_labels)

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, index):

        bag = self.bags_list[index]
        label = [max(self.labels_list[index]), self.labels_list[index]]

        return bag, label


class DatasetBagsHisto(DatasetBags):

    def __init__(self, dataset="pcam", target_class_id=None, neg_class_id=None, mean_bag_length=10, var_bag_length=2,
                 num_bag=None, seed=1, mode="train", resize_size=None, ratio_positive=0.4, test_size=0.2,
                 perf_aug=False):
        if num_bag is None:
            num_bag = [250]
        if target_class_id is None:
            target_class_id = [2]
        self.test_size = test_size  # test size for the train/test split (as splits are not provided for these datasets)
        super().__init__(dataset, target_class_id, neg_class_id, mean_bag_length, var_bag_length, num_bag, seed, mode,
                         resize_size, ratio_positive, perf_aug)

    @staticmethod
    def _process_pcam(filename_data, filename_labels, transform_ops):
        """

        :param filename_data: path to the h5 file containing the images
        :param filename_labels: path to the h5 file containing the labels
        :param transform_ops: transformation to apply to the images
        :return: processed images and labels
        """
        x = h5py.File(filename_data, 'r')['x'][:]
        x = torch.stack([transform_ops(Image.fromarray(im)) for im in x])
        y = h5py.File(filename_labels, 'r')['y'][:].flatten()
        y = torch.from_numpy(y)
        return x, y

    def _create_bags(self):
        """

        :return: bag version of the dataset with variable number of instances per bag
        and variable number of positive instances per bag
        """
        dataset_fct = dataset_dict[self.dataset_name][0]
        dataset_path = dataset_dict[self.dataset_name][1]
        if self.perf_aug:
            transform_ops = transforms.Compose([self.train_transforms, self.transform_ops])
        else:
            transform_ops = self.transform_ops
        if self.dataset_name == "iciar":
            dataset_size = 400
            indices = np.arange(dataset_size)
            train_indices, test_indices = train_test_split(indices, test_size=self.test_size,
                                                           random_state=0)

            dataset = dataset_fct(os.path.join('./datasets', dataset_path), transforms=transform_ops)
            loader = data_utils.DataLoader(data_utils.Subset(dataset, train_indices) if self.mode != "test" else
                                           data_utils.Subset(dataset, test_indices),
                                           batch_size=len(dataset),
                                           shuffle=False)

            for (batch_data, batch_labels) in loader:
                all_imgs = batch_data
                all_labels = batch_labels

            return self._data_to_bags(all_imgs, all_labels)

        elif self.dataset_name == "pcam":
            print(transform_ops)
            
            if self.mode != "test":
                x, y = self._process_pcam('./datasets/PCAM/camelyonpatch_level_2_split_train_x.h5',
                                          './datasets/PCAM/camelyonpatch_level_2_split_train_y.h5',
                                          transform_ops)
            else:
                x, y = self._process_pcam('./datasets/PCAM/camelyonpatch_level_2_split_test_x.h5',
                                          './datasets/PCAM/camelyonpatch_level_2_split_test_y.h5',
                                          transform_ops)

            return self._data_to_bags(x, y)

        else:
            raise NotImplementedError
