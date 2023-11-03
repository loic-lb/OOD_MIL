# On the detection of Out-Of-Distribution samples in Multiple Instance Learning

This repository contains the code to reproduce the results of the paper "On the detection of Out-Of-Distribution 
samples in Multiple Instance Learning".

## Requirements

The code is written in Python 3.10 and the packages needed to run the code are provided in ```environment.yml```:

```
conda env create -f environment.yml
```

## Data

The dataset used for ID task are:
* **MNIST** (http://yann.lecun.com/exdb/mnist/)
* **CIFAR10** (https://www.cs.toronto.edu/~kriz/cifar.html)
* **PCAM** (https://github.com/basveeling/pcam)

The dataset used for OOD task are:
* **FashionMNIST** (https://github.com/zalandoresearch/fashion-mnist)
* **KMNIST** (https://github.com/rois-codh/kmnist)
* **SVHN** (http://ufldl.stanford.edu/housenumbers/)
* **places365** (http://places2.csail.mit.edu/download.html)
* **Textures** (https://www.robots.ox.ac.uk/~vgg/data/dtd/)

The data are automatically downloaded by the code, except for the PCAM dataset that must be downloaded manually, and placed
in the ```./datasets/PCAM``` folder.

## Training and testing the models on the ID task

To reproduce the models on the ID task, you may run the following command:

```
python training.py --params_config <params_config>
```

where ```<params_config>``` is the path to the configuration file to use. All the configuration files are in the 
```./results``` folder along with the results of the experiments. The trained models weights can directly be found
[here](https://drive.google.com/drive/folders/1zKSeAMSVAl3tkwDVO_QUfnxYYjFPx0nu?usp=sharing).

## Performance evaluation on the OOD detection

To evaluate the performance of the models on the OOD detection task, you may run the following command:

```
python perf_OOD.py --params_ID_training <params_config> --dataset_OOD <dataset_OOD> --target_class_id_OOD 5 
--ood_detection_method <ood_detection_method>
```

where ```<params_ID_training>``` is the path to the configuration file used to train the model on the ID task, 
```<dataset_OOD>``` is the name of the dataset to use for the OOD task, and ```<ood_detection_method>``` is the method
to use for the OOD detection. The available methods are:
* **baseline**: maximum softmax probability
* **mls**: maximum logits value
* **ebo**: energy-based OOD detection
* **odin**: ODIN OOD detection
* **dice**: DICE OOD detection
* **knn**: KNN OOD detection

## Acknowledgements

The code is based on the following repositories:
* https://github.com/Jingkang50/OpenOOD/tree/main
* https://github.com/AMLab-Amsterdam/AttentionDeepMIL

We would like to thank the authors for sharing their code.

## Reference

If you find this code useful, please cite the following paper:

```

```

## License

This code is released under the MIT License (refer to the LICENSE file for details).

