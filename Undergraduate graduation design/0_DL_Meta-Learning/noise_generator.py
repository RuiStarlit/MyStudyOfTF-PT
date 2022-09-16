import torch
import numpy as np
import torchvision.datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import copy



# corruption_ratio：噪声率 num_classes：类别数目
# return 噪声矩阵
def uniform_corruption(corruption_ratio, num_classes):
    noise = np.full((num_classes, num_classes), 1/num_classes)
    corruption_matrix = np.eye(num_classes) * (1-corruption_ratio) + \
        noise * corruption_ratio
    return corruption_matrix
# def uniform_corruption(corruption_ratio, num_classes):
#     k = corruption_ratio/(num_classes-1)
#     noise = np.full((num_classes, num_classes), k)
#     corruption_matrix = np.eye(num_classes) * (1-corruption_ratio-k) + noise
#     return corruption_matrix

def flip1_corruption(corruption_ratio, num_classes):
    corruption_matrix = np.eye(num_classes) * (1-corruption_ratio)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        corruption_matrix[i][np.random.choice(
            row_indices[row_indices != i])] = corruption_ratio
    return corruption_matrix

def flip2_corruption(corruption_ratio, num_classes):
    corruption_matrix = np.eye(num_classes) * (1 - corruption_ratio)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        corruption_matrix[i][np.random.choice(
            row_indices[row_indices != i], 2, replace=False)] = corruption_ratio / 2
    return corruption_matrix

def cifar_dataloader(
    seed = 1,
    dataset = 'cifar10',
    num_meta_total = 1000,
    imbalanced_factor=None,
    corruption_type=None,
    corruption_ratio=0.,
    batch_size=128,
    num_workers=4,
    augment = 1,
):
    """
    Args:
        seed:随机数种子
        dataset:所使用的数据集
        num_meta_total:元数据集的总大小
        imbalanced_factor:不均衡比例
        corruption_type:添噪类型
        corruption_ratio:噪声率
        batch_size
    return:
        train_dataloader, meta_dataloader, test_dataloader, imbalanced_num_list
    """
    np.random.seed(seed)

    # normalize = transforms.Normalize(
    #     mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    #     std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    # )
    normalize = torchvision.transforms.Normalize(
        mean=[0.491, 0.482, 0.446],
        std=[0.247, 0.243, 0.261],
    )
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], 
    #     std=[0.229, 0.224, 0.225]
    # ) # if using pretrain-resnet, need to transformer like that 
    #   # https://pytorch.org/hub/pytorch_vision_resnet/
    
    train_transforms1 = transforms.Compose([
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        normalize,
    ])

    train_transforms2 = transforms.Compose([
        transforms.RandAugment(),
        transforms.ToTensor(),
        normalize,
    ])

    train_transforms3 = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    trans = [train_transforms1, train_transforms2, train_transforms3]
    train_transforms = trans[augment]
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset_list = {
        'cifar10': torchvision.datasets.CIFAR10,
        'cifar100': torchvision.datasets.CIFAR100,
    }

    corruption_list = {
        'uniform': uniform_corruption,
        'flip1': flip1_corruption,
        'flip2': flip2_corruption,
    }

    train_dataset = dataset_list[dataset](
        root='../data', train=True, download=True, transform=train_transforms)
    test_dataset = dataset_list[dataset](
        root='../data', train=False, transform=test_transforms)
    
    num_classes = len(train_dataset.classes)
    num_meta = int(num_meta_total / num_classes) # Number of meta per class

    index_to_meta = []
    index_to_train = []

    if imbalanced_factor is not None:
        imbalanced_num_list = []
        sample_num = int((len(train_dataset.targets) - num_meta_total) / num_classes)
        print(sample_num)
        for class_index in range(num_classes):
            imbalanced_num = sample_num / \
                (imbalanced_factor ** (class_index / (num_classes - 1)))
            imbalanced_num_list.append(int(imbalanced_num))
        np.random.shuffle(imbalanced_num_list)
        print(imbalanced_num_list)
    else:
        imbalanced_num_list = None
    
    for class_index in range(num_classes):
        index_to_class = [index for index, label in enumerate(train_dataset.targets) if label == class_index]
        np.random.shuffle(index_to_class)
        index_to_meta.extend(index_to_class[:num_meta])
        index_to_class_for_train = index_to_class[num_meta:]

        if imbalanced_num_list is not None:
            index_to_class_for_train = index_to_class_for_train[:imbalanced_num_list[class_index]]

        index_to_train.extend(index_to_class_for_train)

    meta_dataset = copy.deepcopy(train_dataset)
    train_dataset.data = train_dataset.data[index_to_train]
    train_dataset.targets = list(np.array(train_dataset.targets)[index_to_train])
    meta_dataset.data = meta_dataset.data[index_to_meta]
    meta_dataset.targets = list(np.array(meta_dataset.targets)[index_to_meta])

    if corruption_type is not None:
        corruption_matrix = corruption_list[corruption_type](corruption_ratio, num_classes)
        print(corruption_matrix)
        for index in range(len(train_dataset.targets)):
            p = corruption_matrix[train_dataset.targets[index]]
            train_dataset.targets[index] = np.random.choice(num_classes, p=p)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    meta_dataloader = DataLoader(meta_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=2048, pin_memory=True, num_workers=num_workers)

    return train_dataloader, meta_dataloader, test_dataloader, imbalanced_num_list

def cifar_testdataloader(dataset = 'cifar10',batch_size=128,):
    """
    Args:
        dataset:所使用的数据集
        batch_size
    return:
        test_dataloader
    """
    normalize = torchvision.transforms.Normalize(
        mean=[0.491, 0.482, 0.446],
        std=[0.247, 0.243, 0.261],
    )
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], 
    #     std=[0.229, 0.224, 0.225]
    # ) # if using pretrain-resnet, need to transformer like that 
    #   # https://pytorch.org/hub/pytorch_vision_resnet/
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    dataset_list = {
        'cifar10': torchvision.datasets.CIFAR10,
        'cifar100': torchvision.datasets.CIFAR100,
    }
    test_dataset = dataset_list[dataset](
        root='../data', train=False, transform=test_transforms)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)

    return test_dataloader