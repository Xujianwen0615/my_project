# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math

from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data


mean, std = {}, {}
mean['cifar10'] = [0.485, 0.456, 0.406]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]


std['cifar10'] = [0.229, 0.224, 0.225]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]

mean['cifar10_lt'] = [0.485, 0.456, 0.406]
std['cifar10_lt'] = [0.229, 0.224, 0.225]


def get_cifar(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    
    data_dir = os.path.join(data_dir, name.lower())
    
    
    crop_size = args.img_size
    crop_ratio = args.crop_ratio
    
    # 1. 判断是否为长尾数据集 (假设你的数据集名称为 'cifar10_lt')
    if 'lt' in name.lower():
        # 从本地 npz 文件加载长尾数据
        # 假设你保存的文件路径格式为: data_dir/cifar10_lt/cifar10_lt_gamma{args.gamma}_train.npz
        # 你需要将 args.gamma 作为一个参数传递进来，或者有默认值
        gamma = getattr(args, 'gamma', 50.0)  # 假设gamma参数存储在args中，默认为50
        
        # 构建文件路径 (请根据你的实际文件命名调整)
        train_file = os.path.join(data_dir, f'cifar10_lt_gamma{gamma}_train.npz')
        test_file = os.path.join(data_dir, f'cifar10_lt_gamma{gamma}_test.npz')
        
        # 加载训练数据
        train_data = np.load(train_file)
        data = train_data['X_train']
        targets = train_data['y_train'].astype(np.int64)  # 确保标签为整数
        
        # 加载测试数据 (用于后面的eval_dset)
        test_data_npz = np.load(test_file)
        test_data = test_data_npz['X_test']
        test_targets = test_data_npz['y_test'].astype(np.int64)
        
        print(f"成功加载长尾数据集: {name}, gamma={gamma}")
        print(f"训练集形状: {data.shape}, 测试集形状: {test_data.shape}")
        
    else:
        # 原始逻辑：加载标准 CIFAR 数据集
        dset = getattr(torchvision.datasets, name.upper())
        dset = dset(data_dir, train=True, download=True)
        data, targets = dset.data, dset.targets
        
        # 获取标准测试集
        # dset_test = getattr(torchvision.datasets, name.upper())
        # dset_test = dset_test(data_dir, train=False, download=True)
        # test_data, test_targets = dset_test.data, dset_test.targets
    
    # dset = getattr(torchvision.datasets, name.upper())
    # dset = dset(data_dir, train=True, download=True)
    # data, targets = dset.data, dset.targets

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_medium = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name],)
    ])

    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, targets, num_classes, 
                                                                lb_num_labels=num_labels,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                lb_imbalance_ratio=args.lb_imb_ratio,
                                                                ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                include_lb_to_ulb=include_lb_to_ulb)
    
    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1
    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(ulb_count))
    # lb_count = lb_count / lb_count.sum()
    # ulb_count = ulb_count / ulb_count.sum()
    # args.lb_class_dist = lb_count
    # args.ulb_class_dist = ulb_count

    if alg == 'fullysupervised':
        lb_data = data
        lb_targets = targets
        # if len(ulb_data) == len(data):
        #     lb_data = ulb_data 
        #     lb_targets = ulb_targets
        # else:
        #     lb_data = np.concatenate([lb_data, ulb_data], axis=0)
        #     lb_targets = np.concatenate([lb_targets, ulb_targets], axis=0)
    
    # output the distribution of labeled data for remixmatch
    # count = [0 for _ in range(num_classes)]
    # for c in lb_targets:
    #     count[c] += 1
    # dist = np.array(count, dtype=float)
    # dist = dist / dist.sum()
    # dist = dist.tolist()
    # out = {"distribution": dist}
    # output_file = r"./data_statistics/"
    # output_path = output_file + str(name) + '_' + str(num_labels) + '.json'
    # if not os.path.exists(output_file):
    #     os.makedirs(output_file, exist_ok=True)
    # with open(output_path, 'w') as w:
    #     json.dump(out, w)

    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, transform_strong, transform_strong, False)

    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_medium, transform_strong, False)
    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_medium, transform_strong, False)

    # dset = getattr(torchvision.datasets, name.upper())
    # dset = dset(data_dir, train=False, download=True)
    # test_data, test_targets = dset.data, dset.targets
    if 'lt' not in name.lower():
        # 标准数据集的测试集加载方式保持不变
        dset = getattr(torchvision.datasets, name.upper())
        dset = dset(data_dir, train=False, download=True)
        test_data, test_targets = dset.data, dset.targets

    eval_dset = BasicDataset(alg, test_data, test_targets, num_classes, transform_val, False, None, None, False)

    return lb_dset, ulb_dset, eval_dset
