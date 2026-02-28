# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import numpy as np
from PIL import Image
from torchvision import transforms

from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment
from semilearn.datasets.utils import split_ssl_data


# 自定义数据集的均值和标准差（根据实际数据计算）
mean, std = {}, {}
mean['custom'] = [0.232, 0.227, 0.276]  # 均值
std['custom'] = [0.208, 0.222, 0.226]   # 标准差
mean['custom_png'] = [0.2155,0.1994, 0.157]  # 均值
std['custom_png'] = [0.2875, 0.2402, 0.2327]  # 标准差
mean['custom_01'] = [0.1299,0.1254, 0.176]  # 均值
std['custom_01'] = [0.1372, 0.1538,0.1665]  # 标准差

def get_custom(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    """
    自定义数据集加载函数，与get_cifar接口保持一致
    
    Args:
        args: 配置参数
        alg: 算法名称
        name: 数据集名称（此处为'custom'）
        num_labels: 有标签数据量
        num_classes: 类别数
        data_dir: 数据存储根目录
        include_lb_to_ulb: 是否将有标签数据包含到无标签数据中
    """
    # 1. 数据路径处理（自定义数据集目录结构）
    data_dir = os.path.join(data_dir, name.lower())  # 数据根目录：./data/custom
    train_dir = os.path.join(data_dir, 'train')      # 训练集目录（包含所有训练样本）
    test_dir = os.path.join(data_dir, 'test')        # 测试集目录

    # 2. 加载原始数据（需根据自定义数据集的存储格式修改）
    # 假设训练集所有样本存放在train_dir下，每个类别一个子文件夹
    def load_data_from_dir(root_dir):
        data = []
        targets = []
        for label in range(num_classes):
            cls_dir = os.path.join(root_dir, f'class_{label}')
            if not os.path.exists(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if img_name.endswith(('png', 'jpg', 'jpeg')):
                    data.append(np.array(Image.open(img_path).convert('RGB')))  # 转为numpy数组
                    targets.append(label)
        return np.array(data), np.array(targets)

    # 加载训练集（用于分割为有标签/无标签数据）
    train_data, train_targets = load_data_from_dir(train_dir)
    # 加载测试集（用于验证）
    test_data, test_targets = load_data_from_dir(test_dir)

    # 3. 定义数据增强（与get_cifar保持一致的增强策略）
    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    # 弱增强（用于有标签数据和无标签数据的弱增强视图）
    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    # 中增强（部分算法使用，如自定义算法需要）
    transform_medium = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 5),  # 1层RandAugment，强度5
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    # 强增强（用于无标签数据的强增强视图）
    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),  # 3层RandAugment，强度5
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    # 验证集增强（仅Resize和标准化）
    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    # 4. 分割有标签和无标签数据（复用框架的split_ssl_data函数）
    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(
        args, 
        train_data, 
        train_targets, 
        num_classes, 
        lb_num_labels=num_labels,
        ulb_num_labels=args.ulb_num_labels,  # 从配置参数获取无标签数据相关设置
        lb_imbalance_ratio=args.lb_imb_ratio,  # 有标签数据不平衡比例
        ulb_imbalance_ratio=args.ulb_imb_ratio,  # 无标签数据不平衡比例
        include_lb_to_ulb=include_lb_to_ulb  # 是否将有标签数据加入无标签数据
    )

    # 5. 打印类别分布（可选，用于调试）
    lb_count = [0] * num_classes
    for c in lb_targets:
        lb_count[c] += 1
    ulb_count = [0] * num_classes
    for c in ulb_targets:
        ulb_count[c] += 1
    print(f"Custom dataset labeled count: {lb_count}")
    print(f"Custom dataset unlabeled count: {ulb_count}")

    # 6. 全监督模式下的特殊处理（与get_cifar保持一致）
    if alg == 'fullysupervised':
        lb_data = train_data
        lb_targets = train_targets

    # 7. 创建数据集实例（使用BasicDataset）
    # 有标签数据集（is_ulb=False，使用弱增强和强增强）
    lb_dset = BasicDataset(
        alg=alg,
        data=lb_data,
        targets=lb_targets,
        num_classes=num_classes,
        transform=transform_weak,
        is_ulb=False,
        strong_transform=transform_strong  # 部分算法（如DeFixMatch）需要有标签数据的强增强
    )

    # 无标签数据集（is_ulb=True，使用弱增强、中增强和强增强）
    ulb_dset = BasicDataset(
        alg=alg,
        data=ulb_data,
        targets=ulb_targets,
        num_classes=num_classes,
        transform=transform_weak,
        is_ulb=True,
        strong_transform=transform_strong  # 无标签数据的强增强视图
    )

    # 验证集（使用验证集增强）
    eval_dset = BasicDataset(
        alg=alg,
        data=test_data,
        targets=test_targets,
        num_classes=num_classes,
        transform=transform_val,
        is_ulb=False
    )

    return lb_dset, ulb_dset, eval_dset