# -*- coding: utf-8 -*-
# @Time    : 2024/8/26 10:28
# @Author  : xuxing
# @Site    : 
# @File    : split_dataset.py
# @Software: PyCharm

# 禁用albumentations版本检查警告
import os
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"

import os
import random
from shutil import copy2
from tqdm import tqdm
from loguru import logger

def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    # 设置随机种子以保证结果可复现
    random.seed(seed)

    # 获取图像和标签文件列表
    images = sorted(os.listdir(image_dir))
    labels = sorted(os.listdir(label_dir))

    # 打乱数据
    data = list(zip(images, labels))
    random.shuffle(data)

    # 计算各个集的数量
    total_count = len(data)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = total_count - train_count - val_count

    # 划分数据集
    train_data = data[:train_count]
    val_data = data[train_count:train_count + val_count]
    test_data = data[train_count + val_count:]

    # 创建输出目录
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    # 复制数据到相应的目录
    def copy_files(data, split):
        for image, label in tqdm(data):
            copy2(os.path.join(image_dir, image), os.path.join(output_dir, split, 'images', image))
            copy2(os.path.join(label_dir, label), os.path.join(output_dir, split, 'labels', label))

    copy_files(train_data, 'train')
    copy_files(val_data, 'val')
    copy_files(test_data, 'test')

    logger.info(f"Dataset split into train: {train_count}, val: {val_count}, test: {test_count}")

if __name__ == '__main__':
    
    image_dir = '../dataset/WHDLD/Images'
    label_dir = '../dataset/WHDLD/ImagesPNG'
    save_dir = '../dataset/WHDLD/outputs'
    split_dataset(image_dir, label_dir, save_dir)