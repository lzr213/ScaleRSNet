# -*- coding: utf-8 -*-
# @Time    : 2024/9/23 22:48
# @Author  : xuxing
# @Site    : 
# @File    : color_gray.py
# @Software: PyCharm

import os
import numpy as np
from PIL import Image
import tifffile as tiff
from tqdm import tqdm

# 定义颜色到类别的映射表
color_to_label = {
    (255, 255, 255): 0,  # Impervious surfaces
    (0, 0, 255): 1,  # Building
    (0, 255, 255): 2,  # Low vegetation
    (0, 255, 0): 3,  # Tree
    (255, 255, 0): 4,  # Car
    (255, 0, 0): 5  # Clutter/background
}


def tif_to_label_map(tif_image_path):
    """
    将 TIF 彩色标签图像转换为数值标签。

    :param tif_image_path: 输入的 TIF 标签图像路径
    :return: 数值标签矩阵
    """
    # 打开TIF标签图像
    tif_image = Image.open(tif_image_path)
    
    # 转换为NumPy数组 (H, W, 3)
    tif_data = np.array(tif_image)
    
    # 创建与原图大小一致的数值标签矩阵
    label_map = np.zeros(tif_data.shape[:2], dtype=np.uint8)
    
    # 将每个颜色映射到相应的类别
    for color, label in color_to_label.items():
        mask = np.all(tif_data == color, axis=-1)  # 查找所有匹配颜色的像素位置
        label_map[mask] = label
    
    return label_map


def process_labels_directory(labels_dir, output_dir):
    """
    处理整个 labels 目录中的所有 TIF 彩色标签，并将其转换为单通道数值 TIF 文件。

    :param labels_dir: 标签文件所在目录
    :param output_dir: 保存单通道数值标签的输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历所有 .tif 文件
    for filename in tqdm(os.listdir(labels_dir)):
        if filename.endswith(".tif"):
            tif_path = os.path.join(labels_dir, filename)
            label_map = tif_to_label_map(tif_path)
            
            # 保存为单通道的 TIF 格式
            output_path = os.path.join(output_dir, filename)
            tiff.imwrite(output_path, label_map, dtype=np.uint8)


# 使用示例
labels_dir = r'E:\work\data\potsdam\test\labels'  # 替换为你的标签目录
output_dir = r'E:\work\data\potsdam\test\labels_digital'  # 替换为输出目录
process_labels_directory(labels_dir, output_dir)