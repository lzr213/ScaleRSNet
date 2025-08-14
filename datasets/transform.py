# -*- coding: utf-8 -*-
# @Time    : 2024/8/26 10:59
# @Author  : xuxing
# @Site    : 
# @File    : transform.py
# @Software: PyCharm

# 禁用albumentations版本检查警告（最先导入）
import os
import warnings
import sys

# 设置环境变量
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"

# 忽略所有来自albumentations的警告
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

# 完全禁用与albumentations相关的警告输出
if 'albumentations' in sys.modules:
    sys.modules['albumentations'].check_version.fetch_version_info = lambda: {}
from albumentations.pytorch import ToTensorV2
from albumentations import *


transform = {
        'train':Compose([
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            GaussianBlur(p=0.1),
            Resize(height=512, width=512, p=1.0),
            Normalize(),
            ToTensorV2()
        ]),
        'val':Compose([
            Resize(height=512, width=512),
            Normalize(),
            ToTensorV2()
        ]),
        'test': Compose([
            Resize(height=512, width=512),
            Normalize(),
            ToTensorV2()
        ])
}

# 为EarthVQA数据集创建更大尺寸的transform（可选使用）
earthvqa_transform = {
    'train':Compose([
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.2),
        GaussianBlur(p=0.1),
        Resize(height=1024, width=1024, p=1.0),  # 保持原始分辨率
        Normalize(),
        ToTensorV2()
    ]),
    'val':Compose([
        Resize(height=1024, width=1024),
        Normalize(),
        ToTensorV2()
    ]),
    'test': Compose([
        Resize(height=1024, width=1024),
        Normalize(),
        ToTensorV2()
    ])
}

