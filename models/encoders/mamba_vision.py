# -*- coding: utf-8 -*-
# @Time    : 2024/10/1 1:52
# @Author  : xuxing
# @Site    : 
# @File    : mamba_vision.py
# @Software: PyCharm

# 论文：https://arxiv.org/pdf/2407.08083
# 参考代码：https://github.com/NVlabs/MambaVision

from transformers import AutoModel
from PIL import Image
from timm.data.transforms_factory import create_transform


import torch
import torch.nn as nn
import timm
from timm import models

class MultiScaleEncoderMambaVision(nn.Module):
    def __init__(self,
                 model_name='',
                 pretrained=True,
                 model_path=r'E:\work\codes\RSegment\pretrained\MambaVision_T_1K',
                 out_indices=[0,1,2,3]):
        super(MultiScaleEncoderMambaVision, self).__init__()
        
        self.model = AutoModel.from_pretrained(pretrained=pretrained,
                                               pretrained_model_name_or_path=model_path)
        self.out_indices = out_indices
        
    def forward(self, x):
        _, features = self.model(x)
        features = features[self.out_indices]
        return features

# 使用特征提取模型
if __name__ == "__main__":
    feature_extractor = MultiScaleEncoderMambaVision()
    image_tensor = torch.randn(2, 3, 512, 512)
    features = feature_extractor(image_tensor)
    
    for i, feature in enumerate(features):
        print(f"Feature map {i + 1}: {feature.shape}")
