# -*- coding: utf-8 -*-
# @Time    : 2024/9/14 10:53
# @Author  : xuxing
# @Site    : 
# @File    : timm_encoder.py
# @Software: PyCharm

import torch
import torch.nn as nn
import timm
from timm import models

class MultiScaleEncoderTimm(nn.Module):
    def __init__(self, model_name='resnet34', pretrained=True, out_indices=[1,2,3,4]):
        super(MultiScaleEncoderTimm, self).__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            pretrained_cfg_overlay=dict(file="../pretrained/vit_tiny_patch16_224.bin"),
            features_only=True,
            out_indices=out_indices,  # 指定提取的层
        )
        print(self.model.default_cfg)
        self.out_indices = out_indices
        self.out_channels = [info['num_chs'] for i, info in enumerate(self.model.feature_info) if i in out_indices]
        
        print(self.out_channels)
    
    
    def forward(self, x):
        features = self.model(x)
        return features


# 使用特征提取模型
if __name__ == "__main__":
    feature_extractor = MultiScaleEncoderTimm(model_name='resnet34', pretrained=False)
    image_tensor = torch.randn(2, 3, 512, 512)
    features = feature_extractor(image_tensor)
    
    for i, feature in enumerate(features):
        print(f"Feature map {i + 1}: {feature.shape}")
