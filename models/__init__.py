# -*- coding: utf-8 -*-
# @Time    : 2024/8/27 0:09
# @Author  : xuxing
# @Site    : 
# @File    : __init__.py.py
# @Software: PyCharm

from .scalersnet import ScaleRSNet

def get_scalersnet_model(in_channels=3, out_channels=6, base_channels=64):
    """
    Create and return a ScaleRSNet model instance
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (classes)
        base_channels: Base channel count
    
    Returns:
        ScaleRSNet model instance
    """
    return ScaleRSNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)