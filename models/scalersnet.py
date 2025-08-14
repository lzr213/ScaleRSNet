# -*- coding: utf-8 -*-
# @Time    : 2024/09/01 11:15
# @Author  : xuxing
# @Site    : 
# @File    : scalersnet.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Calculate mean and max values along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate the two feature maps
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        
        return self.sigmoid(out) * x


class MSCABlock(nn.Module):
    """Multi-Scale Context Aggregation Module"""
    def __init__(self, in_channels, reduction=16):
        super(MSCABlock, self).__init__()
        self.channels = in_channels
        
        # Multi-scale dilated convolutions
        self.branch1 = ConvBNReLU(in_channels, in_channels // 4, kernel_size=1, padding=0)
        self.branch2 = ConvBNReLU(in_channels, in_channels // 4, kernel_size=3, padding=1, dilation=1)
        self.branch3 = ConvBNReLU(in_channels, in_channels // 4, kernel_size=3, padding=2, dilation=2)
        self.branch4 = ConvBNReLU(in_channels, in_channels // 4, kernel_size=3, padding=4, dilation=4)
        
        # Channel attention
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio=reduction)
        
        # Final fusion
        self.conv_out = ConvBNReLU(in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        # Concatenate multi-scale features
        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        
        # Apply channel attention
        out = self.channel_attention(out)
        
        # Final convolution
        out = self.conv_out(out)
        
        return out + x  # Residual connection


class MLPBlock(nn.Module):
    """Multi-Layer Perceptron Block"""
    def __init__(self, in_channels, mlp_ratio=4):
        super(MLPBlock, self).__init__()
        hidden_features = int(in_channels * mlp_ratio)
        
        self.fc1 = nn.Conv2d(in_channels, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, in_channels, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x + shortcut


class ScaleRSBlock(nn.Module):
    """ScaleRSNet Basic Building Block"""
    def __init__(self, in_channels, out_channels):
        super(ScaleRSBlock, self).__init__()
        
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.msca = MSCABlock(in_channels)
        self.mlp = MLPBlock(in_channels)
        
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        shortcut = x
        
        # Depthwise convolution
        x = self.dwconv(x)
        x = self.norm(x)
        
        # Multi-scale context attention
        x = self.msca(x)
        
        # MLP
        x = self.mlp(x)
        
        # Project to target channel dimension
        x = self.proj(x + shortcut)
        
        return x


class ScaleRSNet(nn.Module):
    """ScaleRSNet Semantic Segmentation Model"""
    def __init__(self, in_channels=3, out_channels=6, base_channels=64):
        super(ScaleRSNet, self).__init__()
        
        # Initial convolution
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, base_channels, kernel_size=3, stride=2),
            ConvBNReLU(base_channels, base_channels, kernel_size=3, stride=1)
        )
        
        # Encoder stages
        self.stage1 = nn.Sequential(
            ScaleRSBlock(base_channels, base_channels),
            ScaleRSBlock(base_channels, base_channels)
        )
        
        self.down1 = ConvBNReLU(base_channels, base_channels*2, kernel_size=3, stride=2)
        
        self.stage2 = nn.Sequential(
            ScaleRSBlock(base_channels*2, base_channels*2),
            ScaleRSBlock(base_channels*2, base_channels*2)
        )
        
        self.down2 = ConvBNReLU(base_channels*2, base_channels*4, kernel_size=3, stride=2)
        
        self.stage3 = nn.Sequential(
            ScaleRSBlock(base_channels*4, base_channels*4),
            ScaleRSBlock(base_channels*4, base_channels*4),
            ScaleRSBlock(base_channels*4, base_channels*4)
        )
        
        self.down3 = ConvBNReLU(base_channels*4, base_channels*8, kernel_size=3, stride=2)
        
        self.stage4 = nn.Sequential(
            ScaleRSBlock(base_channels*8, base_channels*8),
            ScaleRSBlock(base_channels*8, base_channels*8),
            ScaleRSBlock(base_channels*8, base_channels*8)
        )
        
        # Decoder stages
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True)
        )
        
        self.decoder1 = nn.Sequential(
            ScaleRSBlock(base_channels*8, base_channels*4),  # Channels after concatenation
            ScaleRSBlock(base_channels*4, base_channels*4)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        
        self.decoder2 = nn.Sequential(
            ScaleRSBlock(base_channels*4, base_channels*2),  # Channels after concatenation
            ScaleRSBlock(base_channels*2, base_channels*2)
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.decoder3 = nn.Sequential(
            ScaleRSBlock(base_channels*2, base_channels),  # Channels after concatenation
            ScaleRSBlock(base_channels, base_channels)
        )
        
        # Final output layer
        self.final_up = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder forward pass
        x0 = self.stem(x)              # [B, C, H/2, W/2]
        
        x1 = self.stage1(x0)           # [B, C, H/2, W/2]
        x2 = self.down1(x1)            # [B, 2C, H/4, W/4]
        x2 = self.stage2(x2)           # [B, 2C, H/4, W/4]
        
        x3 = self.down2(x2)            # [B, 4C, H/8, W/8]
        x3 = self.stage3(x3)           # [B, 4C, H/8, W/8]
        
        x4 = self.down3(x3)            # [B, 8C, H/16, W/16]
        x4 = self.stage4(x4)           # [B, 8C, H/16, W/16]
        
        # Decoder forward pass (with skip connections)
        d1 = self.up1(x4)              # [B, 4C, H/8, W/8]
        d1 = torch.cat([d1, x3], dim=1)  # [B, 8C, H/8, W/8]
        d1 = self.decoder1(d1)         # [B, 4C, H/8, W/8]
        
        d2 = self.up2(d1)              # [B, 2C, H/4, W/4]
        d2 = torch.cat([d2, x2], dim=1)  # [B, 4C, H/4, W/4]
        d2 = self.decoder2(d2)         # [B, 2C, H/4, W/4]
        
        d3 = self.up3(d2)              # [B, C, H/2, W/2]
        d3 = torch.cat([d3, x1], dim=1)  # [B, 2C, H/2, W/2]
        d3 = self.decoder3(d3)         # [B, C, H/2, W/2]
        
        # Final output
        out = self.final_up(d3)        # [B, C, H, W]
        out = self.final_conv(out)     # [B, num_classes, H, W]
        
        return out


if __name__ == "__main__":
    # Test code
    x = torch.randn(2, 3, 256, 256)  # Batch size 2, 3 channels, 256x256 resolution
    model = ScaleRSNet(in_channels=3, out_channels=6, base_channels=64)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}") 