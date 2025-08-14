# -*- coding: utf-8 -*-
# @Time    : 2024/8/27 0:15
# @Author  : xuxing
# @Site    : 
# @File    : utils.py
# @Software: PyCharm

# 禁用albumentations版本检查警告
import os
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"

import argparse
import os
from loguru import logger
import random
import numpy as np
import torch

def set_seed(seed=42):
    """设置随机种子以保证结果可复现，默认为42"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置 cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f'Set random seed to {seed}')

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Image Segmentation Training Script")

    parser.add_argument("--train_data_dir", type=str, default='./data/WHDLD/outputs/train', help="Path to training data directory")
    parser.add_argument("--val_data_dir", type=str, default='./data/WHDLD/outputs/val', help="Path to validation data directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_classes", type=int, default=6, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save models and logs")
    parser.add_argument("--resume", type=str, default='./outputs/20240831_202015/best_model_acc_0.8083_miou_0.6783.pth', help="Path to a checkpoint to resume training")

    return parser.parse_args()


def setup_logging(output_dir):
    """
    设置loguru的日志记录，按照当前运行时间作为文件名保存

    :param output_dir: 日志文件保存的目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    log_file_path = os.path.join(output_dir, "training.log")
    
    logger.remove()
    logger.add(log_file_path, rotation="500 MB", retention="10 days", level="INFO")
    
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    logger.info(f"Logging is set up. Logs are being saved to {log_file_path}.")