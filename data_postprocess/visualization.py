# -*- coding: utf-8 -*-
# @Time    : 2024/8/26 10:20
# @Author  : xuxing
# @Site    : 
# @File    : visualization.py
# @Software: PyCharm

from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from config.load_config import DatasetConfig
import matplotlib.pyplot as plt
import matplotlib
# 设置全局字体为Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
import json


def gray_color(input_dir, output_dir, color_mapping):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".tif"):
            gray_image_path = os.path.join(input_dir, filename)
            gray_image = Image.open(gray_image_path).convert('L')
            gray_array = np.array(gray_image)

            # Create a color image array
            color_image = np.zeros((*gray_array.shape, 3), dtype=np.uint8)

            for cls_id, color in color_mapping.items():
                color_image[gray_array == cls_id] = color

            # Convert the color image array back to a PIL image
            color_image_pil = Image.fromarray(color_image)

            # Save the color image
            color_image_path = os.path.join(output_dir, filename)
            color_image_pil.save(color_image_path)
            # print(f"Saved colored image: {color_image_path}")


class LossVisualizer:
    """用于可视化训练和验证损失的类"""
    
    def __init__(self, save_dir=None):
        """
        初始化可视化器
        
        Args:
            save_dir: 图表保存的目录路径
        """
        self.save_dir = save_dir
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def plot_training_history(self, training_history, title="Training History", save_path=None):
        """
        绘制训练历史图表
        
        Args:
            training_history: 包含训练历史数据的列表
            title: 图表标题
            save_path: 保存图表的路径
        """
        # 提取数据
        epochs = [item['epoch'] for item in training_history]
        train_loss = [item['train_loss'] for item in training_history]
        val_acc = [item['val_acc'] for item in training_history]
        val_miou = [item['val_miou'] for item in training_history]
        
        # 创建图表
        plt.figure(figsize=(12, 10))
        
        # 绘制训练损失和验证损失
        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_loss, 'b-', label='Training Loss')
        plt.title('Training and Validation Loss', fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True)
        plt.legend(fontsize=12)
        
        # 绘制验证精度和IoU
        plt.subplot(2, 1, 2)
        plt.plot(epochs, val_acc, 'g-', label='Validation Accuracy')
        plt.plot(epochs, val_miou, 'r-', label='Validation mIoU')
        plt.title('Validation Accuracy and IoU', fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.grid(True)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            if self.save_dir:
                save_path = os.path.join(self.save_dir, 'training_history.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Figure saved to {save_path}")
        
        # 关闭图表而不显示
        plt.close()
    
    def plot_metrics_comparison(self, metrics_list, labels, title="Metrics Comparison", save_path=None):
        """
        绘制不同模型或配置的指标比较
        
        Args:
            metrics_list: 包含多个模型指标的列表
            labels: 每个模型的标签
            title: 图表标题
            save_path: 保存图表的路径
        """
        # 确保数据匹配
        if len(metrics_list) != len(labels):
            raise ValueError("metrics_list and labels must have the same length")
        
        # 选择要比较的指标
        metrics_to_compare = ['mIoU', 'total_PA', 'mF1', 'macro_f1', 'micro_f1']
        
        # 提取数据
        metric_values = {metric: [] for metric in metrics_to_compare}
        
        for metrics in metrics_list:
            for metric in metrics_to_compare:
                if metric in metrics:
                    metric_values[metric].append(metrics[metric])
                else:
                    metric_values[metric].append(0)  # 如果指标不存在，使用0
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 设置x轴位置
        x = np.arange(len(labels))
        width = 0.15
        offsets = np.linspace(-(len(metrics_to_compare)-1)/2*width, (len(metrics_to_compare)-1)/2*width, len(metrics_to_compare))
        
        # 绘制每个指标的柱状图
        for i, metric in enumerate(metrics_to_compare):
            plt.bar(x + offsets[i], metric_values[metric], width, label=metric)
        
        # 设置图表属性
        plt.xlabel('Models/Configurations', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title(title, fontsize=14)
        plt.xticks(x, labels, fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for i, metric in enumerate(metrics_to_compare):
            for j, v in enumerate(metric_values[metric]):
                plt.text(j + offsets[i], v + 0.01, f'{v:.3f}', 
                         ha='center', va='bottom', fontsize=8, rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            if self.save_dir:
                save_path = os.path.join(self.save_dir, 'metrics_comparison.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Figure saved to {save_path}")
        
        # 关闭图表而不显示
        plt.close()


class ImageDisplay:
    def __init__(self, image_folder, label_folder, preds_folder, display_folder):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.preds_folder = preds_folder
        self.display_folder = display_folder
        
        if not os.path.exists(display_folder):
            os.mkdir(display_folder)
        
    def display_single_image(self, image_name):
        img = Image.open(os.path.join(self.image_folder, image_name))
        label = Image.open(os.path.join(self.label_folder, image_name))
        pred = Image.open(os.path.join(self.preds_folder, image_name))
        
        # 设置全局字体为Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title('Image', fontsize=14)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(label)
        plt.title('Label', fontsize=14)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(pred)
        plt.title('Prediction', fontsize=14)
        plt.axis('off')
        
        plt.tight_layout()
        # 保存文件名与原始图像 basename 一致
        basename = os.path.splitext(image_name)[0]
        plt.savefig(os.path.join(self.display_folder, f'{basename}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def display_all_images(self):
        images = sorted(os.listdir(self.image_folder))
        
        for image_name in tqdm(images):
            self.display_single_image(image_name)


if __name__ == '__main__':
    # config = DatasetConfig("../config/config_dataset.yaml")
    # color_mapping = {info['cls']: tuple(info['color']) for info in config.cls_dict.values()}
    # input_dir = r'E:\work\data\potsdam\test\preds'
    # output_dir = r'E:\work\data\potsdam\test\preds_color'
    # gray_color(input_dir, output_dir, color_mapping)
    
    image_fp = '../dataset/WHDLD/outputs/test/images'
    label_fp = '../dataset/WHDLD/outputs/test/labels'
    preds_fp = '../dataset/WHDLD/outputs/test/preds_color'
    save_fp = '../dataset/WHDLD/outputs/test/display'
    image_display = ImageDisplay(image_fp, label_fp, preds_fp, save_fp)

    image_display.display_all_images()
    
    