# -*- coding: utf-8 -*-
# @Time    : 2024/8/26 0:17
# @Author  : xuxing
# @Site    : 
# @File    : torch_dataset.py
# @Software: PyCharm
from torch.utils.data import Dataset, DataLoader
import os
import glob
from PIL import Image
from loguru import logger
from pathlib import Path
import numpy as np
from datasets.transform import transform


class WhdldDataset(Dataset):
    """
    root是划分完数据集后的目录
    """
    def __init__(
            self,
            root: str,
            transform=None,
            image_suffix: str = '.jpg',
            label_suffix: str = '.png',
    ):
        self.images = glob.glob(os.path.join(root, f'images/*{image_suffix}'))
        self.labels = [
            os.path.join(os.path.join(root, 'labels'), Path(image).stem + label_suffix) for image in self.images
        ]
        
        logger.info(f'load {len(self.images)} samples of {root}')
        
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        image = np.array(Image.open(self.images[i]))
        label = np.array(Image.open(self.labels[i])) - 1
        
        res = {
            'image': image,
            'label': label
        }
        
        if self.transform is not None:
            res = self.transform(image=image, mask=label)
            res['label'] = res.pop('mask') # 标签字典改成label,方便统一
        
        return res


class EarthVQADataset(Dataset):
    """
    EarthVQA数据集加载器
    root是划分完数据集后的目录，包含images_png和masks_png子目录
    """
    def __init__(
            self,
            root: str,
            transform=None,
            image_suffix: str = '.png',
            label_suffix: str = '.png',
    ):
        self.images = glob.glob(os.path.join(root, f'images_png/*{image_suffix}'))
        self.labels = [
            os.path.join(os.path.join(root, 'masks_png'), Path(image).stem + label_suffix) for image in self.images
        ]
        
        logger.info(f'load {len(self.images)} samples of {root}')
        
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        try:
            image = np.array(Image.open(self.images[i]))
            label = np.array(Image.open(self.labels[i]))  # EarthVQA使用0-9的标签，无需减1
            
            res = {
                'image': image,
                'label': label
            }
            
            if self.transform is not None:
                res = self.transform(image=image, mask=label)
                res['label'] = res.pop('mask')  # 标签字典改成label，方便统一
            
            return res
        except Exception as e:
            logger.error(f"Error loading image {self.images[i]}: {str(e)}")
            # 返回一个空数据作为替代
            if i > 0:
                return self.__getitem__(0)  # 使用第一个样本替代
            else:
                # 如果是第一个样本就出错，创建一个假样本
                dummy_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
                dummy_label = np.zeros((1024, 1024), dtype=np.uint8)
                if self.transform:
                    res = self.transform(image=dummy_image, mask=dummy_label)
                    res['label'] = res.pop('mask')
                    return res
                else:
                    return {'image': dummy_image, 'label': dummy_label}


class Postdam(Dataset):
    """
    root是划分完数据集后的目录
    """
    
    def __init__(
            self,
            root: str,
            transform=None,
            image_suffix: str = '.tif',
            label_suffix: str = '.tif',
    ):
        self.images = glob.glob(os.path.join(root, f'images/*{image_suffix}'))
        self.labels = [
            os.path.join(os.path.join(root, 'labels_digital'), Path(image).stem + label_suffix) for image in self.images
        ]
        
        logger.info(f'load {len(self.images)} samples of {root}')
        
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        image = np.array(Image.open(self.images[i]))
        label = np.array(Image.open(self.labels[i]))
        
        res = {
            'image': image,
            'label': label
        }
        
        if self.transform is not None:
            res = self.transform(image=image, mask=label)
            res['label'] = res.pop('mask')  # 标签字典改成label,方便统一
        
        return res
    

if __name__ == '__main__':
    root = r'E:\work\data\potsdam\train'
    # dataset = Postdam(root,transform['train'])
    dataset = Postdam(root)
    data = dataset.__getitem__(1)
    
    print(data['image'].shape)
    print(data['label'].shape)
    print(np.unique(data['label']))

    
    
    # for i, data in enumerate(dataset):
    #     print(data['image'].shape)
    #     break