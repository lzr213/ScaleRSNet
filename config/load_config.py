# -*- coding: utf-8 -*-
# @Time    : 2024/8/31 23:55
# @Author  : xuxing
# @Site    : 
# @File    : load_config.py
# @Software: PyCharm

import yaml
import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np

# Load configuration file
from torch.utils.data import DataLoader

# Define DataLoader worker initialization function
def worker_init_function(worker_id):
    # Set different but deterministic seed for each worker
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

from datasets.transform import transform, earthvqa_transform
from datasets.torch_dataset import WhdldDataset, Postdam, EarthVQADataset
from models.scalersnet import ScaleRSNet


class TrainConfig:
    def __init__(self, yaml_file):
        self.config = self.load_config(yaml_file)
        self.device = self.config['training']['device']
        self.model = None
        self.transform = None
        self.optimizer = None
        self.loss_function = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self.setup()


    def load_config(self, yaml_file):
        with open(yaml_file, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config

    def setup(self):
        self.train_loader = self.create_data_loader(mode='train')
        self.val_loader = self.create_data_loader(mode='val')
        self.test_loader = self.create_data_loader(mode='test')
        self.create_model()
        self.create_optimizer()
        self.create_loss_function()
        self.create_scheduler()

    def summary(self):
        return {
            'train_data': self.train_loader,
            'val_data': self.val_loader,
            'test_data': self.test_loader,
            'model': self.model,
            'optimizer': self.optimizer,
            'loss': self.loss_function,
            'scheduler': self.scheduler
        }
    
    def create_data_loader(self, mode='train'):
        # Set whether to shuffle the data
        shuffle = True if mode == 'train' else False
        
        # Ensure fixed data during testing
        if mode == 'test':
            # Set PyTorch random seed to ensure reproducibility of DataLoader workers
            generator = torch.Generator()
            generator.manual_seed(42)
            # Use globally defined worker initialization function
            worker_init_fn = worker_init_function
        else:
            generator = None
            worker_init_fn = None
            
        config = self.config
        name = config['dataset']['name']
        
        # Use corresponding transform for each mode
        if name == 'potsdam':
            dataset = Postdam(self.config['dataset'][mode], transform=transform[mode])
        elif name == 'whdld':
            dataset = WhdldDataset(self.config['dataset'][mode], transform=transform[mode])
        elif name == 'earthvqa':
            # Use transform specifically designed for EarthVQA
            dataset = EarthVQADataset(self.config['dataset'][mode], transform=earthvqa_transform[mode])
        else:
            dataset = None
            
        # For Windows multiprocessing, consider reducing worker count or setting to single process if issues occur
        num_workers = 0 if mode == 'test' else self.config['dataset']['num_workers']
        
        data_loader = DataLoader(
            dataset, 
            batch_size=self.config['training']['batch_size'],
            num_workers=num_workers, 
            shuffle=shuffle,
            generator=generator if mode == 'test' else None,
            worker_init_fn=worker_init_fn if mode == 'test' and num_workers > 0 else None,
            drop_last=False,  # Ensure the last incomplete batch is not dropped
            pin_memory=True   # Speed up data transfer to GPU
        )
        return data_loader
        
    def create_model(self):
        model_config = self.config['model']
        name = model_config['name']
        num_classes = self.config['dataset']['num_classes']
        input_channels = self.config['dataset']['in_channels']
        pretrained = model_config.get('pretrained', False)  # Get pretrained parameter, default False

        if name == "scalersnet":
            base_channels = model_config.get('base_channels', 64)  # Default base channel count is 64
            self.model = ScaleRSNet(in_channels=input_channels, out_channels=num_classes, 
                               base_channels=base_channels).to(self.device)
        else:
            raise ValueError(f"Unknown model name: {name}")
        
        return self.model

    def create_optimizer(self):
        training_config = self.config['training']
        optimizer_name = training_config['optimizer']
        learning_rate = training_config['learning_rate']

        if optimizer_name == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def create_loss_function(self):
        loss_function_name = self.config['training']['loss_function']

        if loss_function_name == "cross_entropy":
            self.loss_function = nn.CrossEntropyLoss()
        elif loss_function_name == "mse":
            self.loss_function = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_function_name}")

    def create_scheduler(self):
        scheduler_config = self.config['training']['scheduler']
        scheduler_type = scheduler_config.get("type")

        if scheduler_type == "step_lr":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config["step_size"],
                gamma=scheduler_config["gamma"]
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")


class DatasetConfig:
    def __init__(self, yaml_file):
        self.config = self.load_config(yaml_file)
        self.cls_dict = self.config.get('cls_dict', {})
    
    def load_config(self, yaml_file):
        with open(yaml_file, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    
    def get_class_info(self):
        return self.cls_dict
    
    def get_class_names(self):
        return list(self.cls_dict.keys())
    
    def get_color_mapping(self):
        return {info['cls']: tuple(info['color']) for info in self.cls_dict.values()}
    
    def get_class_ids(self):
        return {cls_info['color']: cls_info['cls'] for cls_info in self.cls_dict.values()}
    
    def print_summary(self):
        print("Class Dictionary:")
        for class_name, class_info in self.cls_dict.items():
            print(f"{class_name}: Class ID = {class_info['cls']}, Color = {class_info['color']}")



def get_train_config(config_file = 'config_model.yaml'):
    return TrainConfig(config_file)

def get_dataset_config(config_file = 'config_dataset.yaml'):
    return DatasetConfig(config_file)
    


if __name__ == "__main__":
    config_file = 'config_model.yaml'  # Replace with your YAML file path
    train_config = TrainConfig(config_file)

    # Use configuration file 'config_dataset.yaml'
    yaml_file = 'config_dataset.yaml'
    dataset_config = DatasetConfig(yaml_file)

    # Print all class information
    print("Testing DatasetConfig:")
    dataset_config.print_summary()

    # Get class information
    class_info = dataset_config.get_class_info()
    print("\nClass Info:", class_info)

    # Get class names
    class_names = dataset_config.get_class_names()
    print("\nClass Names:", class_names)


