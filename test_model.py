#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Disable albumentations version check warning
import os
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"

from loguru import logger
import os
import torch
import argparse
from tqdm import tqdm
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import math
# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

from config.load_config import TrainConfig
from metrics import Metrics
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Test model performance on test set')
    parser.add_argument('--dataset', type=str, default='whdld', choices=['whdld', 'earthvqa'],
                        help='Select dataset: whdld or earthvqa')
    parser.add_argument('--config', type=str, default='./config/config_model.yaml',
                        help='Configuration file path, if not specified, automatically selected based on dataset')
    parser.add_argument('--weight', type=str, default='./outputs/ScaleRSNet/best_model.pth',
                        help='Model weight file path')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Test result save path, if not specified, use the directory where the weight is located')
    parser.add_argument('--result_file', type=str, default='test_results.txt',
                        help='Test result file name')
    
    args = parser.parse_args()
    
    # Select configuration file based on dataset
    if args.config is None:
        if args.dataset == 'earthvqa':
            args.config = './config/config_dataset_earthvqa.yaml'
        else:  # Default to whdld dataset
            args.config = './config/config_model.yaml'
    
    # If no output directory is specified, use the weight directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.weight)
    
    print(f"Using dataset: {args.dataset}")
    print(f"Config file: {args.config}")
    
    return args


# Add helper function to check for NaN values
def contains_nan(tensor):
    """Check if tensor contains NaN values"""
    if isinstance(tensor, torch.Tensor):
        return torch.isnan(tensor).any().item()
    elif isinstance(tensor, np.ndarray):
        return np.isnan(tensor).any()
    elif isinstance(tensor, (list, tuple)):
        return any(contains_nan(x) for x in tensor)
    elif isinstance(tensor, dict):
        return any(contains_nan(v) for v in tensor.values())
    elif isinstance(tensor, (int, float)):
        return math.isnan(tensor) if isinstance(tensor, float) else False
    return False


def safe_compute(value, default=0.0, name="value"):
    """Safely handle possible NaN values, return default value and log warning"""
    if isinstance(value, (int, float)) and (math.isnan(value) or math.isinf(value)):
        logger.warning(f"{name} is NaN or Inf, replaced with {default}")
        return default
    return value


def test_model(model_path, config_file, output_dir, result_file):
    """Load model weights and evaluate performance on test set"""
    # Set random seed to ensure reproducible results
    set_seed(42)
    
    # Create output directory
    test_output_dir = args.output_dir
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Record start time
    start_time = time.time()
    
    # Output test information
    print(f"Start testing model: {model_path}")
    print(f"Test start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load configuration and data
    train_config = TrainConfig(config_file)
    config = train_config.config
    
    # Get device
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU instead")
    
    # Create model and load weights
    model = train_config.model
    
    # Load model weights
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check if the checkpoint contains a state_dict key
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Try loading directly
            model.load_state_dict(checkpoint)
            
        print(f"Successfully loaded model weights from {model_path}")
    else:
        print(f"Error: Model weight file {model_path} does not exist")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Get test data loader
    test_loader = train_config.test_loader
    
    # Create metrics object
    num_classes = config['dataset']['num_classes']
    metrics = Metrics(num_classes, device)
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            images = data['image'].to(device)
            labels = data['label'].to(device).long()
            
            # Forward pass
            outputs = model(images)
            
            # Handle different output formats
            if isinstance(outputs, list):
                # For models with auxiliary outputs, use the last output
                outputs = outputs[-1]
            elif isinstance(outputs, tuple) and len(outputs) == 2:
                # For models with multiple outputs, use the first output
                mask_logits_per_layer, _ = outputs
                outputs = mask_logits_per_layer[-1]
            
            # Update metrics
            metrics.update(outputs, labels)
    
    # Compute metrics
    results = metrics.compute()
    
    # Save results to file
    result_path = os.path.join(test_output_dir, result_file)
    with open(result_path, 'w') as f:
        f.write(f"Test Results for {os.path.basename(model_path)}\n")
        f.write(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration: {config_file}\n\n")
        
        f.write("Overall Metrics:\n")
        f.write(f"Accuracy: {results['total_PA']:.4f}\n")
        f.write(f"Mean IoU: {results['mIoU']:.4f}\n")
        f.write(f"Frequency Weighted IoU: {results['FWIoU']:.4f}\n\n")
        
        f.write("Class-wise Metrics:\n")
        for i in range(num_classes):
            f.write(f"Class {i}:\n")
            f.write(f"  Precision: {results['precision'][i]:.4f}\n")
            f.write(f"  Recall: {results['recall'][i]:.4f}\n")
            f.write(f"  F1 Score: {results['f1'][i]:.4f}\n")
            f.write(f"  IoU: {results['IoU'][i]:.4f}\n")
            f.write(f"  Pixel Accuracy: {results['PA'][i]:.4f}\n\n")
        
        f.write("Macro and Micro Average Metrics:\n")
        f.write(f"Macro Precision: {results['macro_precision']:.4f}\n")
        f.write(f"Macro Recall: {results['macro_recall']:.4f}\n")
        f.write(f"Macro F1: {results['macro_f1']:.4f}\n")
        f.write(f"Micro Precision: {results['micro_precision']:.4f}\n")
        f.write(f"Micro Recall: {results['micro_recall']:.4f}\n")
        f.write(f"Micro F1: {results['micro_f1']:.4f}\n")
    
    # Print results
    print(f"Test results saved to {result_path}")
    print(f"Accuracy: {results['total_PA']:.4f}")
    print(f"Mean IoU: {results['mIoU']:.4f}")
    
    # Visualize class-wise metrics
    visualize_metrics(results, test_output_dir, num_classes)
    
    return results


def visualize_metrics(results, output_dir, num_classes):
    """Visualize test metrics with charts"""
    # Class-wise IoU visualization
    plt.figure(figsize=(12, 6))
    plt.bar(range(num_classes), results['IoU'], color='skyblue')
    plt.axhline(y=results['mIoU'], color='r', linestyle='-', label=f'Mean IoU: {results["mIoU"]:.4f}')
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('IoU', fontsize=14)
    plt.title('Class-wise IoU', fontsize=16)
    plt.xticks(range(num_classes))
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_iou.png'), dpi=300)
    
    # Class-wise F1 Score visualization
    plt.figure(figsize=(12, 6))
    plt.bar(range(num_classes), results['f1'], color='lightgreen')
    plt.axhline(y=results['macro_f1'], color='r', linestyle='-', label=f'Macro F1: {results["macro_f1"]:.4f}')
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.title('Class-wise F1 Score', fontsize=16)
    plt.xticks(range(num_classes))
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_f1.png'), dpi=300)
    
    # Precision and Recall visualization
    plt.figure(figsize=(12, 6))
    x = np.arange(num_classes)
    width = 0.35
    plt.bar(x - width/2, results['precision'], width, label='Precision', color='coral')
    plt.bar(x + width/2, results['recall'], width, label='Recall', color='cornflowerblue')
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Class-wise Precision and Recall', fontsize=16)
    plt.xticks(x)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall.png'), dpi=300)
    
    print(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    test_model(args.weight, args.config, args.output_dir, args.result_file)