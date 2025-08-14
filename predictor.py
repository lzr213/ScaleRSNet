# -*- coding: utf-8 -*-
# @Time    : 2024/8/31 22:17
# @Author  : xuxing
# @Site    : 
# @File    : predictor.py
# @Software: PyCharm

import os
from PIL import Image
import torch
import numpy as np
import yaml
from pathlib import Path

from config.load_config import get_train_config
from datasets.transform import transform, earthvqa_transform
from tqdm import tqdm
import argparse

class Predictor:
    def __init__(self,
                 weight,
                 dataset_type='whdld',
                 yaml_fp=None,
                 dataset_config_fp=None,
                 transform_type=None,
                 device='cuda'):
        """
        Initialize predictor
        
        Args:
            weight: Model weight file path
            dataset_type: Dataset type, can be 'whdld' or 'earthvqa'
            yaml_fp: Model configuration file path, if None, automatically selected based on dataset_type
            dataset_config_fp: Dataset configuration file path, if None, automatically selected based on dataset_type
            transform_type: Transform type, if None, automatically selected based on dataset_type
            device: Device, 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dataset_type = dataset_type
        
        # Automatically select configuration files and transforms based on dataset type
        if yaml_fp is None:
            if dataset_type == 'earthvqa':
                yaml_fp = './config/config_dataset_earthvqa.yaml'
            else:  # Default to whdld
                yaml_fp = './config/config_model.yaml'
        
        if dataset_config_fp is None:
            if dataset_type == 'earthvqa':
                dataset_config_fp = './config/config_dataset_earthvqa.yaml'
            else:  # Default to whdld
                dataset_config_fp = './config/config_dataset.yaml'
        
        # Select transform based on dataset type
        if transform_type is None:
            if dataset_type == 'earthvqa':
                self.transform = earthvqa_transform['test']
                print("Using EarthVQA dataset transform (1024x1024)")
            else:  # Default to whdld
                self.transform = transform['test']
                print("Using WHDLD dataset transform (512x512)")
        else:
            self.transform = transform_type
        
        print(f"Using model configuration file: {yaml_fp}")
        print(f"Using dataset configuration file: {dataset_config_fp}")
        
        # Load model
        # Get configuration
        config = None
        with open(yaml_fp, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Set model name to ScaleRSNet
        model_name = 'scalersnet'
        print(f"Using model type: {model_name}")
        
        # Update configuration with model name
        if config:
            config['model']['name'] = model_name
            
        # Create model with updated configuration
        train_config = get_train_config(yaml_fp)
        self.model = train_config.model
        
        # Load model weights
        print(f"Loading model weights from: {weight}")
        if os.path.exists(weight):
            checkpoint = torch.load(weight, map_location=self.device)
            
            # Check if the checkpoint contains a state_dict key
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                # Try loading directly
                self.model.load_state_dict(checkpoint)
                
            print(f"Successfully loaded model weights")
        else:
            print(f"Error: Weight file {weight} does not exist")
            raise FileNotFoundError(f"Weight file not found: {weight}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Load dataset configuration to get color mapping
        with open(dataset_config_fp, 'r', encoding='utf-8') as f:
            dataset_config = yaml.safe_load(f)
            
        # Get color mapping from dataset configuration
        self.color_mapping = {}
        if 'cls_dict' in dataset_config:
            for class_name, class_info in dataset_config['cls_dict'].items():
                self.color_mapping[class_info['cls']] = tuple(class_info['color'])
        else:
            # Default color mapping if not found in configuration
            self.color_mapping = {
                0: (255, 255, 255),  # Bare soil - white
                1: (255, 0, 0),      # Building - red
                2: (255, 255, 0),    # Pavement - yellow
                3: (0, 0, 255),      # Road - blue
                4: (0, 255, 0),      # Vegetation - green
                5: (0, 255, 255)     # Water - cyan
            }
        
        print("Color mapping loaded:")
        for cls_id, color in self.color_mapping.items():
            print(f"Class {cls_id}: RGB{color}")

    def predict(self, image_path, output_path=None):
        """
        Predict segmentation for a single image
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the output image, if None, will be automatically generated
            
        Returns:
            Tuple of (prediction array, colored segmentation image)
        """
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} does not exist")
            return None, None
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Generate output path if not provided
        if output_path is None:
            image_name = os.path.basename(image_path)
            image_name_without_ext = os.path.splitext(image_name)[0]
            output_dir = os.path.dirname(image_path)
            output_path = os.path.join(output_dir, f"{image_name_without_ext}_pred.png")
        
        # Apply transform
        transformed = self.transform(image=np.array(image))
        input_tensor = transformed["image"].unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # Handle different output formats
            if isinstance(output, list):
                # For models with auxiliary outputs, use the last output
                output = output[-1]
            elif isinstance(output, tuple) and len(output) == 2:
                # For models with multiple outputs, use the first output
                mask_logits_per_layer, _ = output
                output = mask_logits_per_layer[-1]
            
            # Get prediction
            _, pred = torch.max(output, 1)
            pred = pred.cpu().numpy()[0]  # Shape: (H, W)
        
        # Create colored segmentation image
        colored_pred = self.create_colored_segmentation(pred)
        
        # Save prediction if output path is provided
        if output_path:
            colored_pred.save(output_path)
            print(f"Segmentation saved to: {output_path}")
        
        return pred, colored_pred
    
    def predict_batch(self, image_dir, output_dir=None, extensions=('.jpg', '.jpeg', '.png')):
        """
        Predict segmentation for all images in a directory
        
        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save output images, if None, will use image_dir + '_pred'
            extensions: Tuple of valid image extensions to process
            
        Returns:
            List of output image paths
        """
        # Check if image directory exists
        if not os.path.exists(image_dir):
            print(f"Error: Image directory {image_dir} does not exist")
            return []
        
        # Generate output directory if not provided
        if output_dir is None:
            output_dir = image_dir + '_pred'
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files in the directory
        image_files = []
        for ext in extensions:
            image_files.extend(list(Path(image_dir).glob(f"*{ext}")))
        
        if not image_files:
            print(f"No images found in {image_dir} with extensions {extensions}")
            return []
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        output_paths = []
        for image_path in tqdm(image_files, desc="Processing images"):
            image_name = os.path.basename(image_path)
            image_name_without_ext = os.path.splitext(image_name)[0]
            output_path = os.path.join(output_dir, f"{image_name_without_ext}_pred.png")
            
            _, _ = self.predict(str(image_path), output_path)
            output_paths.append(output_path)
        
        print(f"Processed {len(output_paths)} images, results saved to {output_dir}")
        return output_paths
    
    def create_colored_segmentation(self, pred_array):
        """
        Create a colored segmentation image from prediction array
        
        Args:
            pred_array: Prediction array with class indices
            
        Returns:
            PIL Image with colored segmentation
        """
        # Create RGB array for the segmentation
        height, width = pred_array.shape
        colored_segmentation = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Fill with colors based on class indices
        for cls_id, color in self.color_mapping.items():
            mask = (pred_array == cls_id)
            colored_segmentation[mask] = color
        
        # Convert to PIL Image
        return Image.fromarray(colored_segmentation)


def parse_args():
    parser = argparse.ArgumentParser(description='Predict segmentation for images')
    parser.add_argument('--dataset', type=str, default='whdld', choices=['whdld', 'earthvqa'],
                        help='Select dataset: whdld or earthvqa')
    parser.add_argument('--config', type=str, default=None,
                        help='Configuration file path, if not specified, automatically selected based on dataset')
    parser.add_argument('--dataset_config', type=str, default=None,
                        help='Dataset configuration file path, if not specified, automatically selected based on dataset')
    parser.add_argument('--weight', type=str, default='./outputs/ScaleRSNet/best_model.pth',
                        help='Model weight file path')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path or directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path or directory, if not specified, will be automatically generated')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda or cpu')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    # Create predictor
    predictor = Predictor(
        weight=args.weight,
        dataset_type=args.dataset,
        yaml_fp=args.config,
        dataset_config_fp=args.dataset_config,
        device=args.device
    )
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Single image prediction
        print(f"Processing single image: {args.input}")
        pred, colored_pred = predictor.predict(args.input, args.output)
        print("Prediction complete")
    elif os.path.isdir(args.input):
        # Batch prediction
        print(f"Processing directory: {args.input}")
        output_paths = predictor.predict_batch(args.input, args.output)
        print(f"Batch prediction complete, processed {len(output_paths)} images")
    else:
        print(f"Error: Input path {args.input} does not exist")