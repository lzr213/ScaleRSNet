# ScaleRSNet: Remote Sensing Image Semantic Segmentation

## Introduction
ScaleRSNet is a deep learning model designed for semantic segmentation of remote sensing imagery. It combines multi-scale context aggregation with attention mechanisms to effectively capture features at different scales.

## Model Architecture
ScaleRSNet uses a U-Net-like encoder-decoder structure with the following key components:
- Multi-Scale Context Aggregation (MSCA) blocks
- Channel and spatial attention mechanisms
- Deep MLP blocks for feature transformation
- Skip connections for preserving spatial details

## Training Process

### Configuration
Configure the model in `config/config_model.yaml`:

```yaml
model:
  name: "scalersnet"
  init_weights: true
  base_channels: 64  # Base channel count for ScaleRSNet
```

### Start Training
```bash
python trainer.py
```

The training process automatically sets batch size, learning rate, and other parameters according to the configuration file, and evaluates performance on the validation set.

### Testing Model Performance
```bash
# Using default weights
python test_model.py

# Specifying weights file
python test_model.py --weight ./outputs/your_model_path.pth
```

### Inference on New Images
```bash
# Using default weights
python predictor.py

# Specifying weights file
python predictor.py --weight ./outputs/your_model_path.pth
```

## Evaluation Metrics
The model is evaluated using:
- mPA: Mean Pixel Accuracy
- mIoU: Mean Intersection over Union
- FWIoU: Frequency Weighted IoU
- mPrecision: Mean Precision
- mRecall: Mean Recall
- mF1: Mean F1 Score
