# -*- coding: utf-8 -*-
# @Time    : 2024/8/27 0:07
# @Author  : xuxing
# @Site    : 
# @File    : trainer.py
# @Software: PyCharm

# Disable albumentations version check warning
import os
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
from loguru import logger
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config.load_config import TrainConfig
from datasets.transform import transform
from datasets.torch_dataset import WhdldDataset
from models.scalersnet import ScaleRSNet
from utils import parse_args, setup_logging, set_seed
from tqdm import tqdm
from datetime import datetime
import time
from metrics import Metrics
from data_postprocess.visualization import LossVisualizer
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self,
                 model,
                 num_classes,
                 train_loader,  
                 val_loader,
                 test_loader,
                 criterion,
                 optimizer,
                 scheduler,
                 device,
                 resume,
                 output_dir):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.resume = resume
        self.model_name = ""
        # Get model name as part of the folder name
        model_class_name = model.__class__.__name__
        self.save_dir = os.path.join(output_dir, f"{model_class_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        # Initialize training state variables
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.training_history = []
        self.val_loss_history = []
        self.train_loss_history = []
        # Save models from recent epochs
        self.keep_last_n_epochs = 3  # Keep models from the last 3 epochs
        
        self.init_settings()
        
    # Load the model if a resume path is specified
    def init_settings(self):
        if self.resume:
            self.load_model(self.resume)
            self.model_name = os.path.basename(self.resume)
            
        logger.info('init output dirs ... ')
        os.makedirs(self.save_dir, exist_ok=True)
        
        logger.info('init loggings ... ')
        setup_logging(self.save_dir)
        
        logger.info('init Metrics ... ')
        self.metrics = Metrics(self.num_classes, self.device)



    def train(self, epochs):
        # Record training start time
        start_time = time.time()
        logger.info(f"Training start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            # Record the start time of each epoch
            epoch_start_time = time.time()
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            self.model.train()
            train_loss = 0.0

            for i, data in enumerate(self.train_loader):
                images = data['image']
                labels = data['label']
                
                # Move data to the specified device
                images = images.to(self.device)
                labels = labels.to(self.device).long()  # Ensure labels are Long type
                
                # Clear gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                if isinstance(outputs, list):  # Check if it's DSNet's auxiliary loss output
                    # For DSNet model, output is a list containing multiple predictions
                    loss = 0
                    for output in outputs:
                        loss += self.criterion(output, labels)
                    loss /= len(outputs)  # Average loss
                elif isinstance(outputs, tuple) and len(outputs) == 2:  # Check if it's EOMT model output
                    # For EOMT model, output is (mask_logits_per_layer, class_logits_per_layer)
                    mask_logits_per_layer, class_logits_per_layer = outputs
                    loss = 0
                    # Use the last layer's mask_logits as the main output
                    main_output = mask_logits_per_layer[-1]
                    # Calculate loss for each query and average
                    for q in range(main_output.shape[1]):
                        loss += self.criterion(main_output[:, q], labels)
                    loss /= main_output.shape[1]  # Average loss
                else:
                    # Regular single output model
                    loss = self.criterion(outputs, labels)
                
                # Backpropagation and optimization
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                
                # Print training information every 10 batches
                if (i + 1) % 10 == 0:
                    logger.info(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(self.train_loader)}], Loss: {loss.item():.4f}')
            
            # Calculate average training loss
            train_loss = train_loss / len(self.train_loader)
            self.train_loss_history.append(train_loss)
            
            # Evaluate model on validation set
            val_loss, val_accuracy, val_metrics = self.evaluate()
            self.val_loss_history.append(val_loss)
            
            # Update learning rate
            self.scheduler.step()
            
            # Record current learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Record training history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'learning_rate': current_lr,
                'metrics': val_metrics
            })
            
            # Save model if current performance is the best
            if val_accuracy > self.best_val_acc:
                self.best_val_acc = val_accuracy
                self.save_model('best_model.pth')
                logger.info(f'Saved best model, accuracy: {val_accuracy:.4f}')
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(f'checkpoint_epoch_{epoch + 1}.pth', keep_previous=True)
                logger.info(f'Saved checkpoint for epoch {epoch + 1}')
            
            # Save model for each epoch, but only keep the most recent ones
            self.save_model(f'epoch_{epoch + 1}.pth', keep_previous=True)
            
            # Clean up old epoch models, only keep the most recent keep_last_n_epochs
            if epoch + 1 > self.keep_last_n_epochs:
                old_epoch = epoch + 1 - self.keep_last_n_epochs
                old_model_path = os.path.join(self.save_dir, f'epoch_{old_epoch}.pth')
                if os.path.exists(old_model_path):
                    os.remove(old_model_path)
                    logger.info(f'Deleted old epoch model: epoch_{old_epoch}.pth')
            
            # Calculate and record time spent on each epoch
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            logger.info(f'Epoch {epoch + 1} completed, time: {epoch_time:.2f} seconds')
            logger.info(f'Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}')
            logger.info(f'Current learning rate: {current_lr:.6f}')
            
            # Print current metrics on validation set
            logger.info("Validation metrics:")
            for key, value in val_metrics.items():
                if isinstance(value, float):
                    logger.info(f"{key}: {value:.4f}")
        
        # Training ends, record total time
        end_time = time.time()
        total_time = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"Training ends, total time: {int(hours)} hours {int(minutes)} minutes {seconds:.2f} seconds")
        
        # Save final model
        self.save_model('final_model.pth')
        
        # Visualize training history
        self.visualize_training_history()
        
        # Evaluate final model on test set
        logger.info("Evaluating final model on test set...")
        test_results = self.test_evaluate()
        self.save_test_results(test_results)
        
        return self.training_history

    def evaluate(self):
        self.model.eval()
        val_loss = 0.0
        self.metrics.reset()
        
        with torch.no_grad():
            for data in self.val_loader:
                images = data['image'].to(self.device)
                labels = data['label'].to(self.device).long()  # 确保标签是Long类型
                
                outputs = self.model(images)
                
                # 处理DSNet的输出
                if isinstance(outputs, list):
                    # 使用最后一个输出（主输出）进行评估
                    main_output = outputs[1]  # DSNet的主输出通常是第二个元素
                    loss = self.criterion(main_output, labels)
                    # 获取预测类别
                    _, preds = torch.max(main_output, dim=1)
                    # 更新混淆矩阵
                    self.metrics.sample_add(labels, preds)
                # 处理EOMT模型的输出
                elif isinstance(outputs, tuple) and len(outputs) == 2:
                    mask_logits_per_layer, class_logits_per_layer = outputs
                    # 使用最后一层的mask_logits作为主要输出
                    main_output = mask_logits_per_layer[-1]
                    # 计算每个查询的损失并求平均
                    loss = 0
                    for q in range(main_output.shape[1]):
                        loss += self.criterion(main_output[:, q], labels)
                    loss /= main_output.shape[1]
                    
                    # 获取预测类别：对所有查询的预测进行平均，然后取最大值
                    avg_preds = torch.mean(main_output, dim=1)  # 对查询维度取平均
                    _, preds = torch.max(avg_preds, dim=1)
                    # 更新混淆矩阵
                    self.metrics.sample_add(labels, preds)
                else:
                    loss = self.criterion(outputs, labels)
                    # 获取预测类别
                    _, preds = torch.max(outputs, dim=1)
                    # 更新混淆矩阵
                    self.metrics.sample_add(labels, preds)
                
                val_loss += loss.item()
        
        # 计算平均验证损失
        val_loss = val_loss / len(self.val_loader)
        
        # 获取评估指标
        metrics_dict = self.metrics.compute()
        accuracy = metrics_dict['total_PA']  # 使用total_PA作为准确率
        
        return val_loss, accuracy, metrics_dict

    def test_evaluate(self):
        # 确保使用最佳模型进行测试
        best_model_path = os.path.join(self.save_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            self.load_model(best_model_path)
            logger.info("Loaded best model for test evaluation")
        
        self.model.eval()
        self.metrics.reset()
        
        with torch.no_grad():
            for data in tqdm(self.test_loader, desc="Testing evaluation"):
                images = data['image'].to(self.device)
                labels = data['label'].to(self.device).long()  # 确保标签是Long类型
                
                outputs = self.model(images)
                
                # 处理DSNet的输出
                if isinstance(outputs, list):
                    # 使用最后一个输出（主输出）进行评估
                    main_output = outputs[1]  # DSNet的主输出通常是第二个元素
                    # 获取预测类别
                    _, preds = torch.max(main_output, dim=1)
                    # 更新混淆矩阵
                    self.metrics.sample_add(labels, preds)
                # 处理EOMT模型的输出
                elif isinstance(outputs, tuple) and len(outputs) == 2:
                    mask_logits_per_layer, class_logits_per_layer = outputs
                    # 使用最后一层的mask_logits作为主要输出
                    main_output = mask_logits_per_layer[-1]
                    
                    # 获取预测类别：对所有查询的预测进行平均，然后取最大值
                    avg_preds = torch.mean(main_output, dim=1)  # 对查询维度取平均
                    _, preds = torch.max(avg_preds, dim=1)
                    # 更新混淆矩阵
                    self.metrics.sample_add(labels, preds)
                else:
                    # 获取预测类别
                    _, preds = torch.max(outputs, dim=1)
                    # 更新混淆矩阵
                    self.metrics.sample_add(labels, preds)
        
        # 获取评估指标
        metrics_dict = self.metrics.compute()
        
        logger.info("Test set evaluation completed")
        logger.info(f"Accuracy: {metrics_dict['total_PA']:.4f}")
        logger.info(f"Average IoU: {metrics_dict['mIoU']:.4f}")
        
        return metrics_dict

    def save_model(self, filename, keep_previous=False):
        """
        保存模型检查点
        
        Args:
            filename (str): 保存的文件名
            keep_previous (bool): 是否保留之前的检查点，默认为False
        """
        # 处理文件保存逻辑
        save_path = os.path.join(self.save_dir, filename)
        
        # 如果是周期性检查点，总是保留
        if 'checkpoint_epoch_' in filename:
            # 确保是新文件名，不覆盖现有文件
            keep_previous = True
        # 如果不保留之前的检查点且文件已存在，则删除
        elif not keep_previous and os.path.exists(save_path):
            os.remove(save_path)
            logger.info(f"Deleted old model file: {save_path}")
        
        # 保存模型状态
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'training_history': self.training_history,
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history
        }
        
        # 保存到指定路径
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, checkpoint_path):
        """
        加载模型检查点
        
        Args:
            checkpoint_path (str): 检查点文件路径
        """
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file does not exist: {checkpoint_path}")
            return
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 恢复模型状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        
        # 恢复训练历史（如果存在）
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        if 'train_loss_history' in checkpoint:
            self.train_loss_history = checkpoint['train_loss_history']
        if 'val_loss_history' in checkpoint:
            self.val_loss_history = checkpoint['val_loss_history']
        
        logger.info(f"Loaded model from {checkpoint_path}, current epoch: {self.current_epoch}")

    def visualize_training_history(self):
        """可视化训练历史"""
        # 创建可视化器
        visualizer = LossVisualizer(self.save_dir)
        
        # 提取训练历史数据
        epochs = list(range(1, len(self.training_history) + 1))
        train_losses = [record['train_loss'] for record in self.training_history]
        val_losses = [record['val_loss'] for record in self.training_history]
        val_accuracies = [record['val_accuracy'] for record in self.training_history]
        learning_rates = [record['learning_rate'] for record in self.training_history]
        
        # 绘制损失曲线
        visualizer.plot_losses(epochs, train_losses, val_losses)
        
        # 绘制准确率曲线
        visualizer.plot_accuracy(epochs, val_accuracies)
        
        # 绘制学习率曲线
        visualizer.plot_learning_rate(epochs, learning_rates)
        
        # 绘制每个epoch的耗时
        epoch_times = []
        for i in range(len(self.training_history) - 1):
            if i + 1 < len(self.training_history):
                time_diff = self.training_history[i + 1].get('timestamp', 0) - self.training_history[i].get('timestamp', 0)
                epoch_times.append(time_diff if time_diff > 0 else 0)
        
        # 如果有记录epoch时间，则绘制
        if epoch_times:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(epoch_times) + 1), epoch_times, 'g-', marker='o')
            plt.title('Epoch execution time', fontsize=14)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Time (seconds)', fontsize=12)
            plt.grid(True)
        
        # 添加数值标签
        for i, v in enumerate(epoch_times):
            plt.text(i+1, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
        
        time_path = os.path.join(self.save_dir, 'epoch_times.png')
        plt.savefig(time_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Epoch time visualization saved to {time_path}")
        
        # 绘制训练和验证损失曲线
        plt.figure(figsize=(10, 6))
        epochs = list(range(1, len(self.train_loss_history) + 1))
        
        plt.plot(epochs, self.train_loss_history, 'b-', label='Training Loss')
        plt.plot(epochs, self.val_loss_history, 'r-', label='Validation Loss')
        
        plt.title('Training and Validation Loss over Epochs', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True)
        plt.legend(fontsize=12)
        
        loss_path = os.path.join(self.save_dir, 'loss_history.png')
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Loss visualization saved to {loss_path}")
    
    def save_test_results(self, results):
        """保存测试结果到TXT文件"""
        # 使用metrics类中的方法保存指标
        txt_path = os.path.join(self.save_dir, 'test_metrics.txt')
        self.metrics.save_metrics(results, txt_path)
        
        # 特别记录宏平均和微平均指标
        logger.info("\nMacro and Micro Average Metrics:")
        logger.info(f"Macro Precision: {results['macro_precision']:.4f}")
        logger.info(f"Macro Recall: {results['macro_recall']:.4f}")
        logger.info(f"Macro F1: {results['macro_f1']:.4f}")
        logger.info(f"Micro Precision: {results['micro_precision']:.4f}")
        logger.info(f"Micro Recall: {results['micro_recall']:.4f}")
        logger.info(f"Micro F1: {results['micro_f1']:.4f}")
        
def main(config_file, dataset_config_file=None):
    # 设置随机种子为42，以便于结果复现一致
    set_seed(42)
    
    # 如果指定了dataset_config_file，使用指定的数据集配置文件
    # 否则使用model配置文件中的默认数据集设置
    train_config = TrainConfig(config_file)
    config = train_config.config
    
    # 初始化模型、损失函数、优化器
    device = train_config.device
    train_loader = train_config.train_loader
    val_loader = train_config.val_loader
    test_loader = train_config.test_loader  # 获取测试数据加载器
    model = train_config.model
    criterion = train_config.loss_function
    optimizer = train_config.optimizer
    scheduler = train_config.scheduler
    
    # 创建 Trainer 实例
    trainer = Trainer(model,
                      config['dataset']['num_classes'],
                      train_loader,
                      val_loader,
                      test_loader,
                      criterion,
                      optimizer,
                      scheduler,
                      device,
                      config['training']['resume'],
                      config['save']['output_dir'])
    
    # 开始训练
    trainer.train(config['training']['epochs'])


if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练语义分割模型')
    parser.add_argument('--dataset', type=str, default='whdld', choices=['whdld', 'earthvqa'],
                        help='选择数据集: whdld 或 earthvqa')
    parser.add_argument('--model', type=str, default='sernet',
                        help='指定使用的模型配置文件，例如 dsnet 将使用 config_model_dsnet.yaml，sernet 将使用 config_model_sernet.yaml')
    args = parser.parse_args()
    
    # 根据数据集选择对应的配置文件
    if args.model:
        # 使用指定的模型配置文件，并根据数据集选择对应的配置
        if args.dataset == 'earthvqa':
            config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', f'config_model_{args.model}_earthvqa.yaml')
        else:  # 默认使用whdld
            config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', f'config_model_{args.model}.yaml')
    elif args.dataset == 'earthvqa':
        # 使用绝对路径来确保文件能被找到
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'config_model_earthvqa.yaml')
    else:  # 默认使用 whdld 数据集
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'config_model.yaml')
    
    print(f"Using dataset: {args.dataset}")
    print(f"Config file: {config_file}")
    
    main(config_file)
    
    