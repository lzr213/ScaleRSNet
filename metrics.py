# -*- coding: utf-8 -*-
# @Time    : 2024/8/31 10:45
# @Author  : xuxing
# @Site    : 
# @File    : metrics.py.py
# @Software: PyCharm
import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import time
import math
from loguru import logger

# 添加安全处理NaN的辅助函数
def safe_mean(tensor, default=0.0):
    """安全计算平均值，处理可能的NaN"""
    if tensor.numel() == 0:
        return torch.tensor(default, device=tensor.device)
    
    # 移除NaN和Inf值
    valid_tensor = tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)]
    if valid_tensor.numel() == 0:
        return torch.tensor(default, device=tensor.device)
    
    return valid_tensor.mean()

def safe_sum(tensor, default=0.0):
    """安全计算总和，处理可能的NaN"""
    if tensor.numel() == 0:
        return torch.tensor(default, device=tensor.device)
    
    # 移除NaN和Inf值
    valid_tensor = tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)]
    if valid_tensor.numel() == 0:
        return torch.tensor(default, device=tensor.device)
    
    return valid_tensor.sum()

def safe_divide(numerator, denominator, default=0.0):
    """安全除法，处理除数为0或NaN的情况"""
    if isinstance(numerator, torch.Tensor):
        result = torch.zeros_like(numerator, dtype=torch.float)
        valid_mask = (denominator != 0) & ~torch.isnan(denominator) & ~torch.isinf(denominator) & ~torch.isnan(numerator) & ~torch.isinf(numerator)
        if valid_mask.any():
            result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        else:
            result = torch.full_like(result, default)
        return result
    else:
        if denominator == 0 or math.isnan(denominator) or math.isinf(denominator) or math.isnan(numerator) or math.isinf(numerator):
            return default
        return numerator / denominator


class Metrics:
    def __init__(self, class_num, device = 'cpu'):
        self.class_num = class_num
        self.device = device
        self.cfm = self.cfm_init(self.class_num)
        
    def cfm_init(self, class_num):
        return torch.zeros(size=(class_num, class_num), dtype=torch.int).to(self.device)
    
    def update(self):
        self.cfm = self.cfm_init(self.class_num)
    
    # 添加reset方法作为update的别名，以保持与trainer.py的兼容性
    def reset(self):
        """reset方法是update方法的别名，用于重置混淆矩阵"""
        self.update()
    
    def sample_add(self, true_vector, pre_vector):
        try:
            true_vector = true_vector.flatten()
            pre_vector = pre_vector.flatten()
            
            # 检查是否有NaN值
            if torch.isnan(true_vector).any() or torch.isnan(pre_vector).any():
                logger.warning("检测到NaN值，跳过此样本")
                return
            
            # 检查是否有Inf值
            if torch.isinf(true_vector).any() or torch.isinf(pre_vector).any():
                logger.warning("检测到Inf值，跳过此样本")
                return
            
            mask = (true_vector >= 0) & (true_vector < self.class_num)
            self.cfm += torch.bincount(self.class_num * true_vector[mask] + pre_vector[mask],
                                    minlength=self.class_num ** 2).reshape(self.class_num, self.class_num).to(self.device)
        except Exception as e:
            logger.error(f"更新混淆矩阵时出错: {str(e)}")
    
    def acc(self):
        """计算每类准确率和总体准确率"""
        try:
            row_sum = torch.sum(self.cfm, dim=1).float()
            per_class_PA = safe_divide(torch.diag(self.cfm).float(), row_sum)
            total_PA = safe_divide(torch.diag(self.cfm).sum().float(), self.cfm.sum().float())
            return per_class_PA, total_PA
        except Exception as e:
            logger.error(f"计算准确率时出错: {str(e)}")
            return torch.zeros(self.class_num, device=self.device), torch.tensor(0.0, device=self.device)
    
    def iou(self):
        """计算每类IoU和平均IoU"""
        try:
            diag = torch.diag(self.cfm).float()
            row_sum = torch.sum(self.cfm, dim=1).float()
            col_sum = torch.sum(self.cfm, dim=0).float()
            denominator = row_sum + col_sum - diag
            
            per_class_IoU = safe_divide(diag, denominator)
            mIoU = safe_mean(per_class_IoU)
            return per_class_IoU, mIoU
        except Exception as e:
            logger.error(f"计算IoU时出错: {str(e)}")
            return torch.zeros(self.class_num, device=self.device), torch.tensor(0.0, device=self.device)
    
    def precision(self):
        """计算每类精确率和平均精确率"""
        try:
            col_sum = torch.sum(self.cfm, dim=0).float()
            per_class_precision = safe_divide(torch.diag(self.cfm).float(), col_sum)
            mPrecision = safe_mean(per_class_precision)
            return per_class_precision, mPrecision
        except Exception as e:
            logger.error(f"计算精确率时出错: {str(e)}")
            return torch.zeros(self.class_num, device=self.device), torch.tensor(0.0, device=self.device)
    
    def recall(self):
        """计算每类召回率和平均召回率"""
        try:
            row_sum = torch.sum(self.cfm, dim=1).float()
            per_class_recall = safe_divide(torch.diag(self.cfm).float(), row_sum)
            mRecall = safe_mean(per_class_recall)
            return per_class_recall, mRecall
        except Exception as e:
            logger.error(f"计算召回率时出错: {str(e)}")
            return torch.zeros(self.class_num, device=self.device), torch.tensor(0.0, device=self.device)
    
    def f1_score(self):
        """计算每类F1分数和平均F1分数"""
        try:
            precision = self.precision()[0]
            recall = self.recall()[0]
            per_class_f1 = safe_divide(2 * (precision * recall), (precision + recall))
            mF1 = safe_mean(per_class_f1)
            return per_class_f1, mF1
        except Exception as e:
            logger.error(f"计算F1分数时出错: {str(e)}")
            return torch.zeros(self.class_num, device=self.device), torch.tensor(0.0, device=self.device)
    
    def mean_pixel_accuracy(self):
        """计算平均像素准确率(mPA)"""
        try:
            per_class_PA = self.acc()[0]
            mPA = safe_mean(per_class_PA)
            return mPA
        except Exception as e:
            logger.error(f"计算平均像素准确率时出错: {str(e)}")
            return torch.tensor(0.0, device=self.device)
    
    def mean_intersection_over_union(self):
        """计算平均交并比(mIoU)"""
        try:
            per_class_IoU = self.iou()[0]
            mIoU = safe_mean(per_class_IoU)
            return mIoU
        except Exception as e:
            logger.error(f"计算平均交并比时出错: {str(e)}")
            return torch.tensor(0.0, device=self.device)
    
    def frequency_weighted_intersection_over_union(self):
        """计算频率加权交并比(FWIoU)"""
        try:
            # 计算每个类别的频率（像素比例）
            total_pixels = torch.sum(self.cfm).float()
            if total_pixels == 0:
                logger.warning("混淆矩阵为空，无法计算FWIoU")
                return torch.tensor(0.0, device=self.device)
                
            # 计算每个类别的像素数（真实标签中）
            class_pixels = torch.sum(self.cfm, dim=1).float()
            
            # 计算每个类别的频率
            freq = safe_divide(class_pixels, total_pixels)
            
            # 获取每个类别的IoU
            per_class_IoU = self.iou()[0]
            
            # 检查是否有NaN或Inf
            if torch.isnan(freq).any() or torch.isinf(freq).any() or torch.isnan(per_class_IoU).any() or torch.isinf(per_class_IoU).any():
                logger.warning("FWIoU计算中检测到NaN或Inf值")
                # 移除NaN和Inf
                valid_mask = ~torch.isnan(freq) & ~torch.isinf(freq) & ~torch.isnan(per_class_IoU) & ~torch.isinf(per_class_IoU)
                if not valid_mask.any():
                    logger.warning("FWIoU计算中没有有效值")
                    return torch.tensor(0.0, device=self.device)
                    
                # 只使用有效值计算
                valid_freq = freq[valid_mask]
                valid_iou = per_class_IoU[valid_mask]
                FWIoU = torch.sum(valid_freq * valid_iou)
            else:
                # 计算加权和
                FWIoU = torch.sum(freq * per_class_IoU)
                
            # 打印调试信息
            logger.debug(f"FWIoU计算 - 类别频率: {freq}, 类别IoU: {per_class_IoU}, FWIoU: {FWIoU}")
                
            return FWIoU
        except Exception as e:
            logger.error(f"计算频率加权交并比时出错: {str(e)}")
            return torch.tensor(0.0, device=self.device)
        
    def micro_precision_recall_f1(self):
        """计算微平均（Micro-average）精确率、召回率和F1分数"""
        try:
            # 微平均：先将所有类别的TP, FP, FN加总，再计算指标
            tp = torch.diag(self.cfm).sum().float()
            fp = torch.sum(self.cfm, dim=0).sum().float() - tp
            fn = torch.sum(self.cfm, dim=1).sum().float() - tp
            
            micro_precision = safe_divide(tp, (tp + fp))
            micro_recall = safe_divide(tp, (tp + fn))
            micro_f1 = safe_divide(2 * micro_precision * micro_recall, (micro_precision + micro_recall))
            
            return micro_precision.item(), micro_recall.item(), micro_f1.item()
        except Exception as e:
            logger.error(f"计算微平均指标时出错: {str(e)}")
            return 0.0, 0.0, 0.0
    
    def macro_precision_recall_f1(self):
        """计算宏平均（Macro-average）精确率、召回率和F1分数"""
        try:
            # 宏平均：先计算每个类别的指标，再取平均
            precision = self.precision()[0]
            recall = self.recall()[0]
            f1 = self.f1_score()[0]
            
            macro_precision = safe_mean(precision).item()
            macro_recall = safe_mean(recall).item()
            macro_f1 = safe_mean(f1).item()
            
            return macro_precision, macro_recall, macro_f1
        except Exception as e:
            logger.error(f"计算宏平均指标时出错: {str(e)}")
            return 0.0, 0.0, 0.0
    
    def compute(self):
        """计算所有评估指标"""
        try:
            # 检查混淆矩阵是否有效
            if self.cfm.sum() == 0:
                logger.warning("混淆矩阵为空，无法计算有效指标")
                # 返回全0结果
                return {
                    'per_class_PA': [0.0] * self.class_num,
                    'mPA': 0.0,
                    'total_PA': 0.0,
                    'per_class_IoU': [0.0] * self.class_num,
                    'mIoU': 0.0,
                    'FWIoU': 0.0,
                    'per_class_precision': [0.0] * self.class_num,
                    'mPrecision': 0.0,
                    'per_class_recall': [0.0] * self.class_num,
                    'mRecall': 0.0,
                    'per_class_F1': [0.0] * self.class_num,
                    'mF1': 0.0,
                    'macro_precision': 0.0,
                    'macro_recall': 0.0,
                    'macro_f1': 0.0,
                    'micro_precision': 0.0,
                    'micro_recall': 0.0,
                    'micro_f1': 0.0,
                }
            
            # 计算所有指标
            pixel_acc = self.acc()  # 返回(per_class_PA, total_PA)
            iou = self.iou()        # 返回(per_class_IoU, mIoU)
            precision = self.precision()  # 返回(per_class_precision, mPrecision)
            recall = self.recall()        # 返回(per_class_recall, mRecall)
            f1 = self.f1_score()          # 返回(per_class_F1, mF1)
            mPA = self.mean_pixel_accuracy()
            mIoU = self.mean_intersection_over_union()
            FWIoU = self.frequency_weighted_intersection_over_union()
            
            # 计算宏平均和微平均指标
            micro_precision, micro_recall, micro_f1 = self.micro_precision_recall_f1()
            macro_precision, macro_recall, macro_f1 = self.macro_precision_recall_f1()
            
            # 安全转换为Python列表
            def safe_to_list(tensor):
                try:
                    if isinstance(tensor, torch.Tensor):
                        # 替换NaN和Inf为0
                        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
                        return tensor.tolist()
                    elif isinstance(tensor, list):
                        return tensor
                    else:
                        return [0.0] * self.class_num
                except Exception as e:
                    logger.error(f"转换为列表时出错: {str(e)}")
                    return [0.0] * self.class_num
            
            # 安全获取标量值
            def safe_item(tensor, default=0.0):
                try:
                    if isinstance(tensor, torch.Tensor):
                        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                            return default
                        return tensor.item()
                    return default if (isinstance(tensor, float) and (math.isnan(tensor) or math.isinf(tensor))) else tensor
                except Exception as e:
                    logger.error(f"获取标量值时出错: {str(e)}")
                    return default
            
            # 组织返回结果，按照指标类型分组
            results = {
                # 像素准确率相关指标
                'per_class_PA': safe_to_list(pixel_acc[0]),    # 每个类别的像素准确率
                'mPA': safe_item(mPA),                         # 平均像素准确率
                'total_PA': safe_item(pixel_acc[1]),           # 总体像素准确率
                
                # IoU相关指标
                'per_class_IoU': safe_to_list(iou[0]),        # 每个类别的IoU
                'mIoU': safe_item(mIoU),                      # 平均IoU
                'FWIoU': safe_item(FWIoU),                    # 频率加权IoU
                
                # 精确率、召回率相关指标
                'per_class_precision': safe_to_list(precision[0]),  # 每个类别的精确率
                'mPrecision': safe_item(precision[1]),              # 平均精确率
                'per_class_recall': safe_to_list(recall[0]),        # 每个类别的召回率
                'mRecall': safe_item(recall[1]),                    # 平均召回率
                
                # F1-score相关指标
                'per_class_F1': safe_to_list(f1[0]),              # 每个类别的F1分数
                'mF1': safe_item(f1[1]),                          # 平均F1分数
                
                # 宏平均指标
                'macro_precision': safe_item(macro_precision),        # 宏平均精确率
                'macro_recall': safe_item(macro_recall),              # 宏平均召回率
                'macro_f1': safe_item(macro_f1),                      # 宏平均F1分数
                
                # 微平均指标
                'micro_precision': safe_item(micro_precision),        # 微平均精确率
                'micro_recall': safe_item(micro_recall),              # 微平均召回率
                'micro_f1': safe_item(micro_f1),                      # 微平均F1分数
            }
            
            return results
        except Exception as e:
            logger.error(f"计算指标时出错: {str(e)}")
            # 返回全0结果
            return {
                'per_class_PA': [0.0] * self.class_num,
                'mPA': 0.0,
                'total_PA': 0.0,
                'per_class_IoU': [0.0] * self.class_num,
                'mIoU': 0.0,
                'FWIoU': 0.0,
                'per_class_precision': [0.0] * self.class_num,
                'mPrecision': 0.0,
                'per_class_recall': [0.0] * self.class_num,
                'mRecall': 0.0,
                'per_class_F1': [0.0] * self.class_num,
                'mF1': 0.0,
                'macro_precision': 0.0,
                'macro_recall': 0.0,
                'macro_f1': 0.0,
                'micro_precision': 0.0,
                'micro_recall': 0.0,
                'micro_f1': 0.0,
            }
        
    def save_metrics(self, results, save_path):
        """保存评估指标到TXT文件"""
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                # 保存整体指标
                f.write("整体指标:\n")
                f.write(f"mIoU: {results['mIoU']:.4f}\n")
                f.write(f"mPA: {results['mPA']:.4f}\n")
                f.write(f"FWIoU: {results['FWIoU']:.4f}\n")
                f.write(f"Pixel Accuracy: {results['total_PA']:.4f}\n\n")
                
                # 保存宏平均指标
                f.write("宏平均指标:\n")
                f.write(f"宏平均精确率: {results['macro_precision']:.4f}\n")
                f.write(f"宏平均召回率: {results['macro_recall']:.4f}\n")
                f.write(f"宏平均F1: {results['macro_f1']:.4f}\n\n")
                
                # 保存微平均指标
                f.write("微平均指标:\n")
                f.write(f"微平均精确率: {results['micro_precision']:.4f}\n")
                f.write(f"微平均召回率: {results['micro_recall']:.4f}\n")
                f.write(f"微平均F1: {results['micro_f1']:.4f}\n\n")
                
                # 保存各类别的IoU
                f.write("各类别IoU:\n")
                for i, iou in enumerate(results['per_class_IoU']):
                    f.write(f"类别 {i}: {iou:.4f}\n")
                f.write("\n")
                
                # 保存各类别的PA
                f.write("各类别PA:\n")
                for i, pa in enumerate(results['per_class_PA']):
                    f.write(f"类别 {i}: {pa:.4f}\n")
                
            print(f"Metrics saved to {save_path}")
        except Exception as e:
            logger.error(f"保存指标时出错: {str(e)}")
            print(f"保存指标失败: {str(e)}")
    
if __name__ == '__main__':
    
    # 假设我们有一个3类的语义分割任务
    class_num = 6
    metrics = Metrics(class_num)
    
    # 模拟一些样本的真实标签和预测标签
    # true_vectors = torch.tensor([0, 1, 2, 1, 0, 2, 2, 1, 0, 1])
    # pred_vectors = torch.tensor([0, 2, 2, 1, 0, 2, 0, 1, 1, 1])
    label_fp = r'E:\work\data\potsdam\test\labels_digital'
    preds_fp = r'E:\work\data\potsdam\test\preds'
    
    files = os.listdir(label_fp)
    import tifffile as tiff
    from tqdm import tqdm
    for file in tqdm(files):
        label = torch.from_numpy(tiff.imread(os.path.join(label_fp, file)))
        pred = torch.from_numpy(tiff.imread(os.path.join(preds_fp, file)))
        metrics.sample_add(label, pred)

    # 将这些样本添加到混淆矩阵中
    
    # 计算所有的指标
    results = metrics.compute()
    
    # 输出结果
    print("\n评估指标:")
    # 首先输出每个类别的详细指标
    class_metrics = ['per_class_PA', 'per_class_IoU', 'per_class_precision', 'per_class_recall', 'per_class_F1']
    for metric in class_metrics:
        if metric in results:
            print(f"\n{metric}:")
            for i, value in enumerate(results[metric]):
                print(f"  Class {i}: {value:.4f}")
    
    # 然后输出总体指标
    print("\n总体指标:")
    for key, value in results.items():
        if not isinstance(value, list):  # 只输出非列表类型的指标
            print(f"{key}: {value:.4f}")
            
    # 特别输出宏平均和微平均指标
    print("\n宏平均和微平均指标:")
    print(f"宏平均精确率: {results['macro_precision']:.4f}")
    print(f"宏平均召回率: {results['macro_recall']:.4f}")
    print(f"宏平均F1: {results['macro_f1']:.4f}")
    print(f"微平均精确率: {results['micro_precision']:.4f}")
    print(f"微平均召回率: {results['micro_recall']:.4f}")
    print(f"微平均F1: {results['micro_f1']:.4f}")
    
    # 保存指标到TXT文件