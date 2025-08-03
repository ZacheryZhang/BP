"""
对比学习模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random


class ContrastiveLoss(nn.Module):
    """对比损失函数"""
    
    def __init__(self, temperature=0.07, margin=1.0):
        """
        初始化对比损失
        
        Args:
            temperature: 温度参数
            margin: 边际参数
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, embeddings, labels, method='simclr'):
        """
        计算对比损失
        
        Args:
            embeddings: 特征嵌入 [batch_size, embedding_dim]
            labels: 标签 [batch_size, 2] (收缩压, 舒张压)
            method: 对比学习方法 ('simclr', 'supcon', 'triplet')
            
        Returns:
            torch.Tensor: 损失值
        """
        if method == 'simclr':
            return self._simclr_loss(embeddings, labels)
        elif method == 'supcon':
            return self._supervised_contrastive_loss(embeddings, labels)
        elif method == 'triplet':
            return self._triplet_loss(embeddings, labels)
        else:
            raise ValueError(f"不支持的对比学习方法: {method}")
    
    def _simclr_loss(self, embeddings, labels):
        """SimCLR损失"""
        batch_size = embeddings.shape[0]
        
        # 归一化嵌入
        embeddings = F.normalize(embeddings, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # 创建标签相似度矩阵
        label_similarity = self._compute_label_similarity(labels, labels)
        
        # 创建掩码
        mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        
        # 正样本对
        positive_mask = (label_similarity > 0.8) & (~mask)
        
        # 计算损失
        exp_sim = torch.exp(similarity_matrix)
        
        # 分母：所有非自身样本
        denominator = torch.sum(exp_sim * (~mask).float(), dim=1, keepdim=True)
        
        # 分子：正样本对
        numerator = exp_sim * positive_mask.float()
        
        # 计算损失
        loss = -torch.log(torch.sum(numerator, dim=1) / (denominator.squeeze() + 1e-8))
        
        # 只对有正样本的样本计算损失
        valid_samples = torch.sum(positive_mask, dim=1) > 0
        if torch.sum(valid_samples) > 0:
            loss = torch.mean(loss[valid_samples])
        else:
            loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return loss
    
    def _supervised_contrastive_loss(self, embeddings, labels):
        """监督对比损失"""
        batch_size = embeddings.shape[0]
        
        # 归一化嵌入
        embeddings = F.normalize(embeddings, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # 创建标签相似度矩阵
        label_similarity = self._compute_label_similarity(labels, labels)
        
        # 创建掩码
        mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        
        # 正样本掩码 (相似标签且非自身)
        positive_mask = (label_similarity > 0.7) & (~mask)
        
        # 负样本掩码 (不相似标签)
        negative_mask = (label_similarity < 0.3) & (~mask)
        
        # 计算损失
        exp_sim = torch.exp(similarity_matrix)
        
        loss_list = []
        for i in range(batch_size):
            # 正样本
            pos_exp = exp_sim[i] * positive_mask[i].float()
            
            # 负样本 + 正样本 (分母)
            neg_exp = exp_sim[i] * (positive_mask[i] | negative_mask[i]).float()
            
            if torch.sum(positive_mask[i]) > 0 and torch.sum(neg_exp) > 0:
                loss_i = -torch.log(torch.sum(pos_exp) / torch.sum(neg_exp))
                loss_list.append(loss_i)
        
        if len(loss_list) > 0:
            loss = torch.stack(loss_list).mean()
        else:
            loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return loss
    
    def _triplet_loss(self, embeddings, labels):
        """三元组损失"""
        batch_size = embeddings.shape[0]
        
        # 归一化嵌入
        embeddings = F.normalize(embeddings, dim=1)
        
        # 创建标签相似度矩阵
        label_similarity = self._compute_label_similarity(labels, labels)
        
        triplet_losses = []
        
        for i in range(batch_size):
            anchor = embeddings[i:i+1]  # [1, embedding_dim]
            
            # 找正样本 (相似标签)
            positive_mask = label_similarity[i] > 0.7
            positive_indices = torch.where(positive_mask)[0]
            positive_indices = positive_indices[positive_indices != i]  # 排除自身
            
            # 找负样本 (不相似标签)
            negative_mask = label_similarity[i] < 0.3
            negative_indices = torch.where(negative_mask)[0]
            
            if len(positive_indices) > 0 and len(negative_indices) > 0:
                # 随机选择正负样本
                pos_idx = positive_indices[torch.randint(0, len(positive_indices), (1,))]
                neg_idx = negative_indices[torch.randint(0, len(negative_indices), (1,))]
                
                positive = embeddings[pos_idx:pos_idx+1]
                negative = embeddings[neg_idx:neg_idx+1]
                
                # 计算距离
                pos_dist = F.pairwise_distance(anchor, positive)
                neg_dist = F.pairwise_distance(anchor, negative)
                
                # 三元组损失
                loss = F.relu(pos_dist - neg_dist + self.margin)
                triplet_losses.append(loss)
        
        if len(triplet_losses) > 0:
            return torch.stack(triplet_losses).mean()
        else:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    
    def _compute_label_similarity(self, labels1, labels2):
        """
        计算标签相似度
        
        Args:
            labels1
