#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DOT (Distribution-Oriented Transfer) 模块
实现MDA (Marginal Distribution Alignment) 和 CDA (Conditional Distribution Alignment)
用于提升结构分布对齐与鲁棒性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
import warnings

class LogEig(nn.Module):
    """
    Log-Euclidean层：将SPD矩阵映射到Log-Euclidean空间
    """
    def __init__(self, eps=1e-6):
        super(LogEig, self).__init__()
        self.eps = eps
    
    def forward(self, x):
        """
        Args:
            x: SPD矩阵 [B, D, D] 或 [B, N, D, D]
        Returns:
            log_x: Log-Euclidean映射结果
        """
        if len(x.shape) == 3:
            # [B, D, D]
            return self._log_eig_2d(x)
        elif len(x.shape) == 4:
            # [B, N, D, D]
            B, N, D, _ = x.shape
            x_reshaped = x.view(B * N, D, D)
            log_x = self._log_eig_2d(x_reshaped)
            return log_x.view(B, N, D, D)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
    
    def _log_eig_2d(self, x):
        """处理2D SPD矩阵的Log-Euclidean映射"""
        B, D, _ = x.shape
        log_x = torch.zeros_like(x)
        
        for i in range(B):
            try:
                # 检查输入是否包含NaN或无穷大
                if torch.isnan(x[i]).any() or torch.isinf(x[i]).any():
                    # 使用单位矩阵作为fallback
                    log_x[i] = torch.zeros_like(x[i])
                    continue
                
                # 确保矩阵是正定的
                eigvals, eigvecs = torch.linalg.eigh(x[i])
                # 添加小的对角项确保正定性
                eigvals = torch.clamp(eigvals, min=self.eps)
                # 计算对数
                log_eigvals = torch.log(eigvals)
                # 重构矩阵
                log_x[i] = eigvecs @ torch.diag(log_eigvals) @ eigvecs.T
            except Exception as e:
                # 如果特征值分解失败，使用近似方法
                warnings.warn(f"Eigenvalue decomposition failed for sample {i}: {e}, using approximation")
                log_x[i] = torch.log(x[i] + self.eps * torch.eye(D, device=x.device))
        
        return log_x

class DOTModule(nn.Module):
    """
    Distribution Optimal Transport (DOT) 模块
    用于结构特征的对齐和鲁棒性增强
    """
    def __init__(self, feature_dim=96, beta1=1.0, beta2=1.0, eps=1e-6):
        super(DOTModule, self).__init__()
        self.feature_dim = feature_dim
        self.beta1 = beta1  # MDA损失权重
        self.beta2 = beta2  # CDA损失权重
        self.eps = eps
        
        # Log-Euclidean映射层
        self.log_eig = LogEig(eps=eps)
        
        # 特征到SPD矩阵的映射（如果需要）
        self.to_spd = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim * feature_dim, feature_dim * feature_dim)
        )
    
    def features_to_spd(self, features):
        """
        将特征向量转换为SPD矩阵
        Args:
            features: [B, feature_dim] 或 [B, N, feature_dim]
        Returns:
            spd_matrices: [B, feature_dim, feature_dim] 或 [B, N, feature_dim, feature_dim]
        """
        if len(features.shape) == 2:
            # [B, feature_dim] -> [B, feature_dim, feature_dim]
            B, dim = features.shape
            # 检查输入特征是否包含NaN或无穷大
            if torch.isnan(features).any() or torch.isinf(features).any():
                # 如果输入有问题，使用单位矩阵
                identity = torch.eye(dim, device=features.device).unsqueeze(0).expand(B, -1, -1)
                spd = identity
            else:
                # 使用外积构造对称矩阵
                spd = torch.bmm(features.unsqueeze(2), features.unsqueeze(1))  # [B, dim, dim]
                # 添加对角项确保正定性
                identity = torch.eye(dim, device=features.device).unsqueeze(0).expand(B, -1, -1)
                spd = spd + 1e-3 * identity  # 增加对角项权重以提高数值稳定性
            return spd
        elif len(features.shape) == 3:
            # [B, N, feature_dim] -> [B, N, feature_dim, feature_dim]
            B, N, dim = features.shape
            features_reshaped = features.view(B * N, dim)
            spd = self.features_to_spd(features_reshaped)
            return spd.view(B, N, dim, dim)
        else:
            raise ValueError(f"Unexpected features shape: {features.shape}")
    
    def compute_mda_loss(self, v_log, t_log):
        """
        计算边缘分布对齐损失 (Marginal Distribution Alignment)
        Args:
            v_log: 图像结构特征的Log-Euclidean映射 [B, D, D]
            t_log: 文本结构特征的Log-Euclidean映射 [B, D, D]
        Returns:
            mda_loss: 边缘分布对齐损失
        """
        # 计算批次均值
        v_mean = v_log.mean(dim=0)  # [D, D]
        t_mean = t_log.mean(dim=0)  # [D, D]
        
        # Frobenius范数距离
        mda_loss = torch.norm(v_mean - t_mean, p='fro')
        return mda_loss
    
    def compute_cda_loss(self, v_log, t_log, pos_pair_indices=None):
        """
        计算条件分布对齐损失 (Conditional Distribution Alignment)
        Args:
            v_log: 图像结构特征的Log-Euclidean映射 [B, D, D]
            t_log: 文本结构特征的Log-Euclidean映射 [B, D, D]
            pos_pair_indices: 正配对索引，如果为None则假设所有样本都是正配对
        Returns:
            cda_loss: 条件分布对齐损失
        """
        B = v_log.size(0)
        
        if pos_pair_indices is None:
            # 假设所有样本都是正配对
            pos_pair_indices = list(range(B))
        
        if len(pos_pair_indices) == 0:
            return torch.tensor(0.0, device=v_log.device, requires_grad=True)
        
        # 计算正配对样本的Frobenius距离
        pair_distances = []
        for i in pos_pair_indices:
            if i < B:  # 确保索引有效
                dist = torch.norm(v_log[i] - t_log[i], p='fro')
                pair_distances.append(dist)
        
        if len(pair_distances) == 0:
            return torch.tensor(0.0, device=v_log.device, requires_grad=True)
        
        cda_loss = torch.mean(torch.stack(pair_distances))
        return cda_loss
    
    def forward(self, v_str, t_str, pos_pair_indices=None):
        """
        DOT模块前向传播
        Args:
            v_str: 图像结构特征 [B, feature_dim] 或 [B, N, feature_dim]
            t_str: 文本结构特征 [B, feature_dim] 或 [B, N, feature_dim]
            pos_pair_indices: 正配对索引
        Returns:
            dot_loss: DOT总损失
            v_log: 图像特征的Log-Euclidean映射
            t_log: 文本特征的Log-Euclidean映射
        """
        # 将特征转换为SPD矩阵
        v_spd = self.features_to_spd(v_str)
        t_spd = self.features_to_spd(t_str)
        
        # Log-Euclidean映射
        v_log = self.log_eig(v_spd)
        t_log = self.log_eig(t_spd)
        
        # 计算MDA损失
        mda_loss = self.compute_mda_loss(v_log, t_log)
        
        # 计算CDA损失
        cda_loss = self.compute_cda_loss(v_log, t_log, pos_pair_indices)
        
        # 总DOT损失
        dot_loss = self.beta1 * mda_loss + self.beta2 * cda_loss
        
        return dot_loss, v_log, t_log

def cosine_sim(x, y):
    """计算余弦相似度"""
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return torch.mm(x, y.t()) 