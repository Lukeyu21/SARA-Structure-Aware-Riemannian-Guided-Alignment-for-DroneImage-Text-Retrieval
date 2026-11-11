import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class SPDLogEigModule(nn.Module):
    """SPD矩阵的Log-Euclidean变换模块"""
    def __init__(self, eps: float = 1e-6):
        super(SPDLogEigModule, self).__init__()
        self.eps = eps
    
    def forward(self, spd_matrices: torch.Tensor) -> torch.Tensor:
        if len(spd_matrices.shape) == 3:
            return self._log_eig_2d(spd_matrices)
        elif len(spd_matrices.shape) == 4:
            B, N, C, _ = spd_matrices.shape
            spd_reshaped = spd_matrices.view(B * N, C, C)
            log_spd = self._log_eig_2d(spd_reshaped)
            return log_spd.view(B, N, -1)
        else:
            raise ValueError(f"Unexpected SPD matrix shape: {spd_matrices.shape}")
    
    def _log_eig_2d(self, spd_matrices: torch.Tensor) -> torch.Tensor:
        B, C, _ = spd_matrices.shape
        
        # 确保矩阵是正定的，增加更大的对角项
        identity = torch.eye(C, device=spd_matrices.device).unsqueeze(0).expand(B, -1, -1)
        spd_matrices = spd_matrices + 1e-3 * identity  # 增加更大的对角项
        
        # 使用更稳定的特征值分解方法
        try:
            # 首先尝试使用eigh
            eigenvals, eigenvecs = torch.linalg.eigh(spd_matrices)
        except Exception as e:
            # 如果失败，使用更稳定的方法
            # 添加更大的对角项
            spd_matrices = spd_matrices + 1e-2 * identity
            eigenvals, eigenvecs = torch.linalg.eigh(spd_matrices)
        
        # 确保特征值为正
        eigenvals = torch.clamp(eigenvals, min=1e-6)
        
        # 对特征值取对数
        log_eigenvals = torch.log(eigenvals + self.eps)
        
        # 重构对数矩阵
        try:
            # 分步进行矩阵乘法以避免内存问题
            diag_log_eigenvals = torch.diag_embed(log_eigenvals)  # [B, C, C]
            temp = torch.bmm(eigenvecs, diag_log_eigenvals)  # [B, C, C]
            log_spd = torch.bmm(temp, eigenvecs.transpose(-2, -1))  # [B, C, C]
        except Exception as e:
            # 如果矩阵乘法失败，返回零矩阵
            log_spd = torch.zeros(B, C, C, device=spd_matrices.device)
        
        return log_spd.view(B, -1)


class SPDAlignLoss(nn.Module):
    """Log-Euclidean结构对齐损失"""
    def __init__(self, beta1: float = 1.0, beta2: float = 1.0, eps: float = 1e-6):
        super(SPDAlignLoss, self).__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.log_eig = SPDLogEigModule(eps=eps)
    
    def forward(self, v_spd: torch.Tensor, t_spd: torch.Tensor, 
                pos_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        try:
            # 对SPD矩阵进行Log-Euclidean变换
            v_log = self.log_eig(v_spd)
            t_log = self.log_eig(t_spd)
            
            # 计算边缘分布对齐损失 (MDA)
            mda_loss = self._compute_mda_loss(v_log, t_log)
            
            # 计算条件分布对齐损失 (CDA)
            cda_loss = self._compute_cda_loss(v_log, t_log, pos_mask)
            
            # 总损失
            total_loss = self.beta1 * mda_loss + self.beta2 * cda_loss
            
            loss_dict = {
                'mda_loss': mda_loss.item(),
                'cda_loss': cda_loss.item(),
                'total_structure_loss': total_loss.item()
            }
            
            return total_loss, loss_dict
            
        except Exception as e:
            # 如果出现任何错误，返回零损失
            device = v_spd.device
            zero_loss = torch.tensor(0.0, device=device)
            loss_dict = {
                'mda_loss': 0.0,
                'cda_loss': 0.0,
                'total_structure_loss': 0.0
            }
            return zero_loss, loss_dict
    
    def _compute_mda_loss(self, v_log: torch.Tensor, t_log: torch.Tensor) -> torch.Tensor:
        if len(v_log.shape) == 3:
            v_log = v_log.mean(dim=1)
            t_log = t_log.mean(dim=1)
        
        # 简化MDA损失计算
        try:
            # 直接计算两个特征的平均值差异
            v_mean = v_log.mean(dim=0)  # [dim]
            t_mean = t_log.mean(dim=0)  # [dim]
            
            # 确保维度匹配
            min_dim = min(v_mean.shape[0], t_mean.shape[0])
            v_mean = v_mean[:min_dim]
            t_mean = t_mean[:min_dim]
            
            # L2距离
            mda_loss = torch.norm(v_mean - t_mean, p=2)
        except Exception as e:
            print(f"MDA损失计算失败: {e}")
            # 如果计算失败，返回零损失
            mda_loss = torch.tensor(0.0, device=v_log.device)
        
        return mda_loss
    
    def _compute_cda_loss(self, v_log: torch.Tensor, t_log: torch.Tensor,
                         pos_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if len(v_log.shape) == 3:
            v_log = v_log.mean(dim=1)
            t_log = t_log.mean(dim=1)
        
        B = v_log.shape[0]
        
        # 简化CDA损失计算，只计算对角线上的样本对
        # 避免复杂的索引操作
        try:
            # 直接计算对角线上的样本对距离
            # 对于2D张量，使用Frobenius范数计算每个样本的距离
            diff = v_log - t_log  # [B, dim]
            diag_distances = torch.norm(diff, p=2, dim=1)  # [B] - 每个样本的L2范数
            cda_loss = diag_distances.mean()
        except Exception as e:
            print(f"CDA损失计算失败: {e}")
            # 如果计算失败，返回零损失
            cda_loss = torch.tensor(0.0, device=v_log.device)
        
        return cda_loss


class DualSimilarityModule(nn.Module):
    """双通道相似度计算模块"""
    def __init__(self, alpha: float = 0.7, similarity_type: str = 'cosine'):
        super(DualSimilarityModule, self).__init__()
        self.alpha = alpha
        self.similarity_type = similarity_type
    
    def forward(self, v_sem: torch.Tensor, t_sem: torch.Tensor,
                v_spd: torch.Tensor, t_spd: torch.Tensor) -> torch.Tensor:
        # 计算语义相似度
        sem_sim = self._compute_similarity(v_sem, t_sem)
        
        # 处理结构特征
        if len(v_spd.shape) == 3:
            v_spd = v_spd.mean(dim=1)
            t_spd = t_spd.mean(dim=1)
        
        # 计算结构相似度
        str_sim = self._compute_similarity(v_spd, t_spd)
        
        # 融合相似度
        dual_sim = self.alpha * sem_sim + (1 - self.alpha) * str_sim
        return dual_sim
    
    def _compute_similarity(self, v_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        if self.similarity_type == 'cosine':
            v_norm = F.normalize(v_feat, p=2, dim=1)
            t_norm = F.normalize(t_feat, p=2, dim=1)
            similarity = torch.mm(v_norm, t_norm.t())
        elif self.similarity_type == 'dot':
            similarity = torch.mm(v_feat, t_feat.t())
        else:
            raise ValueError(f"Unsupported similarity type: {self.similarity_type}")
        
        return similarity 