#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成DOT模块的双分支+双通道VCSR模型
实现MDA+CDA机制的结构分布对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import math

from .xvlm import XVLM
from .xbert import BertModel
from .spdnet import SPDNet
from .dot_module import DOTModule


class DualBranchVCSR(nn.Module):
    """
    双分支+双通道VCSR模型
    图像分支：语义特征 + 结构特征
    文本分支：语义特征 + 结构特征
    """
    
    def __init__(self, opt: Dict[str, Any]):
        super().__init__()
        
        self.opt = opt
        
        # 图像编码器 (RestV2)
        self.image_encoder = XVLM(opt)
        
        # 文本编码器 (BERT)
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        
        # 语义特征提取层
        self.image_semantic_proj = nn.Linear(opt.get('image_dim', 768), opt.get('semantic_dim', 512))
        self.text_semantic_proj = nn.Linear(opt.get('text_dim', 768), opt.get('semantic_dim', 512))
        
        # 结构特征提取层
        self.image_structure_proj = nn.Linear(opt.get('image_dim', 768), opt.get('structure_dim', 256))
        self.text_structure_proj = nn.Linear(opt.get('text_dim', 768), opt.get('structure_dim', 256))
        
        # SPDNet用于图像结构特征
        self.spdnet = SPDNet(opt)
        
        # DOT模块用于结构分布对齐
        self.dot_module = DOTModule(
            feature_dim=opt.get('structure_dim', 256),
            spd_dim=opt.get('spd_dim', 64),
            use_pade=opt.get('use_pade', True),
            pade_k=opt.get('pade_k', 12),
            beta1=opt.get('dot_beta1', 1.0),
            beta2=opt.get('dot_beta2', 1.0)
        )
        
        # 相似度融合参数
        self.alpha = opt.get('sim_alpha', 0.7)  # 语义相似度权重
        
        # 损失权重
        self.lambda1 = opt.get('lambda1', 1.0)  # 结构对比损失权重
        self.lambda2 = opt.get('lambda2', 0.5)  # DOT损失权重
        
        # 温度参数
        self.temperature = opt.get('temperature', 0.1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in [self.image_semantic_proj, self.text_semantic_proj,
                  self.image_structure_proj, self.text_structure_proj]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码图像，提取语义和结构特征
        
        Args:
            image: 输入图像 [batch_size, channels, height, width]
            
        Returns:
            v_sem: 图像语义特征 [batch_size, semantic_dim]
            v_str: 图像结构特征 [batch_size, structure_dim]
        """
        # 通过图像编码器
        image_features = self.image_encoder(image)  # [batch_size, image_dim]
        
        # 提取CLS token作为语义特征
        v_sem = self.image_semantic_proj(image_features)  # [batch_size, semantic_dim]
        
        # 提取中间特征用于结构特征
        intermediate_feat = self.image_encoder.get_intermediate_features(image)  # [batch_size, intermediate_dim]
        
        # 通过SPDNet提取结构特征
        spd_output = self.spdnet(intermediate_feat)  # [batch_size, spd_output_dim]
        
        # 投影到结构特征空间
        v_str = self.image_structure_proj(spd_output)  # [batch_size, structure_dim]
        
        return v_sem, v_str
    
    def encode_text(self, text_ids: torch.Tensor, text_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码文本，提取语义和结构特征
        
        Args:
            text_ids: 文本token ids [batch_size, seq_len]
            text_mask: 文本mask [batch_size, seq_len]
            
        Returns:
            t_sem: 文本语义特征 [batch_size, semantic_dim]
            t_str: 文本结构特征 [batch_size, structure_dim]
        """
        # 通过文本编码器
        text_outputs = self.text_encoder(text_ids, attention_mask=text_mask)
        text_features = text_outputs.last_hidden_state  # [batch_size, seq_len, text_dim]
        
        # 提取CLS token作为语义特征
        cls_token = text_features[:, 0, :]  # [batch_size, text_dim]
        t_sem = self.text_semantic_proj(cls_token)  # [batch_size, semantic_dim]
        
        # 计算token embedding的协方差矩阵作为结构特征
        # 移除CLS token，只使用内容token
        content_tokens = text_features[:, 1:, :]  # [batch_size, seq_len-1, text_dim]
        content_mask = text_mask[:, 1:]  # [batch_size, seq_len-1]
        
        # 计算掩码后的协方差矩阵
        batch_size, seq_len, feat_dim = content_tokens.shape
        
        # 应用mask
        masked_tokens = content_tokens * content_mask.unsqueeze(-1)  # [batch_size, seq_len-1, text_dim]
        
        # 计算每个样本的协方差矩阵
        cov_features = []
        for i in range(batch_size):
            # 获取有效token
            valid_mask = content_mask[i]  # [seq_len-1]
            valid_tokens = masked_tokens[i][valid_mask.bool()]  # [num_valid, text_dim]
            
            if valid_tokens.shape[0] > 1:
                # 计算协方差矩阵
                mean_tokens = valid_tokens.mean(dim=0, keepdim=True)  # [1, text_dim]
                centered_tokens = valid_tokens - mean_tokens  # [num_valid, text_dim]
                cov_matrix = centered_tokens.t() @ centered_tokens / (valid_tokens.shape[0] - 1)  # [text_dim, text_dim]
                
                # 确保对称正定
                cov_matrix = 0.5 * (cov_matrix + cov_matrix.t())
                cov_matrix = cov_matrix + torch.eye(feat_dim, device=cov_matrix.device, dtype=cov_matrix.dtype)
                
                # 展平协方差矩阵
                cov_flat = cov_matrix.flatten()  # [text_dim * text_dim]
            else:
                # 如果token太少，使用零向量
                cov_flat = torch.zeros(feat_dim * feat_dim, device=text_ids.device, dtype=text_ids.dtype)
            
            cov_features.append(cov_flat)
        
        # 堆叠所有样本的协方差特征
        cov_features = torch.stack(cov_features)  # [batch_size, text_dim * text_dim]
        
        # 投影到结构特征空间
        t_str = self.text_structure_proj(cov_features)  # [batch_size, structure_dim]
        
        return t_sem, t_str
    
    def compute_similarities(self, v_sem: torch.Tensor, v_str: torch.Tensor,
                           t_sem: torch.Tensor, t_str: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算语义和结构相似度，并融合
        
        Args:
            v_sem: 图像语义特征 [batch_size, semantic_dim]
            v_str: 图像结构特征 [batch_size, structure_dim]
            t_sem: 文本语义特征 [batch_size, semantic_dim]
            t_str: 文本结构特征 [batch_size, structure_dim]
            
        Returns:
            sim_sem: 语义相似度 [batch_size, batch_size]
            sim_str: 结构相似度 [batch_size, batch_size]
            dual_sim: 融合相似度 [batch_size, batch_size]
        """
        # 计算语义相似度
        v_sem_norm = F.normalize(v_sem, p=2, dim=1)
        t_sem_norm = F.normalize(t_sem, p=2, dim=1)
        sim_sem = torch.mm(v_sem_norm, t_sem_norm.t()) / self.temperature
        
        # 计算结构相似度
        v_str_norm = F.normalize(v_str, p=2, dim=1)
        t_str_norm = F.normalize(t_str, p=2, dim=1)
        sim_str = torch.mm(v_str_norm, t_str_norm.t()) / self.temperature
        
        # 融合相似度
        dual_sim = self.alpha * sim_sem + (1 - self.alpha) * sim_str
        
        return sim_sem, sim_str, dual_sim
    
    def compute_losses(self, dual_sim: torch.Tensor, v_str: torch.Tensor, t_str: torch.Tensor,
                      positive_pairs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算所有损失
        
        Args:
            dual_sim: 融合相似度 [batch_size, batch_size]
            v_str: 图像结构特征 [batch_size, structure_dim]
            t_str: 文本结构特征 [batch_size, structure_dim]
            positive_pairs: 正样本对索引
            
        Returns:
            total_loss: 总损失
            loss_dict: 损失详情
        """
        # 对比损失 (InfoNCE)
        labels = torch.arange(dual_sim.size(0), device=dual_sim.device)
        contrastive_loss = F.cross_entropy(dual_sim, labels)
        
        # 结构对比损失
        v_str_norm = F.normalize(v_str, p=2, dim=1)
        t_str_norm = F.normalize(t_str, p=2, dim=1)
        struct_sim = torch.mm(v_str_norm, t_str_norm.t()) / self.temperature
        struct_loss = F.cross_entropy(struct_sim, labels)
        
        # DOT损失
        dot_loss, dot_dict = self.dot_module(v_str, t_str, positive_pairs)
        
        # 总损失
        total_loss = contrastive_loss + self.lambda1 * struct_loss + self.lambda2 * dot_loss
        
        # 损失详情
        loss_dict = {
            'total': total_loss.item(),
            'contrastive': contrastive_loss.item(),
            'structural': struct_loss.item(),
            'dot_total': dot_dict['dot_total'],
            'dot_mda': dot_dict['dot_mda'],
            'dot_cda': dot_dict['dot_cda']
        }
        
        return total_loss, loss_dict
    
    def forward(self, image: torch.Tensor, text_ids: torch.Tensor, text_mask: torch.Tensor,
                positive_pairs: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            image: 输入图像 [batch_size, channels, height, width]
            text_ids: 文本token ids [batch_size, seq_len]
            text_mask: 文本mask [batch_size, seq_len]
            positive_pairs: 正样本对索引 [num_pairs, 2]
            
        Returns:
            包含所有输出和损失的字典
        """
        # 编码图像和文本
        v_sem, v_str = self.encode_image(image)
        t_sem, t_str = self.encode_text(text_ids, text_mask)
        
        # 计算相似度
        sim_sem, sim_str, dual_sim = self.compute_similarities(v_sem, v_str, t_sem, t_str)
        
        # 计算损失
        total_loss, loss_dict = self.compute_losses(dual_sim, v_str, t_str, positive_pairs)
        
        # 返回结果
        outputs = {
            'loss': total_loss,
            'loss_dict': loss_dict,
            'v_sem': v_sem,
            'v_str': v_str,
            't_sem': t_sem,
            't_str': t_str,
            'sim_sem': sim_sem,
            'sim_str': sim_str,
            'dual_sim': dual_sim
        }
        
        return outputs
    
    def encode_image_for_retrieval(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """为检索编码图像"""
        v_sem, v_str = self.encode_image(image)
        return {
            'v_sem': v_sem,
            'v_str': v_str
        }
    
    def encode_text_for_retrieval(self, text_ids: torch.Tensor, text_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """为检索编码文本"""
        t_sem, t_str = self.encode_text(text_ids, text_mask)
        return {
            't_sem': t_sem,
            't_str': t_str
        }
    
    def compute_retrieval_similarity(self, image_features: Dict[str, torch.Tensor],
                                   text_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算检索相似度"""
        v_sem, v_str = image_features['v_sem'], image_features['v_str']
        t_sem, t_str = text_features['t_sem'], text_features['t_str']
        
        _, _, dual_sim = self.compute_similarities(v_sem, v_str, t_sem, t_str)
        return dual_sim 