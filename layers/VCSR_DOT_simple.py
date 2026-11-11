import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from .dot_module import DOTModule


class VCSRWithDOT(nn.Module):
    """集成DOT模块的VCSR模型"""
    
    def __init__(self, opt: Dict[str, Any]):
        super().__init__()
        
        self.opt = opt
        
        # 从现有VCSR模型继承组件
        from .VCSR import VCSR
        self.base_model = VCSR(opt)
        
        # DOT模块
        self.dot_module = DOTModule(
            feature_dim=opt.get('structure_dim', 256),
            spd_dim=opt.get('spd_dim', 64),
            use_pade=opt.get('use_pade', True),
            pade_k=opt.get('pade_k', 12),
            beta1=opt.get('dot_beta1', 1.0),
            beta2=opt.get('dot_beta2', 1.0)
        )
        
        # 损失权重
        self.lambda_dot = opt.get('lambda_dot', 0.5)
        
    def forward(self, image, text_ids, text_mask, positive_pairs=None):
        # 基础VCSR前向传播
        base_outputs = self.base_model(image, text_ids, text_mask)
        
        # 提取结构特征
        v_str = base_outputs.get('v_str', base_outputs.get('image_features'))
        t_str = base_outputs.get('t_str', base_outputs.get('text_features'))
        
        # DOT损失
        dot_loss, dot_dict = self.dot_module(v_str, t_str, positive_pairs)
        
        # 更新总损失
        total_loss = base_outputs['loss'] + self.lambda_dot * dot_loss
        
        # 更新输出
        outputs = base_outputs.copy()
        outputs.update({
            'loss': total_loss,
            'dot_loss': dot_loss,
            'dot_dict': dot_dict
        })
        
        return outputs 