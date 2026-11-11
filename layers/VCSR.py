import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
from collections import OrderedDict
from .VCSR_Modules import *
import copy

from layers.base_model import resnet181, ExtractFeature
from layers.location_model import Location_Net_stage_one

from sklearn import cluster, metrics
import numpy as np
from sklearn.preprocessing import normalize
import time
from .spdnet import SPDNet
from .xvlm import *
from .rest_v2 import *
from .rest_v2 import build_vision_encoder
from .lesa_modules import SPDAlignLoss, DualSimilarityModule, SPDLogEigModule
    
class BaseModel(nn.Module):

    def __init__(self, opt={}):
        super(BaseModel, self).__init__()

        #bert
        text_backbone = build_text_encoder()
        
        #rest_v2
        visual_backbone = build_vision_encoder(opt=opt)

        self.av_model = Location_Net_stage_one(visual_net=visual_backbone, txt_net=text_backbone)
        
        self.vision_proj = nn.Linear(768, 512)
        self.text_proj = nn.Linear(768, 512)

        # SPDNet for structural features
        self.spdconv = SPDNet(opt=opt)
        
        # Fusion components
        self.spd_proj = nn.Linear(96, 512)  # Project SPD features from 96 to 512
        self.attn_weights = nn.Linear(512, 1)  # Attention weights for fusion
        
        # 文本结构分支：MLP+BN替代SPDNet
        self.text_cov_mlp = nn.Sequential(
            nn.Linear(768*768, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 96),
            nn.BatchNorm1d(96),
            nn.ReLU()
        )
        self.spd_proj_text = nn.Linear(96, 512)

        # 图像结构分支MLP和投影层
        self.spd_mlp_img = nn.Sequential(
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Linear(96, 96)
        )
        self.spd_proj = nn.Linear(96, 512)

        # LESA相关组件
        lesa_config = opt.get('lesa', {})
        self.spd_align_loss = SPDAlignLoss(
            beta1=lesa_config.get('beta1', 1.0), 
            beta2=lesa_config.get('beta2', 1.0),
            eps=lesa_config.get('eps', 1e-6)
        )
        self.dual_similarity = DualSimilarityModule(
            alpha=lesa_config.get('alpha', 0.7), 
            similarity_type=lesa_config.get('similarity_type', 'cosine')
        )
        self.log_eig = SPDLogEigModule(eps=lesa_config.get('eps', 1e-6))
        
        # 结构特征投影层
        self.spd_proj_img = nn.Linear(96, 512)
        self.spd_proj_txt = nn.Linear(96, 512)

        self.Eiters = 0
       
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.itm_head = build_mlp(input_dim=768, output_dim=2)

    def get_cross_embeds(self, image_embeds, image_atts, text_ids=None, text_embeds=None, text_atts=None):
        assert text_atts is not None

        encoder = self.av_model.txt_net

        if text_embeds is not None:
            return encoder(encoder_embeds=text_embeds,
                           attention_mask=text_atts,
                           encoder_hidden_states=image_embeds,
                           encoder_attention_mask=image_atts,
                           return_dict=True,
                           mode='fusion',
                           ).last_hidden_state
        else:
            raise ValueError
        
    def get_features(self, image_embeds=None, text_embeds=None):
        return F.normalize(self.vision_proj(image_embeds)), \
                F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
    
    def compute_spd_matrix(self, v_embeds):
        """
        计算SPD矩阵用于结构特征提取
        Args:
            v_embeds: 视觉特征 [B, seq_len, dim]
        Returns:
            SPD矩阵用于SPDNet输入
        """
        # 将序列特征转换为图像格式用于SPDNet
        B, seq_len, dim = v_embeds.shape
        
        # 计算协方差矩阵作为SPD输入
        # 对每个batch样本计算序列特征的协方差矩阵
        v_embeds_centered = v_embeds - v_embeds.mean(dim=1, keepdim=True)  # 去均值
        cov_matrix = torch.bmm(v_embeds_centered.transpose(1, 2), v_embeds_centered) / (seq_len - 1)  # [B, dim, dim]
        
        # 确保协方差矩阵是正定的（添加小的对角项）
        identity = torch.eye(dim, device=v_embeds.device).unsqueeze(0).expand(B, -1, -1)
        cov_matrix = cov_matrix + 1e-6 * identity
        
        return cov_matrix
    
    def _construct_spd_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """
        从特征向量构造SPD矩阵
        Args:
            features: 特征向量 [B, dim]
        Returns:
            spd_matrix: SPD矩阵 [B, dim, dim]
        """
        B, dim = features.shape
        
        # 计算需要的下三角元素数量
        tril_size = dim * (dim + 1) // 2
        
        # 如果特征向量维度不够，进行填充或截断
        if features.shape[1] < tril_size:
            # 填充零
            padded_features = torch.zeros(B, tril_size, device=features.device, dtype=features.dtype)
            padded_features[:, :features.shape[1]] = features
            features = padded_features
        elif features.shape[1] > tril_size:
            # 截断
            features = features[:, :tril_size]
        
        # 构造下三角矩阵
        tril_indices = torch.tril_indices(dim, dim, device=features.device)
        
        # 初始化矩阵
        matrix = torch.zeros(B, dim, dim, device=features.device)
        
        # 填充下三角部分
        matrix[:, tril_indices[0], tril_indices[1]] = features
        
        # 确保对角元素为正
        diag_indices = torch.arange(dim, device=features.device)
        matrix[:, diag_indices, diag_indices] = torch.abs(matrix[:, diag_indices, diag_indices]) + 1e-6
        
        # 构造SPD矩阵: A^T A
        spd_matrix = torch.bmm(matrix.transpose(1, 2), matrix)
        
        return spd_matrix
    
    def _construct_simple_spd_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """
        从特征向量构造严格正定的SPD矩阵
        Args:
            features: 特征向量 [B, dim]
        Returns:
            spd_matrix: SPD矩阵 [B, dim, dim]
        """
        B, dim = features.shape
        
        # 构造严格正定的SPD矩阵
        # 1. 创建对角矩阵，确保对角元素足够大
        diag_values = torch.abs(features) + 1e-2  # 增加最小对角值
        diag_matrix = torch.diag_embed(diag_values)  # [B, dim, dim]
        
        # 2. 添加小的随机非对角元素，但保持对称
        noise = torch.randn(B, dim, dim, device=features.device) * 0.005  # 减小噪声
        noise = (noise + noise.transpose(1, 2)) / 2  # 确保对称
        
        # 3. 组合矩阵
        spd_matrix = diag_matrix + noise
        
        # 4. 确保严格正定性：添加更大的对角项
        identity = torch.eye(dim, device=features.device).unsqueeze(0).expand(B, -1, -1)
        spd_matrix = spd_matrix + 1e-1 * identity  # 增加正则化
        
        # 5. 检查最小特征值，如果太小则增加正则化
        try:
            eigenvals = torch.linalg.eigvals(spd_matrix).real
            min_eigenvals = eigenvals.min(dim=1)[0]
            # 如果最小特征值小于阈值，增加正则化
            for i in range(B):
                if min_eigenvals[i] < 1e-3:
                    spd_matrix[i] = spd_matrix[i] + (1e-3 - min_eigenvals[i]) * identity[i]
        except:
            # 如果特征值计算失败，使用更强的正则化
            spd_matrix = spd_matrix + 1e-1 * identity
        
        # 6. 最终检查并修复数值问题
        if torch.isnan(spd_matrix).any() or torch.isinf(spd_matrix).any():
            print("警告：SPD矩阵包含NaN或Inf，使用单位矩阵")
            spd_matrix = identity
        
        return spd_matrix
                
    def get_matching_loss(self, dual_sim,image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat):
        """
        Matching Loss with hard negatives
        """
        bs = image_embeds.size(0)
        with torch.no_grad():
            sim_i2t=dual_sim
            sim_t2i = dual_sim.t()
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-5
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-5
           
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        #---------------------------------------------------------------------------------
        image_embeds_neg = []
        image_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()  # 概率随机采样
            image_embeds_neg.append(image_embeds[neg_idx])
            image_atts_neg.append(image_atts[neg_idx])

        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        image_atts_neg = torch.stack(image_atts_neg, dim=0)
        #---------------------------------------------------------------------------------


        #---------------------------------------------------------------------------------
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])

        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)
        #---------------------------------------------------------------------------------

        #叠加
        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts_neg, image_atts], dim=0)

        #[120]
        cross_pos = self.get_cross_embeds(image_embeds, image_atts, text_embeds=text_embeds,
                                           text_atts=text_atts)[:, 0,:]
        
        #[240]
        cross_neg = self.get_cross_embeds(image_embeds_all, image_atts_all, text_embeds=text_embeds_all,
                                          text_atts=text_atts_all)[:, 0, :]
        
        #[360]
        output = self.itm_head(torch.cat([cross_pos, cross_neg], dim=0))
        
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long),
                                torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(image_embeds.device)

        return F.cross_entropy(output, itm_labels)
        
    def forward(self, img,  text_ids, text_atts, return_structure_feats=False):

        # 获取CLS token、文本特征、序列特征和中间feature map
        v_embeds_cls, t_embeds, v_embeds, intermediate_feat = self.av_model(
            img, text_ids, text_atts, return_intermediate=True
        )
       
        # 语义特征：CLS token (通过RestV2完整流程)
        v_cls = self.vision_proj(v_embeds_cls)  # [B, 512]
        
        # 图像结构分支：SPDNet
        spd_output_img = self.spdconv(intermediate_feat)   # [B, 3136, 96]
        spd_output_img = self.spd_mlp_img(spd_output_img)  # [B, 3136, 96]
        spd_feat_img = self.spd_proj(spd_output_img.mean(dim=1))   # [B, 512]

        # 文本结构分支：协方差矩阵->展平->MLP+BN->结构特征
        t_cov = self.compute_spd_matrix(t_embeds)  # [B, D, D]
        t_cov_flat = t_cov.view(t_cov.size(0), -1)  # [B, D*D]
        txt_struct_feat = self.text_cov_mlp(t_cov_flat)  # [B, 96]
        spd_feat_txt = self.spd_proj_text(txt_struct_feat)  # [B, 512]

        print("spd_feat_img mean:", spd_feat_img.mean().item(), "std:", spd_feat_img.std().item(), "max:", spd_feat_img.max().item(), "min:", spd_feat_img.min().item())
        print("spd_feat_txt mean:", spd_feat_txt.mean().item(), "std:", spd_feat_txt.std().item(), "max:", spd_feat_txt.max().item(), "min:", spd_feat_txt.min().item())


        # 构造SPD矩阵用于结构对齐
        print("开始构造SPD矩阵...")
        # 图像SPD矩阵：从spd_output_img构造
        img_spd_seq = spd_output_img.mean(dim=1)  # [B, 96]
        print(f"img_spd_seq shape: {img_spd_seq.shape}")
        # 对于96维特征，需要4656个元素来构造96x96的下三角矩阵
        # 我们使用一个更简单的方法：直接构造对角占优的SPD矩阵
        img_spd_matrix = self._construct_simple_spd_matrix(img_spd_seq)  # [B, 96, 96]
        print(f"img_spd_matrix shape: {img_spd_matrix.shape}")
        
        # 文本SPD矩阵：从txt_struct_feat构造
        txt_spd_matrix = self._construct_simple_spd_matrix(txt_struct_feat)  # [B, 96, 96]
        print(f"txt_spd_matrix shape: {txt_spd_matrix.shape}")
        print("SPD矩阵构造完成")
        
        # 尝试计算LESA损失，如果失败则返回零损失
        print("开始计算LESA损失...")
        try:
            structure_loss, loss_dict = self.spd_align_loss(img_spd_matrix, txt_spd_matrix)
            print(f"LESA损失计算成功: {structure_loss.item():.6f}")
        except Exception as e:
            print(f"LESA损失计算失败: {e}")
            # 如果LESA失败，返回零损失
            device = img_spd_matrix.device
            structure_loss = torch.tensor(0.0, device=device)
            loss_dict = {
                'mda_loss': 0.0,
                'cda_loss': 0.0,
                'total_structure_loss': 0.0
            }
        
        # 注意力融合
        fusion_input = torch.stack([v_cls, spd_feat_img], dim=1)  # [B, 2, 512]
        fusion_weight = self.attn_weights(fusion_input)       # [B, 2, 1]
        fusion_weight = fusion_weight.softmax(dim=1)

        v_feature = (fusion_input * fusion_weight).sum(dim=1) # 加权求和 [B, 512]

        # 文本特征
        t_feature = self.text_proj(t_embeds[:, 0, :])  # CLS token

        # 使用双通道相似度计算
        dual_sim = cosine_sim(v_feature, t_feature)

        #很小，感觉没必要 匹配损失 Geo——1652里面
        #itc = self.get_matching_loss(dual_sim,v_embeds, v_atts, v_feature, t_embeds, text_atts, t_feature)
        
        if return_structure_feats:
            return dual_sim, spd_feat_img, spd_feat_txt, structure_loss, loss_dict
        return dual_sim
    

        


def factory(opt, cuda=True, rank=0):
    opt = copy.copy(opt)
    device = torch.device(f"cuda:{rank}")
    model = BaseModel(opt)

    
    if cuda:
        model.to(device)

    for name, param in model.named_parameters():
        if 'spd' in name:
            print(name, param.requires_grad)

    for name, param in model.named_parameters():
        if 'spd' in name and param.grad is not None:
            print(name, param.grad.abs().mean().item())

    return model
    

        





