import math
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
#import spdnets.cplx.nn as nn_cplx
from . import modules as modules
#from spdnets.SPDMLR import SPDRMLR


# class SPDNet(nn.Module):
#     def __init__(self,opt):
#         super(__class__, self).__init__()
        
#         dims = [int(dim) for dim in opt['spdnet']['architecture']]
        
#         self.feature = []
        
#         for i in range(len(dims) - 2):
#             shape=[dims[i], dims[i + 1]]
#             self.feature.append(modules.BiMap(shape,init_mode=opt['spdnet']['init_mode'],manifold=opt['spdnet']['bimap_manifold']))
#             self.feature.append(modules.ReEig())

#         self.feature.append(modules.BiMap([dims[-2], dims[-1]],init_mode=opt['spdnet']['init_mode'],manifold=opt['spdnet']['bimap_manifold']))
#         self.feature.append(modules.LogEig(1))
#         self.feature = nn.Sequential(*self.feature)

#         #self.construct_classifier(args.classifier,dims[-1],args.class_num,args.metric,args.power,args.alpha,args.beta)

#     def forward(self, x):
#         #x = process_images_to_covariance(x, patch_size=16)
#         x = compute_covariance_per_channel(x)
#         x_spd = self.feature(x)
        
#         #y = self.classifier(x_spd)
#         out = x_spd.view(x_spd.size(0), -1)
       
#         return out


class SPDNet(nn.Module):
    def __init__(self, opt):
        super(SPDNet, self).__init__()
        
        dims = [int(dim) for dim in opt['spdnet']['architecture']]
        self.proj = nn.Conv2d(1, 96, kernel_size=1)  # 在 __init__ 里添加

        self.features = nn.ModuleList()
        
        feature=[]
          
        for i in range(len(dims) - 2):
            shape = [dims[i], dims[i + 1]]
            feature.append(modules.BiMap(shape, init_mode=opt['spdnet']['init_mode'], manifold=opt['spdnet']['bimap_manifold']))
            feature.append(modules.ReEig())
            
        feature.append(modules.BiMap([dims[-2], dims[-1]], init_mode=opt['spdnet']['init_mode'], manifold=opt['spdnet']['bimap_manifold']))
        feature.append(modules.LogEig(1))
            
        # 将 feature 转换为 nn.Sequential 并添加到 ModuleList
        self.features = nn.Sequential(*feature)

    def forward(self, x):
        # 检查输入类型
        if len(x.shape) == 4:  # [B, channels, height, width] - 图像格式（中间feature map）
            return self.forward_image(x)
        elif len(x.shape) == 3:  # [B, dim, dim] - 协方差矩阵
            return self.forward_covariance(x)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
    
    def forward_image(self, x):
        """
        处理图像格式输入（中间feature map）
        Args:
            x: 中间feature map [B, channels, height, width]
        Returns:
            输出特征 [B, seq_len, feat_dim]
        """
        batch_size, channels, height, width = x.shape
        
        # 确保输入尺寸适合SPD处理
        if height != 56 or width != 56:
            x = F.interpolate(x, size=(56, 56), mode='bilinear', align_corners=False)
        
        # 智能选择patch_size以获得接近768的维度
        target_dim = 768
        best_patch_size = None
        best_dim_diff = float('inf')
        
        for patch_size in [1, 2, 4, 8, 16]:
            if height % patch_size == 0 and width % patch_size == 0:
                flattened_dim = channels * patch_size * patch_size
                dim_diff = abs(flattened_dim - target_dim)
                if dim_diff < best_dim_diff:
                    best_dim_diff = dim_diff
                    best_patch_size = patch_size
        
        if best_patch_size is None:
            best_patch_size = 2  # 默认值
        
        # 将图像转换为协方差矩阵
        x = process_images_to_covariance(x, patch_size=best_patch_size)  # [B, flattened_dim, flattened_dim]
        
        # 检查协方差矩阵的维度
        cov_dim = x.shape[-1]
        expected_dim = self.features[0].W.shape[-2]  # 第一层BiMap期望的输入维度
        
        # 如果协方差矩阵维度不匹配，进行维度适配
        if cov_dim != expected_dim:
            if cov_dim > expected_dim:
                # 降维：使用PCA-like方法
                # 计算特征值和特征向量
                eigenvals, eigenvecs = torch.linalg.eigh(x)
                # 选择最大的expected_dim个特征值对应的特征向量
                eigenvecs = eigenvecs[:, :, -expected_dim:]  # [B, cov_dim, expected_dim]
                # 投影到较低维度
                x = torch.bmm(torch.bmm(eigenvecs.transpose(-2, -1), x), eigenvecs)
            else:
                # 升维：填充零
                new_x = torch.zeros(batch_size, expected_dim, expected_dim, device=x.device, dtype=x.dtype)
                new_x[:, :cov_dim, :cov_dim] = x
                x = new_x
        
        # 通过SPD特征提取
        spd_feature = self.features(x)  # [B, 56*56]
        
        # 重塑为序列格式
        spd_feature = spd_feature.view(batch_size, 56, 56)  # [B, 56, 56]
        spd_feature = spd_feature.unsqueeze(1)  # [B, 1, 56, 56]
        
        # 投影到96维
        spd_feature = self.proj(spd_feature)  # [B, 96, 56, 56]
        
        # 展平为序列格式
        outputs = spd_feature.view(batch_size, 96, -1)  # [B, 96, 3136]
        out = outputs.transpose(1, 2).contiguous()  # [B, 3136, 96]
        
        return out

    def forward_covariance(self, x):
        """
        直接处理协方差矩阵输入
        Args:
            x: 协方差矩阵 [B, dim, dim]
        Returns:
            输出特征 [B, seq_len, feat_dim]
        """
        batch_size = x.shape[0]
        
        # 直接通过SPD特征提取
        spd_feature = self.features(x)  # [B, 56*56]
        
        # 重塑为序列格式
        spd_feature = spd_feature.view(batch_size, 56, 56)  # [B, 56, 56]
        spd_feature = spd_feature.unsqueeze(1)  # [B, 1, 56, 56]
        
        # 投影到96维
        spd_feature = self.proj(spd_feature)  # [B, 96, 56, 56]
        
        # 展平为序列格式
        outputs = spd_feature.view(batch_size, 96, -1)  # [B, 96, 3136]
        out = outputs.transpose(1, 2).contiguous()  # [B, 3136, 96]
        
        return out


# def compute_covariance_per_channel(input_tensor):
#     """
#     对输入张量的每个 [8, 8] 矩阵计算协方差。

#     Args:
#         input_tensor (torch.Tensor): 输入张量，形状为 [batch_size, channels, height, width]。

#     Returns:
#         torch.Tensor: 输出协方差矩阵，形状为 [batch_size, channels, height, height]。
#     """
#     batch_size, channels, height, width = input_tensor.shape
#     assert height == width, "输入的每个通道特征图必须是方阵"

#     # 初始化协方差矩阵
#     covariance_matrices = torch.zeros(batch_size, channels, height, height, device=input_tensor.device)

#     # 遍历每个样本和通道
#     for b in range(batch_size):
#         for c in range(channels):
#             # 获取 [8, 8] 矩阵
#             feature_map = input_tensor[b, c, :, :]  # [8, 8]

#             # 去均值
#             feature_map = feature_map - feature_map.mean(dim=0, keepdim=True)

#             # 计算协方差矩阵
#             covariance_matrix = feature_map @ feature_map.T / (width - 1)  # [8, 8]
#             covariance_matrices[b, c, :, :] = covariance_matrix

#     return covariance_matrices


def process_images_to_covariance(images, patch_size=None):
    """
    将输入图像划分为 patch，展平特征后计算协方差矩阵。

    Args:
        images (torch.Tensor): 输入图像，形状为 [batch_size, channels, height, width]。
        patch_size (int): 每个 patch 的大小，如果为None则自动计算。

    Returns:
        torch.Tensor: 每张图片的协方差矩阵，形状为 [batch_size, flattened_dim, flattened_dim]。
    """
    batch_size, channels, height, width = images.shape
    
    # 自动计算合适的patch_size
    if patch_size is None:
        # 对于中间feature map，使用较小的patch_size以保持更多细节
        if height <= 14 and width <= 14:
            patch_size = 2  # 对于小尺寸feature map
        elif height <= 28 and width <= 28:
            patch_size = 4  # 对于中等尺寸feature map
        elif height <= 56 and width <= 56:
            patch_size = 8  # 对于较大尺寸feature map
        else:
            patch_size = 16  # 默认值
    
    # 如果图像尺寸不能被patch_size整除，使用插值调整
    if height % patch_size != 0 or width % patch_size != 0:
        # 计算最接近的能被patch_size整除的尺寸
        target_height = ((height + patch_size - 1) // patch_size) * patch_size
        target_width = ((width + patch_size - 1) // patch_size) * patch_size
        
        # 使用插值调整尺寸
        images = F.interpolate(images, size=(target_height, target_width), 
                              mode='bilinear', align_corners=False)
        height, width = target_height, target_width

    # 1. 划分图像为 patch
    patches = F.unfold(images, kernel_size=patch_size, stride=patch_size)  # [batch_size, channels * patch_size^2, num_patches]
    num_patches = (height // patch_size) * (width // patch_size)  # 每张图像的 patch 数量

    # 2. 转置并展平特征
    patches = patches.transpose(1, 2)  # [batch_size, num_patches, channels * patch_size^2]

    # 3. 计算协方差矩阵
    # 对每张图片的 patch 特征矩阵计算协方差
    covariance_matrices = []
    for i in range(batch_size):
        patch_features = patches[i]  # [num_patches, flattened_dim]
        patch_features = patch_features - patch_features.mean(dim=0, keepdim=True)  # 去均值
        covariance_matrix = patch_features.T @ patch_features / (num_patches - 1)  # 协方差矩阵
        covariance_matrices.append(covariance_matrix)

    # 堆叠所有协方差矩阵
    covariance_matrices = torch.stack(covariance_matrices, dim=0)  # [batch_size, flattened_dim, flattened_dim]

    return covariance_matrices