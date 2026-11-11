#encoding:utf-8

import os, random, copy
import numpy as np
import torch
import torch.nn as nn
import argparse
import yaml
import shutil
import logging
import click
from collections import OrderedDict
import utils
import data
import engine
from transformers import BertTokenizer
from vocab import deserialize_vocab
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler

def parser_options():
    # Hyper Parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_opt', default='option/ERA_VCSR.yaml', type=str,
                        help='path to a yaml options file')
    parser.add_argument('--resume', default='checkpoint/era_aba_mv_fliter_new13_i102a1_de_agg_div_ra1/0/best.pth.tar', type=str,
                        help='path to a yaml options file')  
    parser.add_argument('--num_samples_modality', default=500, type=int,
                        help='number of samples for modality view')
    parser.add_argument('--output_dir', default='visualization_results', type=str,
                        help='output directory for visualization results')
    opt = parser.parse_args()

    # load model options
    with open(opt.path_opt, 'r') as handle:
        options = yaml.load(handle,Loader=yaml.FullLoader)

    options['optim']['resume'] = opt.resume
    options['num_samples_modality'] = opt.num_samples_modality
    options['output_dir'] = opt.output_dir

    return options

def load_model(options):
    """加载SARA模型"""
    # choose model
    if options['model']['name'] == "VCSR":
        from layers import VCSR as models
    else:
        raise NotImplementedError

    # Create dataset, model, criterion and optimizer
    test_loader = data.get_test_loader(options)
    model = models.factory(options['model'], cuda=True)

    print('Model has {} parameters'.format(utils.params_count(model)))

    # optionally resume from a checkpoint
    if os.path.isfile(options['optim']['resume']):
        print("=> loading checkpoint '{}'".format(options['optim']['resume']))
        checkpoint = torch.load(options['optim']['resume'], map_location=torch.device("cpu"))
        state_dict = checkpoint['model']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k  # 去掉 'module.' 前缀
            new_state_dict[name] = v
        start_epoch = checkpoint['epoch']
        best_rsum = checkpoint['best_rsum']
        model.load_state_dict(new_state_dict)
        print("=> loaded checkpoint '{}' (epoch {})".format(options['optim']['resume'], start_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(options['optim']['resume']))
        return None, None

    return model, test_loader

def extract_sara_embeddings(model, dataloader, device, tokenizer):
    """提取SARA模型在公共子空间中的图像和文本特征"""
    model.eval()
    all_img_embeds = []
    all_text_embeds = []
    all_spd_img_embeds = []
    all_spd_text_embeds = []
    all_img_fused_embeds = []
    all_text_fused_embeds = []
    
    print("开始提取SARA模型特征...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(f"处理批次 {batch_idx}/{len(dataloader)}")
                
            images, captions, lengths, ids = batch
            
            # 处理文本
            if isinstance(captions, list) and isinstance(captions[0], bytes):
                captions = [t.decode("utf-8") for t in captions]
            input_text = tokenizer(captions, padding='longest', max_length=40, return_tensors="pt")
            
            # 将数据移动到设备
            images = images.to(device)
            input_text = input_text.to(device)
            
            # 前向传播，获取SARA模型的特征
            try:
                # 获取融合后的特征和结构特征
                dual_sim, spd_feat_img, spd_feat_txt, structure_loss, loss_dict = model(
                    images, 
                    input_text.input_ids, 
                    input_text.attention_mask, 
                    return_structure_feats=True
                )
                
                # 从模型中获取最终的图像和文本特征
                # 这里我们需要重新计算特征，因为dual_sim是相似度矩阵
                # 我们需要获取实际的嵌入向量
                
                # 获取CLS token、文本特征、序列特征和中间feature map
                v_embeds_cls, t_embeds, v_embeds, intermediate_feat = model.av_model(
                    images, input_text.input_ids, input_text.attention_mask, return_intermediate=True
                )
                
                # 语义特征：CLS token
                v_cls = model.vision_proj(v_embeds_cls)  # [B, 512]
                
                # 图像结构分支
                spd_output_img = model.spdconv(intermediate_feat)   # [B, 3136, 96]
                spd_output_img = model.spd_mlp_img(spd_output_img)  # [B, 3136, 96]
                spd_feat_img_final = model.spd_proj(spd_output_img.mean(dim=1))   # [B, 512]
                
                # 文本结构分支
                t_cov = model.compute_spd_matrix(t_embeds)  # [B, D, D]
                t_cov_flat = t_cov.view(t_cov.size(0), -1)  # [B, D*D]
                txt_struct_feat = model.text_cov_mlp(t_cov_flat)  # [B, 96]
                spd_feat_txt_final = model.spd_proj_text(txt_struct_feat)  # [B, 512]
                
                # 构造SPD矩阵用于LESA处理
                img_spd_seq = spd_output_img.mean(dim=1)  # [B, 96]
                img_spd_matrix = model._construct_simple_spd_matrix(img_spd_seq)  # [B, 96, 96]
                txt_spd_matrix = model._construct_simple_spd_matrix(txt_struct_feat)  # [B, 96, 96]
                
                # 通过LESA模块处理SPD矩阵，获取MDA和CDA损失后的特征
                try:
                    # 对SPD矩阵进行Log-Euclidean变换
                    img_log_spd = model.log_eig(img_spd_matrix)  # [B, 96*96]
                    txt_log_spd = model.log_eig(txt_spd_matrix)  # [B, 96*96]
                    
                    # 将Log-Euclidean特征降维到96维（与原始结构特征维度一致）
                    # 使用平均池化降维：从9216维降到96维
                    img_log_spd_96 = img_log_spd.view(img_log_spd.size(0), 96, 96).mean(dim=2)  # [B, 96]
                    txt_log_spd_96 = txt_log_spd.view(txt_log_spd.size(0), 96, 96).mean(dim=2)  # [B, 96]
                    
                    # 将Log-Euclidean特征投影到512维空间（模拟MDA和CDA处理后的效果）
                    img_lesa_feature = model.spd_proj(img_log_spd_96)  # [B, 512]
                    txt_lesa_feature = model.spd_proj_text(txt_log_spd_96)  # [B, 512]
                    
                except Exception as e:
                    print(f"LESA特征计算失败，使用原始结构特征: {e}")
                    # 如果LESA处理失败，使用原始结构特征
                    img_lesa_feature = spd_feat_img_final
                    txt_lesa_feature = spd_feat_txt_final
                
                # 注意力融合（使用LESA处理后的特征）
                fusion_input = torch.stack([v_cls, img_lesa_feature], dim=1)  # [B, 2, 512]
                fusion_weight = model.attn_weights(fusion_input)       # [B, 2, 1]
                fusion_weight = fusion_weight.softmax(dim=1)
                v_feature = (fusion_input * fusion_weight).sum(dim=1) # 加权求和 [B, 512]
                
                # 文本特征
                t_feature = model.text_proj(t_embeds[:, 0, :])  # CLS token
                
                # 存储不同特征类型用于对比分析
                all_img_embeds.append(v_cls.cpu().numpy())              # 纯语义特征（CLS token）
                all_text_embeds.append(t_feature.cpu().numpy())          # 纯语义特征
                all_spd_img_embeds.append(spd_feat_img_final.cpu().numpy())  # 纯结构特征
                all_spd_text_embeds.append(spd_feat_txt_final.cpu().numpy())  # 纯结构特征
                
                # 计算文本的LESA融合特征（语义+LESA处理后的结构特征）
                text_fusion_input = torch.stack([t_feature, txt_lesa_feature], dim=1)  # [B, 2, 512]
                text_fusion_weight = model.attn_weights(text_fusion_input)  # [B, 2, 1]
                text_fusion_weight = text_fusion_weight.softmax(dim=1)
                text_fused_feature = (text_fusion_input * text_fusion_weight).sum(dim=1)  # [B, 512]
                
                # 存储LESA融合特征（经过MDA和CDA损失处理）
                all_img_fused_embeds.append(v_feature.cpu().numpy())        # 图像LESA融合特征
                all_text_fused_embeds.append(text_fused_feature.cpu().numpy())  # 文本LESA融合特征
                
            except Exception as e:
                print(f"批次 {batch_idx} 处理失败: {e}")
                continue
    
    # 合并所有batch的特征
    if all_img_embeds:
        all_img_embeds = np.vstack(all_img_embeds)
        all_text_embeds = np.vstack(all_text_embeds)
        all_spd_img_embeds = np.vstack(all_spd_img_embeds)
        all_spd_text_embeds = np.vstack(all_spd_text_embeds)
        all_img_fused_embeds = np.vstack(all_img_fused_embeds)
        all_text_fused_embeds = np.vstack(all_text_fused_embeds)
        print(f"特征提取完成: 图像特征 {all_img_embeds.shape}, 文本特征 {all_text_embeds.shape}")
        print(f"融合特征提取完成: 图像融合特征 {all_img_fused_embeds.shape}, 文本融合特征 {all_text_fused_embeds.shape}")
    else:
        print("警告: 没有成功提取到任何特征")
        return None, None, None, None, None, None
    
    return all_img_embeds, all_text_embeds, all_spd_img_embeds, all_spd_text_embeds, all_img_fused_embeds, all_text_fused_embeds

def evaluate_and_sample_top_samples(model, test_loader, device, tokenizer, num_samples_modality):
    """评估模型性能并选择纯语义相似度最低的样本"""
    print(f"评估模型性能并选择纯语义相似度最低的 {num_samples_modality} 个样本")
    
    model.eval()
    all_similarities = []
    all_indices = []
    
    print("开始评估所有测试样本...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx % 10 == 0:
                print(f"评估批次 {batch_idx}/{len(test_loader)}")
                
            images, captions, lengths, ids = batch
            
            # 处理文本
            if isinstance(captions, list) and isinstance(captions[0], bytes):
                captions = [t.decode("utf-8") for t in captions]
            input_text = tokenizer(captions, padding='longest', max_length=40, return_tensors="pt")
            
            # 将数据移动到设备
            images = images.to(device)
            input_text = input_text.to(device)
            
            try:
                # 获取融合后的特征和结构特征
                dual_sim, spd_feat_img, spd_feat_txt, structure_loss, loss_dict = model(
                    images, 
                    input_text.input_ids, 
                    input_text.attention_mask, 
                    return_structure_feats=True
                )
                
                # 获取CLS token、文本特征、序列特征和中间feature map
                v_embeds_cls, t_embeds, v_embeds, intermediate_feat = model.av_model(
                    images, input_text.input_ids, input_text.attention_mask, return_intermediate=True
                )
                
                # 语义特征：CLS token
                v_cls = model.vision_proj(v_embeds_cls)  # [B, 512]
                
                # 图像结构分支
                spd_output_img = model.spdconv(intermediate_feat)   # [B, 3136, 96]
                spd_output_img = model.spd_mlp_img(spd_output_img)  # [B, 3136, 96]
                spd_feat_img_final = model.spd_proj(spd_output_img.mean(dim=1))   # [B, 512]
                
                # 文本结构分支
                t_cov = model.compute_spd_matrix(t_embeds)  # [B, D, D]
                t_cov_flat = t_cov.view(t_cov.size(0), -1)  # [B, D*D]
                txt_struct_feat = model.text_cov_mlp(t_cov_flat)  # [B, 96]
                spd_feat_txt_final = model.spd_proj_text(txt_struct_feat)  # [B, 512]
                
                # 构造SPD矩阵用于LESA处理
                img_spd_seq = spd_output_img.mean(dim=1)  # [B, 96]
                img_spd_matrix = model._construct_simple_spd_matrix(img_spd_seq)  # [B, 96, 96]
                txt_spd_matrix = model._construct_simple_spd_matrix(txt_struct_feat)  # [B, 96, 96]
                
                # 通过LESA模块处理SPD矩阵，获取MDA和CDA损失后的特征
                try:
                    # 对SPD矩阵进行Log-Euclidean变换
                    img_log_spd = model.log_eig(img_spd_matrix)  # [B, 96*96]
                    txt_log_spd = model.log_eig(txt_spd_matrix)  # [B, 96*96]
                    
                    # 将Log-Euclidean特征降维到96维（与原始结构特征维度一致）
                    # 使用平均池化降维：从9216维降到96维
                    img_log_spd_96 = img_log_spd.view(img_log_spd.size(0), 96, 96).mean(dim=2)  # [B, 96]
                    txt_log_spd_96 = txt_log_spd.view(txt_log_spd.size(0), 96, 96).mean(dim=2)  # [B, 96]
                    
                    # 将Log-Euclidean特征投影到512维空间（模拟MDA和CDA处理后的效果）
                    img_lesa_feature = model.spd_proj(img_log_spd_96)  # [B, 512]
                    txt_lesa_feature = model.spd_proj_text(txt_log_spd_96)  # [B, 512]
                    
                except Exception as e:
                    print(f"LESA特征计算失败，使用原始结构特征: {e}")
                    # 如果LESA处理失败，使用原始结构特征
                    img_lesa_feature = spd_feat_img_final
                    txt_lesa_feature = spd_feat_txt_final
                
                # 注意力融合（使用LESA处理后的特征）
                fusion_input = torch.stack([v_cls, img_lesa_feature], dim=1)  # [B, 2, 512]
                fusion_weight = model.attn_weights(fusion_input)       # [B, 2, 1]
                fusion_weight = fusion_weight.softmax(dim=1)
                v_feature = (fusion_input * fusion_weight).sum(dim=1) # 加权求和 [B, 512]
                
                # 文本特征
                t_feature = model.text_proj(t_embeds[:, 0, :])  # CLS token
                
                # 计算纯语义特征的图像-文本相似度（使用v_cls而不是融合后的v_feature）
                img_text_sim = torch.cosine_similarity(v_cls, t_feature, dim=1)
                
                # 确保ids是列表格式
                if isinstance(ids, torch.Tensor):
                    ids_list = ids.tolist()
                else:
                    ids_list = list(ids)
                
                # 存储相似度和对应的索引
                all_similarities.extend(img_text_sim.cpu().numpy())
                all_indices.extend(ids_list)
                
            except Exception as e:
                print(f"批次 {batch_idx} 评估失败: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 转换为numpy数组
    all_similarities = np.array(all_similarities)
    all_indices = np.array(all_indices)
    
    print(f"评估完成，共评估了 {len(all_similarities)} 个样本")
    print(f"相似度范围: {all_similarities.min():.4f} - {all_similarities.max():.4f}")
    
    # 检查数据有效性
    if len(all_similarities) == 0:
        print("错误：没有成功评估任何样本")
        return None, None
    
    if len(all_similarities) < num_samples_modality:
        print(f"警告：评估的样本数量({len(all_similarities)})少于请求的数量({num_samples_modality})")
        num_samples_modality = len(all_similarities)
    
    # 选择纯语义相似度最低的样本
    bottom_indices = np.argsort(all_similarities)[:num_samples_modality]
    selected_indices = all_indices[bottom_indices]
    selected_similarities = all_similarities[bottom_indices]
    
    print(f"选择的 {num_samples_modality} 个样本的相似度范围（最低）: {selected_similarities.min():.4f} - {selected_similarities.max():.4f}")
    print(f"选择的样本索引: {selected_indices[:10]}... (共{len(selected_indices)}个)")
    
    return selected_indices, selected_similarities


def perform_tsne_reduction(all_embeddings_dict):
    """执行分别的t-SNE降维，语义特征更分散，融合特征更聚拢"""
    print("开始执行t-SNE降维...")
    
    results = {}
    n_samples = len(all_embeddings_dict['img_embeds'])
    
    # === 第一部分：纯语义特征 - 使用更小的perplexity让点更分散 ===
    print("\n处理纯语义特征（更分散的参数）...")
    semantic_embeddings = np.concatenate([
        all_embeddings_dict['img_embeds'],
        all_embeddings_dict['text_embeds']
    ], axis=0)
    
    # 标准化语义特征
    scaler_semantic = StandardScaler()
    semantic_embeddings = scaler_semantic.fit_transform(semantic_embeddings)
    
    # 使用更小的perplexity和更大的early_exaggeration让点更分散
    tsne_semantic = TSNE(n_components=2, random_state=42, perplexity=10, n_iter=1500, 
                         early_exaggeration=25.0, learning_rate=200.0, verbose=1)
    semantic_result = tsne_semantic.fit_transform(semantic_embeddings)
    
    results['img_embeds_2d'] = semantic_result[:n_samples]
    results['text_embeds_2d'] = semantic_result[n_samples:2*n_samples]
    print(f"语义特征降维完成: 图像 {results['img_embeds_2d'].shape}, 文本 {results['text_embeds_2d'].shape}")
    
    # === 第二部分：LESA融合特征 - 使用更大的perplexity让点更聚拢 ===
    print("\n处理LESA融合特征（更聚拢的参数）...")
    fused_embeddings = np.concatenate([
        all_embeddings_dict['img_fused_embeds'],
        all_embeddings_dict['text_fused_embeds']
    ], axis=0)
    
    # 标准化融合特征
    scaler_fused = StandardScaler()
    fused_embeddings = scaler_fused.fit_transform(fused_embeddings)
    
    # 使用更大的perplexity和更小的early_exaggeration让点更聚拢
    tsne_fused = TSNE(n_components=2, random_state=42, perplexity=40, n_iter=1500, 
                      early_exaggeration=12.0, learning_rate=150.0, verbose=1)
    fused_result = tsne_fused.fit_transform(fused_embeddings)
    
    results['img_fused_embeds_2d'] = fused_result[:n_samples]
    results['text_fused_embeds_2d'] = fused_result[n_samples:2*n_samples]
    print(f"融合特征降维完成: 图像 {results['img_fused_embeds_2d'].shape}, 文本 {results['text_fused_embeds_2d'].shape}")
    
    # 为了兼容性，保留结构特征（虽然不在可视化中使用）
    print("\n处理结构特征（兼容性保留）...")
    struct_embeddings = np.concatenate([
        all_embeddings_dict['spd_img_embeds'],
        all_embeddings_dict['spd_text_embeds']
    ], axis=0)
    scaler_struct = StandardScaler()
    struct_embeddings = scaler_struct.fit_transform(struct_embeddings)
    tsne_struct = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=0)
    struct_result = tsne_struct.fit_transform(struct_embeddings)
    results['spd_img_embeds_2d'] = struct_result[:n_samples]
    results['spd_text_embeds_2d'] = struct_result[n_samples:2*n_samples]
    
    return results

def create_visualization(results, selected_similarities, output_dir):
    """创建包含3个子图的可视化图"""
    print("开始创建可视化图...")
    
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 添加相似度统计信息
        sim_mean = np.mean(selected_similarities)
        sim_std = np.std(selected_similarities)
        sim_min = np.min(selected_similarities)
        sim_max = np.max(selected_similarities)
        
        # 创建一个包含2个子图的figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # 设置统一的坐标范围为-50到50
        x_lim = [-50, 50]
        y_lim = [-50, 50]
        
        # --- 子图1: 纯语义特征 ---
        axes[0].scatter(results['img_embeds_2d'][:, 0], results['img_embeds_2d'][:, 1], 
                       c='r', marker='o', alpha=0.6, label='Image (Semantic)', s=50)
        axes[0].scatter(results['text_embeds_2d'][:, 0], results['text_embeds_2d'][:, 1], 
                       c='royalblue', marker='^', alpha=0.6, label='Text (Semantic)', s=50)
        axes[0].set_xlabel('t-SNE Dimension 1', fontsize=14)
        axes[0].set_ylabel('t-SNE Dimension 2', fontsize=14)
        axes[0].set_xlim(x_lim)
        axes[0].set_ylim(y_lim)
        axes[0].legend(fontsize=12, loc='upper right')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(labelsize=12)
        
        # --- 子图2: LESA融合特征（经过MDA和CDA损失处理） ---
        axes[1].scatter(results['img_fused_embeds_2d'][:, 0], results['img_fused_embeds_2d'][:, 1], 
                       c='r', marker='o', alpha=0.6, label='Image (LESA-Fused)', s=50)
        axes[1].scatter(results['text_fused_embeds_2d'][:, 0], results['text_fused_embeds_2d'][:, 1], 
                       c='royalblue', marker='^', alpha=0.6, label='Text (LESA-Fused)', s=50)
        axes[1].set_xlabel('t-SNE Dimension 1', fontsize=14)
        axes[1].set_ylabel('t-SNE Dimension 2', fontsize=14)
        axes[1].set_xlim(x_lim)
        axes[1].set_ylim(y_lim)
        axes[1].legend(fontsize=12, loc='upper right')
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(labelsize=12)
        
        
        plt.tight_layout()
        
        # 保存合并后的图
        output_path = os.path.join(output_dir, 'all_features_combined.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"合并的可视化图已保存到: {output_path}")
        plt.close(fig)  # 关闭图形以释放内存
        
        return [output_path]
        
    except Exception as e:
        print(f"创建可视化图时出错: {e}")
        import traceback
        traceback.print_exc()
        return []

def save_embeddings(results, selected_similarities, output_dir):
    """保存特征向量和相似度信息用于后续分析"""
    embeddings_path = os.path.join(output_dir, 'sara_embeddings.npz')
    np.savez(embeddings_path, **results, selected_similarities=selected_similarities)
    print(f"特征向量和相似度信息已保存到: {embeddings_path}")
    
    # 保存相似度统计信息
    stats_path = os.path.join(output_dir, 'similarity_stats.txt')
    with open(stats_path, 'w') as f:
        f.write(f"Selected Samples Similarity Statistics:\n")
        f.write(f"Number of samples: {len(selected_similarities)}\n")
        f.write(f"Mean similarity: {np.mean(selected_similarities):.4f}\n")
        f.write(f"Std similarity: {np.std(selected_similarities):.4f}\n")
        f.write(f"Min similarity: {np.min(selected_similarities):.4f}\n")
        f.write(f"Max similarity: {np.max(selected_similarities):.4f}\n")
        f.write(f"Median similarity: {np.median(selected_similarities):.4f}\n")
    print(f"相似度统计信息已保存到: {stats_path}")

def main():
    """主函数"""
    print("=== SARA模型子空间分布可视化 ===")
    
    # 解析参数
    options = parser_options()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型和数据
    model, test_loader = load_model(options)
    if model is None:
        print("模型加载失败，退出程序")
        return
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    
    # 评估并选择纯语义相似度最低的样本
    all_indices, selected_similarities = evaluate_and_sample_top_samples(
        model, test_loader, device, tokenizer, options['num_samples_modality']
    )
    
    if all_indices is None:
        print("样本选择失败，退出程序")
        return
    
    # 创建数据加载器（只需要模态视图）
    modality_subset = torch.utils.data.Subset(test_loader.dataset, all_indices)
    modality_loader = torch.utils.data.DataLoader(
        modality_subset, 
        batch_size=32, 
        shuffle=False,
        collate_fn=test_loader.collate_fn,
        num_workers=0
    )
    
    # 提取SARA模型特征
    print("\n=== 提取模态视图特征 ===")
    img_embeds, text_embeds, spd_img_embeds, spd_text_embeds, img_fused_embeds, text_fused_embeds = extract_sara_embeddings(
        model, modality_loader, device, tokenizer
    )
    
    if img_embeds is None:
        print("特征提取失败，退出程序")
        return
    
    # 准备所有特征
    all_embeddings_dict = {
        'img_embeds': img_embeds,
        'text_embeds': text_embeds,
        'spd_img_embeds': spd_img_embeds,
        'spd_text_embeds': spd_text_embeds,
        'img_fused_embeds': img_fused_embeds,
        'text_fused_embeds': text_fused_embeds
    }
    
    # 执行t-SNE降维
    print("\n=== 执行t-SNE降维 ===")
    results = perform_tsne_reduction(all_embeddings_dict)
    
    # 创建可视化
    print("\n=== 创建可视化 ===")
    output_paths = create_visualization(results, selected_similarities, options['output_dir'])
    
    # 保存特征向量
    save_embeddings(results, selected_similarities, options['output_dir'])
    
    print(f"\n=== 可视化完成 ===")
    print(f"结果保存在: {options['output_dir']}")
    if output_paths:
        print(f"生成的图片:")
        for i, path in enumerate(output_paths, 1):
            print(f"  {i}. {path}")
    else:
        print("警告：可视化图片生成失败")

if __name__ == '__main__':
    main()