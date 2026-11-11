#encoding:utf-8

import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import argparse
import yaml
from collections import OrderedDict
import utils
import data
from transformers import BertTokenizer
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import textwrap

def parser_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_opt', default='option/ERA_VCSR.yaml', type=str,
                        help='path to a yaml options file')
    parser.add_argument('--resume', default='checkpoint/era_aba_mv_fliter_new13_i102a1_de_agg_div_ra1/0/best.pth.tar', type=str,
                        help='path to checkpoint file')
    parser.add_argument('--num_examples', type=int, default=20,
                        help='number of examples to visualize')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='random seed for selecting query samples (use -1 for truly random)')
    parser.add_argument('--img_indices', type=str, default=None,
                        help='comma-separated image indices to use as queries (e.g., "0,10,20,30,40")')
    parser.add_argument('--cap_indices', type=str, default=None,
                        help='comma-separated caption indices to use as queries (e.g., "0,50,100,150,200")')
    opt = parser.parse_args()

    with open(opt.path_opt, 'r') as handle:
        options = yaml.load(handle, Loader=yaml.FullLoader)

    options['optim']['resume'] = opt.resume
    options['num_examples'] = opt.num_examples
    options['random_seed'] = opt.random_seed
    options['img_indices'] = opt.img_indices
    options['cap_indices'] = opt.cap_indices

    return options

def load_model_and_data(options):
    """加载模型和测试数据"""
    if options['model']['name'] == "VCSR":
        from layers import VCSR as models
    else:
        raise NotImplementedError

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    test_loader = data.get_test_loader(options)
    
    model = models.factory(options['model'], cuda=True)
    print('Model has {} parameters'.format(utils.params_count(model)))

    # 加载checkpoint
    if os.path.isfile(options['optim']['resume']):
        print("=> loading checkpoint '{}'".format(options['optim']['resume']))
        checkpoint = torch.load(options['optim']['resume'], map_location=torch.device("cpu"))
        state_dict = checkpoint['model']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("=> loaded checkpoint successfully")
    else:
        print("=> no checkpoint found at '{}'".format(options['optim']['resume']))
        return None, None, None, None

    # 加载测试数据信息
    data_path = options['dataset']['data_path']
    image_path = options['dataset']['image_path']
    
    # 加载文本描述和图像文件名
    captions = []
    with open(data_path + 'test_caps.txt', 'rb') as f:
        for line in f:
            cap = line.strip().decode('utf-8', errors='ignore')
            captions.append(cap)
    
    images_raw = []
    with open(data_path + 'test_filename.txt', 'rb') as f:
        for line in f:
            img_name = line.strip().decode('utf-8', errors='ignore')
            # 处理字节字符串格式
            if img_name.startswith("b'") or img_name.startswith("b\""):
                if img_name.startswith("b'"):
                    img_name = img_name[2:-1] if img_name.endswith("'") else img_name[2:]
                else:
                    img_name = img_name[2:-1] if img_name.endswith("\"") else img_name[2:]
            images_raw.append(img_name)
    
    # 根据数据集结构，每5个文本对应1张图像
    # 如果图像列表长度等于文本列表长度，说明图像被重复了，需要去重
    if len(images_raw) == len(captions):
        # 每5个文本对应1张图像，提取唯一的图像
        images = [images_raw[i] for i in range(0, len(images_raw), 5)]
        print(f"Loaded {len(captions)} captions and {len(images_raw)} image entries")
        print(f"Unique images: {len(images)} (each image corresponds to 5 captions)")
    else:
        # 如果已经去重，直接使用
        images = images_raw
        print(f"Loaded {len(captions)} captions and {len(images)} images")
    
    # 验证对应关系
    print(f"\nVerifying correspondence:")
    print(f"  - Caption indices 0-4 should correspond to image index 0")
    print(f"  - Caption indices 5-9 should correspond to image index 1")
    print(f"  - Caption indices 10-14 should correspond to image index 2")
    print(f"  - And so on...")
    
    # 验证几个样本
    if len(captions) >= 10 and len(images) >= 2:
        print(f"\nSample verification:")
        print(f"  - Caption 0 (index 0) -> Image {0 // 5} (should be image 0)")
        print(f"  - Caption 5 (index 5) -> Image {5 // 5} (should be image 1)")
        print(f"  - Caption 10 (index 10) -> Image {10 // 5} (should be image 2)")
    
    return model, test_loader, captions, images, tokenizer

def compute_similarity_matrix(model, test_loader, tokenizer, device=0):
    """计算相似度矩阵"""
    model.eval()
    
    input_visual = np.zeros((len(test_loader.dataset), 3, 224, 224))
    input_text_ids = [None] * len(test_loader.dataset)
    input_text_mask = [None] * len(test_loader.dataset)
    
    for i, val_data in enumerate(test_loader):
        images, captions, lengths, ids = val_data
        
        if isinstance(captions, list) and isinstance(captions[0], bytes):
            captions = [t.decode("utf-8") for t in captions]
        captions_tokenized = tokenizer(captions, padding='longest', max_length=40, return_tensors="pt")
        
        for (id, img, ids, mask) in zip(ids, (images.numpy().copy()), 
                                         (captions_tokenized['input_ids'].numpy().copy()),
                                         (captions_tokenized['attention_mask'].numpy().copy())):
            input_visual[id] = img
            input_text_ids[id] = ids
            input_text_mask[id] = mask
    
    # 每5个文本描述对应一张图像
    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])
    
    print("Computing similarity matrix...")
    sim_matrix = utils.shard_dis(input_visual, input_text_ids, input_text_mask, model, 
                                 shard_size=options['dataset']['batch_size_val'], device=device)
    
    return sim_matrix

def get_top_k_results(sim_matrix, k=5):
    """获取top-k检索结果"""
    num_images = sim_matrix.shape[0]
    num_captions = sim_matrix.shape[1]
    
    # Image-Query-Text: 每张图像检索top-k文本
    i2t_results = []
    for img_idx in range(num_images):
        scores = sim_matrix[img_idx, :]
        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_scores = scores[top_k_indices]
        i2t_results.append({
            'img_idx': img_idx,
            'top_k_indices': top_k_indices,
            'top_k_scores': top_k_scores
        })
    
    # Text-Query-Image: 每个文本检索top-k图像
    t2i_results = []
    for cap_idx in range(num_captions):
        scores = sim_matrix[:, cap_idx]
        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_scores = scores[top_k_indices]
        t2i_results.append({
            'cap_idx': cap_idx,
            'top_k_indices': top_k_indices,
            'top_k_scores': top_k_scores
        })
    
    return i2t_results, t2i_results

def clean_image_filename(img_filename):
    """清理图像文件名，去除字节字符串格式"""
    if isinstance(img_filename, str):
        if img_filename.startswith("b'") and img_filename.endswith("'"):
            return img_filename[2:-1]
        elif img_filename.startswith("b\"") and img_filename.endswith("\""):
            return img_filename[2:-1]
    return img_filename

def load_image(image_path, img_filename):
    """加载图像"""
    # 清理图像文件名
    img_filename = clean_image_filename(img_filename)
    
    # 尝试不同的路径格式
    full_path = os.path.join(image_path, img_filename)
    if os.path.exists(full_path):
        try:
            img = Image.open(full_path).convert('RGB')
            return img
        except Exception as e:
            print(f"Warning: Error loading image {full_path}: {e}")
    
    # 尝试直接拼接路径
    full_path = image_path + img_filename
    if os.path.exists(full_path):
        try:
            img = Image.open(full_path).convert('RGB')
            return img
        except Exception as e:
            print(f"Warning: Error loading image {full_path}: {e}")
    
    # 如果都失败，返回空白图像
    print(f"Warning: Image not found: {img_filename}")
    return Image.new('RGB', (224, 224), color='white')

def visualize_retrieval_results(sim_matrix, captions, images, i2t_results, t2i_results, 
                                options, num_examples=20):
    """可视化检索结果"""
    data_path = options['dataset']['data_path']
    image_path = options['dataset']['image_path']
    
    # 从相似度矩阵获取实际图像数量（因为每5个文本对应1张图像）
    num_images_in_matrix = sim_matrix.shape[0]  # 实际参与计算的图像数量
    num_captions_in_matrix = sim_matrix.shape[1]  # 实际参与计算的文本数量
    
    # i2t_results和t2i_results的长度应该与相似度矩阵的维度一致
    print(f"Similarity matrix: {num_images_in_matrix} images, {num_captions_in_matrix} captions")
    print(f"i2t_results length: {len(i2t_results)}, t2i_results length: {len(t2i_results)}")
    print(f"Available data: {len(images)} images, {len(captions)} captions")
    
    # 使用相似度矩阵中的实际图像数量（确保不超过i2t_results的长度）
    num_images = min(num_images_in_matrix, len(i2t_results), len(images))
    num_captions = min(num_captions_in_matrix, len(t2i_results), len(captions))
    
    # 选择查询样本（只使用图像查询文本）
    num_img_queries_default = num_examples  # 所有样例都用于图像查询
    num_cap_queries_default = 0  # 不使用文本查询
    
    if options.get('img_indices') is not None:
        # 使用指定的图像索引
        img_indices_str = options['img_indices'].split(',')
        selected_img_indices = np.array([int(idx.strip()) for idx in img_indices_str])
        selected_img_indices = selected_img_indices[selected_img_indices < num_images]
        print(f"Using specified image indices: {selected_img_indices}")
    else:
        # 随机选择图像查询
        num_img_queries = min(num_img_queries_default, num_images)
        random_seed = options.get('random_seed', 42)
        if random_seed >= 0:
            np.random.seed(random_seed)  # 固定随机种子以便可复现
            print(f"Using random seed {random_seed} for image selection")
        else:
            print("Using truly random selection for images")
        if num_images > 0:
            selected_img_indices = np.random.choice(num_images, num_img_queries, replace=False)
        else:
            selected_img_indices = np.array([])
            print("Warning: No valid images available")
    
    # 不再需要文本查询索引（只做图像查询文本）
    selected_cap_indices = np.array([])
    
    print(f"\nSelected {len(selected_img_indices)} image queries for text retrieval")
    print(f"Total examples to visualize: {len(selected_img_indices)}")
    
    # 创建figure（只显示图像查询文本）
    actual_num_examples = len(selected_img_indices)
    fig = plt.figure(figsize=(24, 4 * actual_num_examples))
    gs = GridSpec(actual_num_examples, 6, figure=fig, hspace=0.3, wspace=0.1)  # 6列：1个查询图像 + 5个文本结果
    
    example_idx = 0
    
    # 可视化Image-Query-Text结果（前5个）
    for i, img_idx in enumerate(selected_img_indices):
        if example_idx >= num_examples:
            break
        
        # 确保索引在有效范围内
        if img_idx >= len(i2t_results):
            print(f"Warning: Image index {img_idx} out of range for i2t_results (length: {len(i2t_results)})")
            continue
            
        result = i2t_results[img_idx]
        top_k_indices = result['top_k_indices']
        
        # 左侧：查询图像
        ax_img = fig.add_subplot(gs[example_idx, 0])
        # 确保图像索引在有效范围内
        if img_idx < len(images):
            img_filename = images[img_idx]
        else:
            print(f"Warning: Image index {img_idx} out of range for images list (length: {len(images)})")
            img_filename = f"image_{img_idx}.jpg"  # 使用占位符
        clean_img_filename = clean_image_filename(img_filename)
        
        img = load_image(image_path, clean_img_filename)
        ax_img.imshow(img)
        ax_img.axis('off')
        ax_img.set_title(f'Query Image {i+1}', fontsize=10, fontweight='bold')
        
        # 显示图像路径
        img_path_text = textwrap.fill(f'Path: {clean_img_filename}', width=30)
        ax_img.text(0.5, -0.15, img_path_text, transform=ax_img.transAxes, 
                   ha='center', fontsize=8, wrap=True)
        
        # 右侧：Top-5文本结果
        for k in range(5):
            ax_text = fig.add_subplot(gs[example_idx, k+1])
            ax_text.axis('off')
            
            cap_idx = top_k_indices[k]
            score = result['top_k_scores'][k]
            # 确保文本索引在有效范围内
            if cap_idx < len(captions):
                caption = captions[cap_idx]
            else:
                caption = f"[Caption index {cap_idx} out of range]"
            
            # 判断是否为ground truth（每5个文本对应一张图像）
            # 对应关系：文本索引 0-4 对应图像 0，文本索引 5-9 对应图像 1，以此类推
            img_idx_for_cap = cap_idx // 5
            is_gt = (img_idx_for_cap == img_idx) and (cap_idx < len(captions)) and (img_idx_for_cap < len(images))
            
            # 调试信息（仅在第一个样本的第一个结果时打印）
            if i == 0 and k == 0:
                print(f"\n[Debug] Image-Query-Text Example:")
                print(f"  Query image index: {img_idx}")
                print(f"  Retrieved caption index: {cap_idx}")
                print(f"  Caption's corresponding image index: {img_idx_for_cap} (calculated as {cap_idx} // 5)")
                print(f"  Is ground truth: {is_gt}")
                print(f"  Images list length: {len(images)}, Captions list length: {len(captions)}")
            
            # 显示文本
            wrapped_text = textwrap.fill(caption, width=50)
            ax_text.text(0.05, 0.95, f'Rank {k+1} (Score: {score:.3f})', 
                        transform=ax_text.transAxes, fontsize=9, 
                        fontweight='bold', verticalalignment='top',
                        color='green' if is_gt else 'black')
            ax_text.text(0.05, 0.75, wrapped_text, transform=ax_text.transAxes, 
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen' if is_gt else 'lightgray', alpha=0.5))
        
        example_idx += 1
    
    plt.suptitle('Image-to-Text Retrieval Results Visualization', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # 保存结果
    output_path = 'retrieval_results_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # 保存文本结果（只保存图像查询文本的结果）
    save_text_results(sim_matrix, captions, images, i2t_results, 
                     selected_img_indices, options)
    
    plt.close()

def save_text_results(sim_matrix, captions, images, i2t_results,
                     selected_img_indices, options):
    """保存文本格式的检索结果"""
    data_path = options['dataset']['data_path']
    image_path = options['dataset']['image_path']
    
    output_file = 'retrieval_results_text.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RETRIEVAL RESULTS - TEXT OUTPUT\n")
        f.write("=" * 80 + "\n\n")
        
        # Image-Query-Text结果
        f.write("IMAGE-QUERY-TEXT RESULTS\n")
        f.write("-" * 80 + "\n")
        for i, img_idx in enumerate(selected_img_indices):
            f.write(f"\nExample {i+1}: Image Query\n")
            img_filename = images[img_idx]
            clean_img_filename = clean_image_filename(img_filename)
            f.write(f"Query Image Path: {clean_img_filename}\n")
            f.write(f"Full Path: {image_path}{clean_img_filename}\n")
            f.write(f"\nTop-5 Text Results:\n")
            
            result = i2t_results[img_idx]
            for k in range(5):
                cap_idx = result['top_k_indices'][k]
                score = result['top_k_scores'][k]
                # 确保文本索引在有效范围内
                if cap_idx >= len(captions):
                    caption = f"[Caption index {cap_idx} out of range]"
                    img_idx_for_cap = -1
                else:
                    caption = captions[cap_idx]
                    img_idx_for_cap = cap_idx // 5
                is_gt = (img_idx_for_cap == img_idx)
                
                f.write(f"\n  Rank {k+1} (Score: {score:.4f}) {'[GT]' if is_gt else ''}\n")
                f.write(f"  Text: {caption}\n")
                if 0 <= img_idx_for_cap < len(images):
                    gt_img_name = images[img_idx_for_cap]
                    gt_img_name = clean_image_filename(gt_img_name)
                    f.write(f"  Corresponding Image: {gt_img_name}\n")
                else:
                    f.write(f"  Corresponding Image: N/A\n")
        

    
    print(f"Text results saved to: {output_file}")

def main():
    global options
    options = parser_options()
    
    # 加载模型和数据
    print("Loading model and data...")
    model, test_loader, captions, images, tokenizer = load_model_and_data(options)
    if model is None:
        return
    
    # 计算相似度矩阵
    print("Computing similarity matrix...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sim_matrix = compute_similarity_matrix(model, test_loader, tokenizer, device=device)
    
    print(f"Similarity matrix shape: {sim_matrix.shape}")
    
    # 获取top-k结果
    print("Getting top-k results...")
    i2t_results, t2i_results = get_top_k_results(sim_matrix, k=5)
    
    # 可视化结果（只显示图像查询文本）
    print("Visualizing results...")
    visualize_retrieval_results(sim_matrix, captions, images, i2t_results, t2i_results, 
                               options, num_examples=options['num_examples'])
    
    print("Done!")

if __name__ == '__main__':
    main()

