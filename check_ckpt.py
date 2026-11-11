import torch
import sys
import os

ckpt_path = r"/opt/data/private/VCSRv1/VCSR/checkpoint/era_aba_mv_fliter_new13_i102a1_de_agg_div_ra1/0/VCSR_best.pth.tar"

if not os.path.exists(ckpt_path):
    print(f'权重文件不存在: {ckpt_path}')
    sys.exit(1)

try:
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    print('权重文件加载成功！')
except Exception as e:
    print('权重文件加载失败:', e)
    sys.exit(1)

print('\n== checkpoint keys ==')
print(list(checkpoint.keys()))

# 检查 epoch 和 best_rsum
print('\n== epoch & best_rsum ==')
epoch = checkpoint.get('epoch', None)
best_rsum = checkpoint.get('best_rsum', None)
print('epoch:', epoch)
print('best_rsum:', best_rsum)

# 检查 model 权重
model_dict = checkpoint.get('model', None)
if model_dict is None:
    print('未找到 model 权重参数！')
    sys.exit(1)

print(f'\n== 权重参数数量: {len(model_dict)} ==')
param_zero_count = 0
has_module_prefix = False
for i, (k, v) in enumerate(model_dict.items()):
    if k.startswith('module.'):
        has_module_prefix = True
    v_data = v.float()
    mean = v_data.mean().item()
    std = v_data.std().item()
    zero_count = (v_data == 0).sum().item()
    total_count = v_data.numel()
    if zero_count == total_count:
        param_zero_count += 1
    print(f'{i+1:3d}. {k:40s} shape={tuple(v.shape)} mean={mean:.6f} std={std:.6f} zero_ratio={zero_count/total_count:.4f}')
    if i >= 9:
        print('...')
        break

print(f'\n全零参数个数: {param_zero_count}')
print(f'是否为DataParallel保存（module.前缀）: {has_module_prefix}')

print('\n检查完毕！') 