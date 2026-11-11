#encoding:utf-8

import os, random, copy
import numpy as np
import torch
import torch.nn as nn
import argparse
import yaml
import shutil
import tensorboard_logger as tb_logger
import logging
import click
from collections import OrderedDict
import utils
import data
import engine
from transformers import BertTokenizer
from vocab import deserialize_vocab

# 计算 FLOPs 和参数量（thop）
try:
    from thop import profile
    _HAS_THOP = True
except ImportError:
    _HAS_THOP = False

def parser_options():
    # Hyper Parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_opt', default='option/ERA_VCSR.yaml', type=str,
                        help='path to a yaml options file')
    parser.add_argument('--resume', default='checkpoint/era_aba_mv_fliter_new13_i102a1_de_agg_div_ra1/0/best.pth.tar', type=str,
                        help='path to a yaml options file')  
    #parser.add_argument('--resume', default='checkpoint/drone_aba_mv_fliter_new13_2_de_agg_div/0/VCSR_best.pth.tar', type=str,
    #                    help='path to a yaml options file')  
    opt = parser.parse_args()

    # load model options
    with open(opt.path_opt, 'r') as handle:
        options = yaml.load(handle,Loader=yaml.FullLoader)

    options['optim']['resume'] = opt.resume

    return options

def main(options):
    # choose model
    if options['model']['name'] == "VCSR":
        from layers import VCSR as models
    else:
        raise NotImplementedError

    
    tokenizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bert-base-uncased')
    tonkenizer = BertTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    # Create dataset, model, criterion and optimizer
    test_loader = data.get_test_loader(options)

    model = models.factory(options['model'],
                           cuda=True,
                           )

    print('Model has {} parameters'.format(utils.params_count(model)))

    # optionally resume from a checkpoint
    if os.path.isfile(options['optim']['resume']):
        print("=> loading checkpoint '{}'".format(options['optim']['resume']))
        checkpoint = torch.load(options['optim']['resume'],map_location=torch.device("cpu"))
        state_dict = checkpoint['model']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k  # 去掉 'module.' 前缀
            new_state_dict[name] = v
        start_epoch = checkpoint['epoch']
        best_rsum = checkpoint['best_rsum']
        #model.load_state_dict(checkpoint['model'])
        model.load_state_dict(new_state_dict)
    else:
        print("=> no checkpoint found at '{}'".format(options['optim']['resume']))

    # ================= 使用 thop 计算 FLOPs 和参数量 =================
    if _HAS_THOP:
        model.eval()
        # 构造一条假样本：1 张 3×224×224 图像 + 一句长度 40 的文本
        dummy_image = torch.randn(1, 3, 224, 224)
        dummy_ids = torch.ones(1, 40, dtype=torch.long)
        dummy_mask = torch.ones(1, 40, dtype=torch.long)

        if torch.cuda.is_available():
            dummy_image = dummy_image.cuda()
            dummy_ids = dummy_ids.cuda()
            dummy_mask = dummy_mask.cuda()

        try:
            macs, params = profile(model, inputs=(dummy_image, dummy_ids, dummy_mask), verbose=False)
            flops = 2 * macs  # 1 MAC = 2 FLOPs（乘加）
            print(f"FLOPs: {flops/1e9:.2f} GFLOPs")
            print(f"Parameters: {params/1e6:.2f} M")
        except Exception as e:
            print("FLOPs 计算失败（thop）:", e)
    else:
        print("未安装 thop，无法计算 FLOPs，请先执行：pip install thop")

    # evaluate on test set
    sims = engine.validate_test(test_loader,tonkenizer ,model)

    return sims

def update_options_savepath(options, k):
    updated_options = copy.deepcopy(options)

    updated_options['optim']['resume'] = options['logs']['ckpt_save_path'] + options['k_fold']['experiment_name'] + "/" \
                                         + str(k) + "/" + options['model']['name'] + '_best.pth.tar'

    return updated_options


def reduce_matrix(one_sims):
    num_rows, num_cols = one_sims.shape
    assert num_cols % 5 == 0, "列数必须是5的倍数"
    
    reduced_rows = []

    for i in range(num_rows):
        row = one_sims[i]
        
        # 将整行按每5列划分为一个 block
        blocks = [row[j:j+5] for j in range(0, num_cols, 5)]

        # 正样本：第 i 个 block 的第一个元素
        pos_sample = blocks[i][0]

        # 负样本：其他 block 的第一个元素
        neg_samples = [block[0] for idx, block in enumerate(blocks) if idx != i]

        # 合并为一行（正样本 + 所有负样本）
        new_row = np.array([pos_sample] + neg_samples)
        reduced_rows.append(new_row)

    reduced_matrix = np.vstack(reduced_rows)
    return reduced_matrix

if __name__ == '__main__':
    options = parser_options()

    # run experiment
    one_sims = main(options)
    #print(one_sims.shape)
    #rs = np.argsort(one_sims.transpose(),axis=1)+1#.transpose()
    '''
    #img-text
    c=1
    result=""
    for i in rs[:,-5:]:
        result+=str(c)+"-"+str(c+4)+" "+str(i[::-1])+"\n"
        c+=5
    '''
    '''
    # text-img #need to t()
    c=1
    im=1
    result=""
    for i in rs[:,-5:]:
        result+=str(c)+"-"+str(im)+" "+str(i[::-1])+"\n"
        c+=1
        if c%6==0:
           im+=1
    
    f=open("ERA_result_text_img.txt","a+")
    f.write(result)
    '''
    # ave
    last_sims = one_sims
 

    

    # get indicators
    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t2(last_sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i2(last_sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1t, r5t, r10t, medrt, meanrt))
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i)/6.0

    all_score = "r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore
    )

    #print(all_score)
    # 既打印又保存
    print(all_score)
    with open('all_score.txt', 'w') as f:
        f.write(str(all_score))


