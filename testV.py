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
import time  # 新增计时

def parser_options():
    # Hyper Parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_opt', default='option/ERA_VCSR.yaml', type=str,
                        help='path to a yaml options file')
    parser.add_argument('--resume', default='checkpoint/era_aba_mv_fliter_new13_i102a1_de_agg_div_ra1/0/best.pth.tar', type=str,
                        help='path to a yaml options file')
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

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
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
        model.load_state_dict(new_state_dict)
    else:
        print("=> no checkpoint found at '{}'".format(options['optim']['resume']))

    # =============================
    # Evaluate on test set + measure throughput
    # =============================
    model.eval()
    all_sims = []
    total_samples = 0
    start_time = time.time()
    with torch.no_grad():
        for batch in test_loader:
            # 根据你的数据结构调整
            images, texts = batch['image'].cuda(), batch['text'].cuda()
            torch.cuda.synchronize()
            sims = model(images, texts)  # 如果你的 engine.validate_test 里有复杂逻辑，可直接替换成下面方式
            torch.cuda.synchronize()
            all_sims.append(sims.cpu().numpy())
            total_samples += images.size(0)
    end_time = time.time()

    one_sims = np.vstack(all_sims)
    throughput = total_samples / (end_time - start_time)

    return one_sims, throughput

# 其余函数保持不变
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
        blocks = [row[j:j+5] for j in range(0, num_cols, 5)]
        pos_sample = blocks[i][0]
        neg_samples = [block[0] for idx, block in enumerate(blocks) if idx != i]
        new_row = np.array([pos_sample] + neg_samples)
        reduced_rows.append(new_row)
    reduced_matrix = np.vstack(reduced_rows)
    return reduced_matrix

if __name__ == '__main__':
    options = parser_options()

    # run experiment
    one_sims, throughput = main(options)

    # get indicators
    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t2(one_sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri, meanri))
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i2(one_sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1t, r5t, r10t, medrt, meanrt))
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i)/6.0

    all_score = "r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n throughput:{:.2f} samples/sec\n------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore, throughput
    )

    print(all_score)
    with open('all_score.txt', 'w') as f:
        f.write(str(all_score))
