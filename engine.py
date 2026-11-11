#encoding:utf-8

import time
import torch
import numpy as np
import sys
from torch.autograd import Variable
import utils
import tensorboard_logger as tb_logger
import logging
from torch.nn.utils.clip_grad import clip_grad_norm
import pandas as pd
import torch.nn.functional as F

def main_contrastive_loss(dual_sim, temperature=0.07):
    logits = dual_sim / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2

def train(train_loader, model,tonkenizer, optimizer_manifold, epoch,progress_bar,opt={}):

    # extract value
    grad_clip = opt['optim']['grad_clip']
    max_violation = opt['optim']['max_violation']
    margin = opt['optim']['margin']
    loss_name = opt['model']['name'] + "_" + opt['dataset']['datatype']
    print_freq = opt['logs']['print_freq']
    lambda_struct = opt['optim'].get('lambda_struct', 0.1)

    # switch to train mode
    model.train()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    train_logger = utils.LogCollector()
    
    lossf=utils.NceLoss()

    end = time.time()
    params = list(model.parameters())
    local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    device = torch.device(f"cuda:{local_rank}")

    for i, train_data in enumerate(progress_bar):
        input_visual, captions, _, _= train_data
        
        margin = float(margin)
        # measure data loading time
        data_time.update(time.time() - end)

        if isinstance(captions, list) and isinstance(captions[0], bytes):
            captions = [t.decode("utf-8") for t in captions]
        input_text = tonkenizer(captions, padding='longest', max_length=40, return_tensors="pt")


        if torch.cuda.is_available():
            input_visual = input_visual.to(device)
            input_text = input_text.to(device)

        # 获取结构特征和LESA损失
        scores, spd_feat_img, spd_feat_txt, structure_loss, loss_dict = model(input_visual, input_text.input_ids, input_text.attention_mask, return_structure_feats=True)
        torch.cuda.synchronize()
     
        # 主对比损失（融合后的dual_sim）
        loss_main = main_contrastive_loss(scores, temperature=0.07)
        # Log-Euclidean结构对齐损失
        loss = loss_main + lambda_struct * structure_loss


        if grad_clip > 0:
            clip_grad_norm(params, grad_clip)

        train_logger.update('L', loss.cpu().data.numpy())
        train_logger.update('L_main', loss_main.cpu().data.numpy())
        train_logger.update('L_struct', structure_loss.cpu().data.numpy())
        train_logger.update('L_mda', loss_dict['mda_loss'])
        train_logger.update('L_cda', loss_dict['cda_loss'])
        
        progress_bar.set_postfix({
            'loss': f"{loss.cpu().data.numpy():.4f}",
            'main': f"{loss_main.cpu().data.numpy():.4f}",
            'struct': f"{structure_loss.cpu().data.numpy():.4f}",
            'mda': f"{loss_dict['mda_loss']:.4f}",
            'cda': f"{loss_dict['cda_loss']:.4f}"
        })
        backward_start_time = time.time()
        optimizer_manifold.zero_grad()
        
        
        loss.backward()
        torch.cuda.synchronize()
        backward_end_time = time.time()
        #print(f"Backward pass time: {backward_end_time - backward_start_time:.4f} seconds")

       
        optimizer_manifold.step()
        
        torch.cuda.synchronize()
    

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if local_rank == 0 and i % print_freq == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                'Loss {loss:.4f}\t'
                'Main {main:.4f}\t'
                'Struct {struct:.4f}\t'
                'MDA {mda:.4f}\t'
                'CDA {cda:.4f}\t'
                '{elog}\t'
                .format(epoch, i, len(train_loader),
                        batch_time=batch_time,
                        loss=loss.cpu().data.numpy(),
                        main=loss_main.cpu().data.numpy(),
                        struct=structure_loss.cpu().data.numpy(),
                        mda=loss_dict['mda_loss'],
                        cda=loss_dict['cda_loss'],
                        elog=str(train_logger)))

            utils.log_to_txt(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'
                    .format(epoch, i, len(train_loader),
                            batch_time=batch_time,
                            elog=str(train_logger)),
                opt['logs']['ckpt_save_path']+ opt['model']['name'] + "_" + opt['dataset']['datatype'] +".txt"
            )

def validate(val_loader, model,tonkenizer,shard_size,rank=0):
    
    #验证模式启动
    model.eval()
    val_logger = utils.LogCollector()
    

    start = time.time()
    
    input_visual = np.zeros((len(val_loader.dataset), 3, 224, 224))
    input_text_ids = [None] * (len(val_loader.dataset))
    input_text_mask = [None] * (len(val_loader.dataset))


  
    for i, val_data in enumerate(val_loader):
  
        images, captions, _, ids = val_data
        if isinstance(captions, list) and isinstance(captions[0], bytes):
            captions = [t.decode("utf-8") for t in captions]
        captions = tonkenizer(captions, padding='longest', max_length=40, return_tensors="pt")
     
        
        for (id, img,ids,mask) in zip(ids, (images.numpy().copy()), (captions['input_ids'].numpy().copy()),(captions['attention_mask'].numpy().copy())):
            input_visual[id] = img
            input_text_ids[id] = ids
            input_text_mask[id] = mask
         
   
            
    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    # input_visual = np.zeros((10000, 3, 224, 224))
    # input_text_ids = [None] * (10000)
    # input_text_mask = [None] * (10000)
    # print("input_visual:",input_visual.shape)
    # for i, val_data in enumerate(val_loader):
    #     if i >= 50:  # 只取前2000个batch
    #         break
    #     images, captions, _, ids = val_data
    #     print("ids:", len(ids))
    #     if isinstance(captions, list) and isinstance(captions[0], bytes):
    #         captions = [t.decode("utf-8") for t in captions]
    #     captions = tonkenizer(captions, padding='longest', max_length=40, return_tensors="pt")
    #     print(i)
        
    #     for (id, img,ids,mask) in zip(ids, (images.numpy().copy()), (captions['input_ids'].numpy().copy()),(captions['attention_mask'].numpy().copy())):
    #         input_visual[id] = img
    #         input_text_ids[id] = ids
    #         input_text_mask[id] = mask
   
            
    # print("input_visual:",input_visual.shape)
    # input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    d = utils.shard_dis(input_visual, input_text_ids,input_text_mask, model,shard_size=shard_size ,device=rank)




    end = time.time()
    print("calculate similarity time:", end - start)

    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t2(d)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i2(d)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1t, r5t, r10t, medrt, meanrt))
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i)/6.0

    all_score = "r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore
    )
    print(all_score)
  


    return currscore, all_score

    


def validate_test(val_loader,tonkenizer, model):
    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger
    
    # 记录检索开始时间（包括特征提取与相似度计算）
    start_time = time.time()

    input_visual = np.zeros((len(val_loader.dataset), 3, 224, 224))
    input_text_ids = [None] * len(val_loader.dataset)
    input_text_mask = [None] * len(val_loader.dataset)
    

    for i, val_data in enumerate(val_loader):

        images, captions, lengths, ids = val_data

        if isinstance(captions, list) and isinstance(captions[0], bytes):
            captions = [t.decode("utf-8") for t in captions]
        captions = tonkenizer(captions, padding='longest', max_length=40, return_tensors="pt")

        for (id, img,ids,mask) in zip(ids, (images.numpy().copy()), (captions['input_ids'].numpy().copy()),(captions['attention_mask'].numpy().copy())):
            input_visual[id] = img
            input_text_ids[id] = ids
            input_text_mask[id] = mask
            
            
    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    # 计算相似度矩阵
    d = utils.shard_dis(input_visual, input_text_ids,input_text_mask, model)

    # 记录结束时间并统计每张图片的平均检索时间
    end_time = time.time()
    num_images = input_visual.shape[0]
    total_time = end_time - start_time
    time_per_image = total_time / float(num_images) if num_images > 0 else 0.0

    print("Total retrieval time: {:.6f}s, num_images: {}, time per image: {:.6f}s"
          .format(total_time, num_images, time_per_image))

    return d


# 结构特征对比损失函数
import torch.nn.functional as F

def structure_contrastive_loss(spd_feat_img, spd_feat_txt, temperature=0.07):
    spd_feat_img = F.normalize(spd_feat_img, dim=1)
    spd_feat_txt = F.normalize(spd_feat_txt, dim=1)
    logits = torch.matmul(spd_feat_img, spd_feat_txt.t()) / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2
