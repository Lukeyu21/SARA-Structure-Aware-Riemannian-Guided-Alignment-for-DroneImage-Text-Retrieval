#encoding:utf-8

import os,random,copy
import torch
import torch.nn as nn
import argparse
import yaml
import shutil
import tensorboard_logger as tb_logger
import logging
import click
from tqdm import tqdm
import utils
import data
import engine

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import geoopt

from vocab import deserialize_vocab
from transformers import BertTokenizer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def setup(rank, world_size):
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "23456"
        print(f"Rank {rank}: Initializing process group...")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        print(f"Rank {rank}: Process group initialized.")
    except Exception as e:
        print(f"Rank {rank}: Failed to initialize process group. Error: {e}")
        raise

def cleanup():
    dist.destroy_process_group()

def parser_options():
    # Hyper Parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_opt', default='option/ERA_VCSR.yaml', type=str,
                        help='path to a yaml options file')
    # parser.add_argument('--text_sim_path', default='data/ucm_precomp/train_caps.npy', type=str,help='path to t2t sim matrix')
    opt = parser.parse_args()

    # load model options
    with open(opt.path_opt, 'r') as handle:
        options = yaml.load(handle,Loader=yaml.FullLoader)

    return options

def main(rank, world_size, options):

    setup(rank, world_size)
    # choose model
    if options['model']['name'] == "VCSR":
        from layers import VCSR as models
    else:
        raise NotImplementedError

    # make ckpt save dir
    if not os.path.exists(options['logs']['ckpt_save_path']):
        os.makedirs(options['logs']['ckpt_save_path'])


    
    # Create dataset, model, criterion and optimizer
    train_loader, val_loader = data.get_loaders( options)
    tokenizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bert-base-uncased')
    tonkenizer = BertTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    model = models.factory(options['model'],
                           cuda=True, 
                           rank=rank,)
    model = DDP(model, device_ids=[rank],find_unused_parameters=True)

    #spdnet_params, other_params = get_spdnet_params(model)


    # optimizer_manifold = geoopt.optim.RiemannianAdam(spdnet_params,lr=0.0001)
    # optimizer_euclidean = torch.optim.Adam(other_params,lr=options['optim']['lr'])
    optimizer_manifold = geoopt.optim.RiemannianAdam(model.parameters(), lr=options['optim']['lr'])

    print('Model has {} parameters'.format(utils.params_count(model)))

    # optionally resume from a checkpoint
    if options['optim']['resume']:
        if os.path.isfile(options['optim']['resume']):
            print("=> loading checkpoint '{}'".format(options['optim']['resume']))
            checkpoint = torch.load(options['optim']['resume'])
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
         
            # Eiters is used to show logs as the continuation of another
            # training
            model.moudle.Eiters = checkpoint['Eiters']
   
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(options['optim']['resume'], start_epoch, best_rsum))
            rsum, all_scores =  engine.validate(val_loader, model)
            print(all_scores)
        else:
            print("=> no checkpoint found at '{}'".format(options['optim']['resume']))
    else:
        start_epoch = 0

    # Train the Model
    best_rsum = 0
    best_score = ""

    for epoch in range(start_epoch, options['optim']['epochs']):
       
        train_loader.sampler.set_epoch(epoch)
        progress_bar = tqdm(train_loader, desc=f"Rank {rank} | Epoch {epoch+1}/{options['optim']['epochs']}",position=rank,disable=(rank != 0))

        # myComment
        # adjust learning rate
        utils.adjust_learning_rate(options, optimizer_manifold, epoch)

        # train for one epoch
        engine.train(train_loader, model,tonkenizer, optimizer_manifold,epoch, progress_bar,opt=options)

        # evaluate on validation set
        if  epoch % options['logs']['eval_step'] == 0:
            
            if rank == 0:
                rsum, all_scores = engine.validate(val_loader, model,tonkenizer,shard_size=options['dataset']['batch_size_val'],rank=rank)
                is_best = rsum > best_rsum
                if is_best:
                    best_score = all_scores
                best_rsum = max(rsum, best_rsum)
                
                # save ckpt
                utils.save_checkpoint(
                    {
                    'epoch': epoch + 1,
                    'arch': 'baseline',
                    'model': model.state_dict(),
                    'best_rsum': best_rsum,
                    'options': options,
                    'Eiters': model.module.Eiters,
                },
                    is_best,
                    filename='ckpt_{}_{}_{:.2f}.pth.tar'.format(options['model']['name'] ,epoch, best_rsum),
                    prefix=options['logs']['ckpt_save_path'],
                    model_name=options['model']['name']
                )



                utils.log_to_txt(
                    contexts= "Epoch:{} ".format(epoch+1) + all_scores,
                    filename=options['logs']['ckpt_save_path']+ options['model']['name'] + "_" + options['dataset']['datatype'] +".txt"
                )
                utils.log_to_txt(
                    contexts= "Best:   " + best_score,
                    filename=options['logs']['ckpt_save_path']+ options['model']['name'] + "_" + options['dataset']['datatype'] +".txt"
                )
            else:
                rsum1, all_scores1 = engine.validate(val_loader, model,tonkenizer,shard_size=options['dataset']['batch_size_val'],rank=rank)
                

            
        print(f"Rank {rank}: Reached before barrier")
        dist.barrier()
        print(f"Rank {rank}: Passed barrier")
    cleanup()

def generate_random_samples(options):
    # load all anns
    caps = utils.load_from_txt(options['dataset']['data_path']+'train_caps.txt')
    fnames = utils.load_from_txt(options['dataset']['data_path']+'train_filename.txt')

    # merge
    assert len(caps) // 5 == len(fnames)
    all_infos = []
    for img_id in range(len(fnames)):
        cap_id = [img_id * 5 ,(img_id+1) * 5]
        all_infos.append([caps[cap_id[0]:cap_id[1]], fnames[img_id]])

    # shuffle
    random.shuffle(all_infos)

    # split_trainval
    percent = 0.8
    train_infos = all_infos[:int(len(all_infos)*percent)]
    val_infos = all_infos[int(len(all_infos)*percent):]

    # save to txt
    train_caps = []
    train_fnames = []
    for item in train_infos:
        for cap in item[0]:
            train_caps.append(cap)
        train_fnames.append(item[1])
    utils.log_to_txt(train_caps, options['dataset']['data_path']+'train_caps_verify.txt',mode='w')
    utils.log_to_txt(train_fnames, options['dataset']['data_path']+'train_filename_verify.txt',mode='w')

    val_caps = []
    val_fnames = []
    for item in val_infos:
        for cap in item[0]:
            val_caps.append(cap)
        val_fnames.append(item[1])
    utils.log_to_txt(val_caps, options['dataset']['data_path']+'val_caps_verify.txt',mode='w')
    utils.log_to_txt(val_fnames, options['dataset']['data_path']+'val_filename_verify.txt',mode='w')

    print("Generate random samples to {} complete.".format(options['dataset']['data_path']))

def update_options_savepath(options, k):
    updated_options = copy.deepcopy(options)

    updated_options['k_fold']['current_num'] = k
    updated_options['logs']['ckpt_save_path'] = options['logs']['ckpt_save_path'] + \
                                                options['k_fold']['experiment_name'] + "/" + str(k) + "/"
    return updated_options


def get_spdnet_params(model):
    spdnet_params = []
    other_params = []
    for name, param in model.named_parameters():
        print(f"Name: {name}, Param: {param.size()}")
        if "spdconv" in name:  # SPDNet 的参数
            spdnet_params.append(param)
        else:  # 其他参数
            other_params.append(param)
    print(f"SPDNet params: {len(spdnet_params)}")
    print(f"Other params: {len(other_params)}")
    return spdnet_params, other_params

if __name__ == '__main__':
    options = parser_options()
    # make logger
    tb_logger.configure(options['logs']['logger_name'], flush_secs=5)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)



    # generate random train and val samples
    generate_random_samples(options)

    # update save path
    update_options = update_options_savepath(options, 0)
    world_size = torch.cuda.device_count()
    # run experiment
    mp.spawn(main,
             args=(world_size, update_options,),
             nprocs=world_size,
             join=True)
