#encoding:utf-8

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
import numpy as np
import yaml
import argparse
import utils
from vocab import deserialize_vocab
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler
class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, data_split,  opt):
        
        self.loc = opt['dataset']['data_path']
        self.img_path = opt['dataset']['image_path']

        # Captions
        self.captions = []
        self.maxlength = 0

        if data_split != 'test':
            with open(self.loc+'%s_caps_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            with open(self.loc + '%s_filename_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.images.append(line.strip())
        else:
            with open(self.loc + '%s_caps.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            with open(self.loc + '%s_filename.txt' % data_split, 'rb') as f:
                for line in f:
                    self.images.append(line.strip())
        
        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if len(self.images) != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        if data_split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomRotation((0, 90)),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index//self.im_div
        caption = self.captions[index]


        image = Image.open(self.img_path  +str(self.images[img_id])[2:-1]).convert('RGB')
        image = self.transform(image)  # torch.Size([3, 256, 256])

        return image, caption, index, img_id

    def __len__(self):
        return self.length


def collate_fn(data):

    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions,  ids, img_ids = zip(*data)
   
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = []
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets.append(cap)

    lengths = [l if l !=0 else 1 for l in lengths]

    return images, targets, lengths, ids


def get_precomp_loader(data_split,batch_size=100,
                       num_workers=0, drop_last=False, opt={}):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if data_split == 'train':
        dset = PrecompDataset(data_split, opt)

        data_sampler = DistributedSampler(dset, num_replicas=opt['distributed']['world_size'], rank=opt['distributed']['rank'])

        data_loader = DataLoader(dataset=dset,
                                batch_size=batch_size,
                                sampler=data_sampler,
                                pin_memory=True,
                                collate_fn=collate_fn,
                                num_workers=num_workers,
                                drop_last=drop_last)
    
    if data_split == 'val':
        
        dset = PrecompDataset(data_split, opt)
        data_loader = DataLoader(dataset=dset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=True,
                                 collate_fn=collate_fn,
                                 num_workers=num_workers,
                                 drop_last=drop_last)
    if data_split == 'test':
       
        dset = PrecompDataset(data_split,  opt)
        data_loader = DataLoader(dataset=dset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=True,
                                 collate_fn=collate_fn,
                                 num_workers=num_workers,
                                 drop_last=drop_last)
    return data_loader

def get_loaders( opt):
    train_loader = get_precomp_loader( 'train',
                                      opt['dataset']['batch_size'], opt['dataset']['workers'], False, opt=opt)
    val_loader = get_precomp_loader( 'val', 
                                    opt['dataset']['batch_size_val'], opt['dataset']['workers'], False, opt=opt)
    return train_loader, val_loader


def get_test_loader( opt):
    test_loader = get_precomp_loader( 'test', 
                                      opt['dataset']['batch_size_val'], opt['dataset']['workers'], False, opt=opt)
    return test_loader
