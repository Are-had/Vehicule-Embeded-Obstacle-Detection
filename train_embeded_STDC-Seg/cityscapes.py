#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os.path as osp
import os
from PIL import Image
import numpy as np
from transform import *

class CityScapes(Dataset):
    def __init__(self, rootpth, cropsize=(640, 480), mode='train', 
                 randomscale=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0), *args, **kwargs):
        super(CityScapes, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        self.ignore_lb = 255

        # --- DATA PARSING ---
        self.imgs = {}
        imgnames = []
        
        # 1. Parse Images (leftImg8bit)
        impth = osp.join(rootpth, 'leftImg8bit', mode)
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = osp.join(impth, fd)
            if not os.path.isdir(fdpth): continue # Safety check
            
            im_names = os.listdir(fdpth)
            names = [el.replace('_leftImg8bit.png', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))

        # 2. Parse Labels (gtFine)
        self.labels = {}
        gtnames = []
        gtpth = osp.join(rootpth, 'gtFine', mode)
        folders = os.listdir(gtpth)
        for fd in folders:
            fdpth = osp.join(gtpth, fd)
            if not os.path.isdir(fdpth): continue
            
            lbnames = os.listdir(fdpth)
            # We look for 'labelIds' specifically
            lbnames = [el for el in lbnames if 'labelIds' in el]
            names = [el.replace('_gtFine_labelIds.png', '') for el in lbnames]
            lbpths = [osp.join(fdpth, el) for el in lbnames]
            gtnames.extend(names)
            self.labels.update(dict(zip(names, lbpths)))

        self.imnames = imgnames
        self.len = len(self.imnames)

        # Sanity Check
        if self.len == 0:
            print(f"ERROR: No images found in {impth}. Check your folder structure.")

        # --- PRE-PROCESSING ---
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        self.trans_train = Compose([
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            HorizontalFlip(),
            RandomScale(randomscale),
            RandomCrop(cropsize)
        ])

    def __getitem__(self, idx):
        fn  = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        
        img = Image.open(impth).convert('RGB')
        label = Image.open(lbpth)
        
        if self.mode == 'train' or self.mode == 'trainval':
            im_lb = dict(im=img, lb=label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
            
        img = self.to_tensor(img)
        
        # Convert label to numpy 64-bit integer
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        
        # Apply the Binary Mapping
        label = self.convert_labels(label)
        
        return img, label

    def __len__(self):
        return self.len

    def convert_labels(self, label):
        """
        Maps Cityscapes IDs to Binary Road Segmentation.
        Original ID 7 (Road) -> 1
        Original ID 255 (Ignore) -> 255
        Everything else -> 0
        """
        # Create masks
        mask_road = (label == 7)
        mask_ignore = (label == 255)
        
        # Initialize canvas as Background (0)
        new_label = np.zeros_like(label)
        
        # Apply mappings
        new_label[mask_road] = 1
        new_label[mask_ignore] = 255
        
        return new_label

if __name__ == "__main__":
    # Test Block to verify mapping
    from tqdm import tqdm
    print("Testing Dataloader...")
    
    # Update path to where your organized folder is
    ds = CityScapes('./data/cityscapes_massyl', mode='val') 
    
    print(f"Found {len(ds)} images.")
    uni = []
    for i in tqdm(range(min(100, len(ds)))): # Check first 100 images
        img, lb = ds[i]
        unique_vals = np.unique(lb).tolist()
        uni.extend(unique_vals)
    
    final_classes = set(uni)
    print(f"\nUnique classes found in batches: {final_classes}")
    
    if final_classes.issubset({0, 1, 255}):
        print("SUCCESS: Dataloader is producing only [0, 1, 255].")
    else:
        print("WARNING: Dataloader produced unexpected classes!")