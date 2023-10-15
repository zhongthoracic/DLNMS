from numpy.random.mtrand import random
#import torchvision
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
#import torch
import pandas as pd
import os
from scipy.ndimage import affine_transform
import cupy as cp
from cupyx.scipy import ndimage as cu_ndimage
import cv2
import torch
import kornia.augmentation as K





class PetCT(Dataset):
    def __init__(self, path, mode, label, transform, train_mode, device, mean_pet=(0.89308566, 0.89308566, 0.89308566),std_pet=(0.9870795, 0.9870795, 0.9870795),mean_ct=(-410.96716, -410.96716, -410.96716),std_ct=(460.16315, 460.16315, 460.16315)):
        super().__init__()
        self.data_info = path
        self.mode = mode
        self.label = label
        if mode == 'train':
            self.transform = torch.nn.Sequential(K.RandomCrop((132,132)),K.RandomResizedCrop((112,112),(0.4,1)),K.RandomSharpness(),K.RandomMotionBlur(3, 35., 0.5, p=0.5))
        else:
            self.transform = torch.nn.Sequential(K.CenterCrop((132,132)),K.Resize((112,112)))
        self.pet_norm = torch.nn.Sequential(K.Normalize(mean_pet,std_pet))
        self.ct_norm = torch.nn.Sequential(K.Normalize(mean_ct,std_ct))
        self.rotate_transform = K.RandomRotation3D((-180,180))
        self.train_mode = train_mode
        self.device = device
    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        #idx = self.p_all[index]
        meta = self.get_train_data(index)
        if self.train_mode =="pet" or self.train_mode=="ct":
            img = meta[self.train_mode]
            img = self.transform(img)[0]
            if self.train_mode == 'pet':
                img = self.pet_norm(img)[0]
            else:
                img = self.ct_norm(img)[0]
            target = meta['target']
            return index, img, target
        else:
            img_pet = meta['pet']
            img_ct = meta['ct']
            img_pet = self.transform(img_pet)[0]
            img_ct = self.transform(img_ct)[0]
            img_pet = self.pet_norm(img_pet)[0]
            img_ct = self.ct_norm(img_ct)[0]
            target = meta['target']
            #img_ct = self.transform(img_ct)
            return index, torch.stack((img_pet,img_ct),0), target

    def get_train_data(self, idx):
        img_path = self.data_info['ID'][idx]+'.npy'
        var = 3
        aug = np.random.randint(-var,var)
        if self.mode != 'train':
            aug = 0
        
        meta = {}
        if "pet" in self.train_mode:
            pet = np.load(os.path.join('/home/gu1h/178data/pet-ct/pet-ct/pet', img_path)).astype(np.float32)
            pet = torch.as_tensor(pet).cuda(self.device)
            if self.mode == 'train':
                pet = self.rotate_transform(pet)[0][0]
            i_middle = pet.shape[0] // 2
            img0 = pet[i_middle+aug, :, :]
            img1 = pet[:, i_middle+aug, :]
            img2 = pet[:, :, i_middle+aug]
            
            f = torch.stack([img0, img1,img2], axis=0)
            #f = (f - torch.mean(f))/torch.std(f)
            #f = np.uint8(np.clip(f * 255, a_min=0, a_max=255))
            meta['pet'] = f
        if "ct" in self.train_mode:
            ct = np.load(os.path.join('/home/gu1h/178data/pet-ct/pet-ct/ct', img_path)).astype(np.float32)
            ct = torch.as_tensor(ct).cuda(self.device)
            if self.mode == 'train':
                ct = self.rotate_transform(ct)[0][0]
            i_middle = ct.shape[0] // 2
            img0 = ct[i_middle+aug, :, :]
            img1 = ct[:, i_middle+aug, :]
            img2 = ct[:, :, i_middle+aug]
            f = torch.stack([img0, img1,img2], axis=0)
            meta['ct'] = f

        meta['target'] = self.data_info[self.label][idx]

        return meta

