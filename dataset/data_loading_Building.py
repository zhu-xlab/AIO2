'''
Dataset for the Massachusetts dataset
'''
from os import listdir
from os.path import join, exists

import numpy as np
from torch.utils.data import Dataset,DataLoader

from torchvision import transforms
import random
import cv2

import matplotlib.pyplot as plt


EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def is_valid_file(filename):
    return filename.lower().endswith(EXTENSIONS)


class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            assert size>0, 'Please provide the array or size of selected indexes for subset.'
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


class BuildingDataset(Dataset):
    def __init__(self, data_path, noise_dir_name='ns_seg', split='train', 
                 aug=True, origin_ns_dir=None):
        # data paths
        self.data_path = join(data_path, split, 'data')
        self.gt_path = join(data_path, split, 'seg')
        
        if exists(noise_dir_name):
            self.ns_path = noise_dir_name
        else:
            self.ns_path = join(data_path, split, noise_dir_name)
        
        if origin_ns_dir is not None:
            self.ons_path = join(data_path, split, origin_ns_dir)
        else:
            self.ons_path = None
        self.split = split
        self.aug = aug
        
        # transforms
        self.transform = transforms.Compose([transforms.ToTensor()])
        
        # paths for each sample
        self.imgs = []
        self.gts = []
        self.ns_labels = []
        
        # get img and label paths
        fnames = sorted(listdir(self.data_path))
        rm_fn = []
        for fname in fnames: 
            if not is_valid_file(fname):
                rm_fn.append(fname)
        for rfn in rm_fn:
            fnames.remove(rfn)
        self.fnames = fnames
        
        # number of samples in total
        self.length = len(self.fnames)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        fname = self.fnames[index]
        ipath = join(self.data_path, fname)
        gpath = join(self.gt_path, fname)
        npath = join(self.ns_path, fname)
        if self.ons_path is not None:
            onpath = join(self.ons_path, fname)
        
        # load data
        img = cv2.imread(ipath)/255.0
        gt = cv2.imread(gpath,0).astype(float)
        ns = cv2.imread(npath,0).astype(float)
        if self.ons_path is not None:
            ons = cv2.imread(onpath,0).astype(float)

        # transforms
        if self.split == 'train' and self.aug:
            if random.random() > 0.5:
                fcode = random.choice([-1,0,1])
                img = cv2.flip(img,fcode)
                gt = cv2.flip(gt,fcode)
                ns = cv2.flip(ns,fcode)
                if self.ons_path is not None:
                    ons = cv2.flip(ons,fcode)
        
        
        if self.transform is not None:
            img = self.transform(img)                
            gt = self.transform(gt)
            ns = self.transform(ns)
            if self.ons_path is not None:
                ons = self.transform(ons)
                
        return_dict = {'img': img.float(), 'gt': gt.long(), 'ns': ns.long(), 'fname':fname}
        if self.ons_path is not None: return_dict['ons'] = ons.long()
        
        return return_dict
        
            
            
if __name__ == '__main__':

    data_path = 'C:\\liuchy\\Research\\Projects\\Datasets\\Building_QY\\mass'
    
    batchsize = 50
    num_workers = 0

    train_dataset = BuildingDataset(data_path, noise_dir_name='ns_seg', split='train')
    
    test_dataset = BuildingDataset(data_path, noise_dir_name='ns_seg', split='test')

    train_loader = DataLoader(train_dataset,
                              batch_size=batchsize,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=batchsize,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=False,
                             drop_last=True)
    
    print(len(train_dataset), len(test_dataset))
    
    
    for batch in train_loader:
        for i in [0,10,40]:
            img = batch['img'][i].numpy().squeeze().transpose((1,2,0))
            gt = batch['gt'][i].numpy().squeeze()
            ns = batch['ns'][i].numpy().squeeze()
            
            plt.figure(figsize=(12,4))
            plt.subplot(131)
            plt.imshow(img)
            plt.axis('off')
            plt.subplot(132)
            plt.imshow(gt, interpolation='none')
            plt.axis('off')
            plt.subplot(133)
            plt.imshow(ns, interpolation='none')
            plt.axis('off')
        
        
        
        
        
        
        