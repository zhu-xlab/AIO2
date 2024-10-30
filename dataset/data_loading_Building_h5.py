'''
Dataset for the Germany dataset
'''
from os.path import join, exists

import numpy as np
from torch.utils.data import Dataset,DataLoader

from torchvision import transforms
import random
import cv2, h5py

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
    def __init__(self, data_dir, data_name, noisy_label_name, 
                 partition_name, partition_version=1,
                 split='train', aug=True, pcorrect_label_dirname=None,
                 load_origin=False, only_load_gt=False):
        # data paths
        self.data_path = join(data_dir, data_name)
        partition_path = join(data_dir, partition_name)
        self.load_ns = (not only_load_gt)
        
        if self.load_ns:
            self.noisy_label_path = join(data_dir, noisy_label_name)
            # pixel-wise corrected label path
            if pcorrect_label_dirname is not None:
                self.pcn_dir = join(data_dir, pcorrect_label_dirname) # pixel corrected noisy labels
                assert exists(self.pcn_dir), "corrected label dir doesn't exist!"
                self.load_origin = load_origin
            else:
                self.pcn_dir = None
            
        # check split name
        if split == 'train':
            sp = 'tr'
        elif split == 'test':
            sp = 'ts'
        elif split == 'val':
            sp = 'val'
        else:
            raise NameError('Please provide correct split name!') 
        
        # load indexes
        with h5py.File(partition_path, 'r') as fp:
            self.inds = fp[f'{sp}_{partition_version}'][()]
        
        # other parameters
        self.split = split
        self.aug = aug
        # transforms
        self.transform = transforms.Compose([transforms.ToTensor()])
        # number of samples in total
        self.length = len(self.inds)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        ind = self.inds[index].astype(int)
        
        # load data
        with h5py.File(self.data_path,'r') as f:
            img = f['dat'][ind].transpose((1,2,0))
            gt = f['lab'][ind].astype(float)
        if self.load_ns:
            with h5py.File(self.noisy_label_path,'r') as fn:
                ns = fn['lab'][ind].astype(float)
            if self.pcn_dir is not None:
                pcn_path = join(self.pcn_dir,f'{ind}.png')
                cns = cv2.imread(pcn_path,0).astype(float)

        # transforms
        if self.split == 'train' and self.aug:
            if random.random() > 0.5:
                fcode = random.choice([-1,0,1])
                img = cv2.flip(img,fcode)
                gt = cv2.flip(gt,fcode)
                if self.load_ns:
                    ns = cv2.flip(ns,fcode)
                    if self.pcn_dir is not None:
                        cns = cv2.flip(cns,fcode)
        
        # to tensor
        if self.transform is not None:
            img = self.transform(img)                
            gt = self.transform(gt)
            if self.load_ns:
                ns = self.transform(ns)
                if self.pcn_dir is not None:
                    cns = self.transform(cns)
        
        # return dict
        if self.load_ns:
            if self.pcn_dir is None: 
                return_dict = {'img': img.float(), 'gt': gt.long(), 'ns': ns.long()}
            else:
                return_dict = {'img': img.float(), 'gt': gt.long(), 
                            'ns': cns.long(), 'fname':pcn_path}
                if self.load_origin: return_dict['ons'] = ns.long()
            return return_dict
        else:
            return {'img': img.float(), 'gt': gt.long()}


def copy_original_labels(partition_path, noisy_label_path, save_dir,
                         split='train', partition_version=1):
    # check split name
    if split == 'train':
        sp = 'tr'
    elif split == 'test':
        sp = 'ts'
    elif split == 'val':
        sp = 'val'
    else:
        raise NameError('Please provide correct split name!') 
    # read inds
    with h5py.File(partition_path, 'r') as fp:
        inds = fp[f'{sp}_{partition_version}'][()]
    # load data and resave data
    with h5py.File(noisy_label_path,'r') as fn:
        for ind in inds:
            ns = fn['lab'][ind].astype(np.uint8)
            fpath = join(save_dir,f'{ind}.png')
            cv2.imwrite(fpath, ns)
    return
            
            
if __name__ == '__main__':
    data_dir = r'C:\liuchy\Research\Projects\Datasets\Planet_Cadastral_Data\data\NRWV2_all'
    data_name = 'data_all_cities_gt.h5'
    noisy_label_name = 'data_all_cities_ns_rm_50.h5'
    partition_name = 'partitions_v50_ts200.h5'
    partition_version = 1
    pcorrect_label_dirname=None

    batchsize = 50
    num_workers = 0

    train_dataset = BuildingDataset(data_dir, data_name, noisy_label_name, 
                                    partition_name, partition_version, split='train', 
                                    aug=True, pcorrect_label_dirname=pcorrect_label_dirname)
    
    test_dataset = BuildingDataset(data_dir, data_name, noisy_label_name, 
                                   partition_name, partition_version, split='test', 
                                   aug=False, pcorrect_label_dirname=pcorrect_label_dirname)

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
        break
        
        
        
        
        
        
        