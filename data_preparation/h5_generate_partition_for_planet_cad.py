# -*- coding: utf-8 -*-
"""
generate training, validation and test partitions for planet-cadastral data

@author: liu_ch
"""

import os
import argparse
import h5py
import numpy as np

from random import seed, shuffle

def get_args():
    parser = argparse.ArgumentParser(description='Generate training, validation, test partitions for the Germany dataset.')
    # data settings
    parser.add_argument('--data_dir', type=str, help='path-to-germany-dataset')
    parser.add_argument('--data_name', type=str, default='PatchSz320_LabPerc0_nov.h5')
    # save settings
    parser.add_argument('--save_dir', type=str, help='path-to-save-dir')
    # settings for partition
    parser.add_argument('--num_val', dest='nval', type=int, default=50,
                        help='Size of validation set.')
    parser.add_argument('--num_test', dest='nts', type=int, default=200,
                        help='Size of test set.') 
    # version
    parser.add_argument('--version', dest='v', type=int, default=5,
                        help='Repeat version of partitions.') 
    # seed
    parser.add_argument('--seed', type=int, default=117,
                        help='Random seed.')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    seed(args.seed)
    
    # 1 - check whether concatenate file already exists
    con_cities = True
    con_fname = os.path.join(args.save_dir,'data_all_cities_gt.h5')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    else:
        if os.path.exists(con_fname):
            con_cities = False
    
    # 2 - get number of samples in total - n_sam
    if con_cities:
        # get number of samples from single files w.r.t. each city
        all_cities = os.listdir(args.data_dir)
        n_cities = len(all_cities)
        n_ps = np.zeros([n_cities]) 
        for i, city in enumerate(all_cities):
            fp = os.path.join(args.data_dir,city,args.data_name)
            with h5py.File(fp,'r') as fr:
                n_ps[i] = fr['dat'][()].shape[0]
        n_sam = np.sum(n_ps)
    else:
        with h5py.File(con_fname,'r') as fr:
            n_sam = fr['lab'][()].shape[0]
    
    # 3 - concatenate the data from all the cities
    if con_cities:
        with h5py.File(con_fname,'w') as cfw:
            for i, city in enumerate(all_cities):
                # save indexes
                si = int(n_ps[:i].sum())
                ei = int(n_ps[:i+1].sum())
                # open file of current city and resave
                fp = os.path.join(args.data_dir,city,args.data_name)
                with h5py.File(fp,'r') as fr:
                    dat0 = fr['dat'][()]
                    lab0 = fr['lab'][()]
                # create dataset
                if i==0:
                    _, d, h, w = dat0.shape
                    dat = cfw.create_dataset('dat',shape=(n_sam,d,h,w),dtype=dat0.dtype)
                    lab = cfw.create_dataset('lab',shape=(n_sam,h,w),dtype=lab0.dtype)
                # save
                dat[si:ei] = dat0
                lab[si:ei] = lab0
                del dat0, lab0
            # record number of patches of each city
            cfw['cities'] = all_cities
            cfw['n_patch'] = n_ps
        
    # 4 - generate and save partitions
    uinds = np.arange(n_sam)
    shuffle(uinds)
    fname_sv = os.path.join(args.save_dir,f'partitions_v{args.nval}_ts{args.nts}.h5')
    with h5py.File(fname_sv, 'a') as fw:
        for tps in ['val','ts', 'tr','seed']:
            k = f'{tps}_{args.v}'
            if k in fw.keys(): del fw[k]
        fw[f'val_{args.v}'] = uinds[:args.nval] 
        fw[f'ts_{args.v}'] = uinds[args.nval:args.nval+args.nts] 
        fw[f'tr_{args.v}'] = uinds[args.nval+args.nts:] 
        fw[f'seed_{args.v}'] = args.seed 
    
    
    
    
    