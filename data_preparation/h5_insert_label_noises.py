# -*- coding: utf-8 -*-
"""
Noisy label injection for building extraction task 
    - For all the samples stored in an h5 file (Germany dataset)

@author: liu_ch
"""

import os
import argparse
import cv2, h5py
import numpy as np

from random import seed, shuffle, randint, choice, random
    

def get_args():
    parser = argparse.ArgumentParser(description='Insert label noises into building extractiond data for the Germany dataset.')
    # data settings
    parser.add_argument('--data_dir', type=str, help='Path-to-germany-dataset')
    parser.add_argument('--data_name', type=str, default='PatchSz320_LabPerc0_nov.h5')
    # saving file name
    parser.add_argument('--save_name', type=str, default='PatchSz320_LabPerc0_nov_rm_30_5.h5')  
    parser.add_argument('--save_dir', type=str, default='noisy_labels',
                        help='Directory for saving concatenated data.')
    # threshold for binary label generation
    parser.add_argument('--lab_from_covarea', action='store_true',
                        help='Generate binary labels from covarea or not (load labels from files).')
    parser.add_argument('--label_key', dest='lkey', type=str, default='lab',
                        help='Name of the key for label variables (covarea when lab_from_covarea=True; lab otherwise).')
    parser.add_argument('--covarea_threshold', dest='cth', type=int, default=50,
                        help='Threshold set to generate binary labels from covarea (only required when lab_from_covarea=true).')
    # noise insertion settings
    parser.add_argument('--ns_types', nargs="+", default=['remove'], #['shift','erosion','dilation','rotation','remove'],
                        help='Candidate noise types.') 
    parser.add_argument('--ns_rates',  nargs="+", type=float, default=[0,0.1,0.2,0.3,0.4,0.5,0.6],#[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], #, 0.8],
                        help='Candidate noise rates.')
    # generate index layer on the fly or directly load generated ones
    parser.add_argument('--generate_index', dest='gind', action='store_false')  
    # if only concatenate all the noisy labels from existing files
    parser.add_argument('--concate_only', dest='ccto', action='store_true') 
    # seed
    parser.add_argument('--seed',  type=int, default=7,
                        help='Random seed.')
    return parser.parse_args()


def shift_noise(mask):
    contours, _ = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    e = max([cnt[:,:,i].max()-cnt[:,:,i].min()+1 for i in [0,1]]+[12])
    shift_bound = int(e/2)
    
    # shift
    rows,cols = mask.shape
    M = np.float32([[1,0,randint(-shift_bound,shift_bound)],[0,1,randint(-shift_bound,shift_bound)]])
    dst = cv2.warpAffine(mask, M, (cols,rows))
    return dst


def erosion_noise(mask):
    kss = np.arange(5,15) # 3-7
    # kernel size
    while 1:
        ks1 = choice(kss)
        ks2 = choice(kss)
        if ks1+ks2>2:
            break
    # erosion
    kernel = np.ones((ks1,ks2), np.uint8)
    dst = cv2.erode(mask, kernel)
    return dst


def dilate_noise(mask):
    kss = np.arange(5,16) # 3-8
    # kernel size
    while 1:
        ks1 = choice(kss)
        ks2 = choice(kss)
        if ks1+ks2>2:
            break
    # dilation
    kernel = np.ones((ks1,ks2), np.uint8)
    dst = cv2.dilate(mask, kernel)
    return dst


def rotate_noise(mask):
    # img shape
    rows,cols = mask.shape
    # find origin of object
    contours, _ = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    origin = [(cnt[:,:,i].max()+cnt[:,:,i].min())/2 for i in [0,1]]
    # rotate
    M = cv2.getRotationMatrix2D(origin,randint(10,350),1)
    dst = cv2.warpAffine(mask, M, (cols,rows))  # rotating the image by random degree with respect to building center without any scaling
    return dst


def remove_noise(mask):
    return np.zeros_like(mask, dtype=np.uint8)


def add_noise(mask):
    if random()>0.5:
        mask = rotate_noise(mask)
    if random()>0.5:
        mask = shift_noise(mask)
    if random()>0.5:
        mask = dilate_noise(mask)
    return mask


def insert_single_item_noise(mask, nst):
    if nst=='shift':
        dst = shift_noise(mask)
    elif nst=='erosion':
        dst = erosion_noise(mask)
    elif nst=='dilation':
        dst = dilate_noise(mask)
    elif nst=='rotation':
        dst = rotate_noise(mask)
    elif nst=='remove':
        dst = remove_noise(mask)
    else:
        raise ValueError('Given noise type is invalid!')
    return dst


def count_and_index_building(gt):
    # find contours
    contours, hierarchy = cv2.findContours(gt,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    # index buildings
    mask = np.zeros(list(gt.shape)+[3],dtype=np.int_)   # int32
    for ci, cnt in enumerate(contours):
        cv2.drawContours(mask, [cnt], 0, (ci+1,ci+1,ci+1), thickness=cv2.FILLED)
    # convert from RGB to gray
    mask = mask[:,:,0]  # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask*gt
    return mask
            


if __name__ == '__main__':
    # 0> preparations
    args = get_args()
    seed(args.seed)
    nsrs = np.array(args.ns_rates)
    
    # directories
    # get all dir names for all cities
    all_cities = os.listdir(args.data_dir)
    
    # 1> insert label noises
    if not args.ccto:
        # 1 - check whether require adding noises for no-object patches
        if 'add' in args.ns_types:
            add_bd = True
            add_obj = None
            args.ns_types.remove('add')
        else:
            add_bd = False
        
        # candidates of numbers/types of label noise insertion into each patch
        nsts = np.arange(len(args.ns_types))+1
        
        # 2 - starting inserting label noises one by one in each city
        for ci, ct in enumerate(all_cities):
            print(f'{ci+1} - Process {ct}')
            # get file paths
            fpr = os.path.join(args.data_dir, ct, args.data_name)
            fpw = os.path.join(args.data_dir, ct, args.save_name)
            
            fr = h5py.File(fpr,'r')
            fw = h5py.File(fpw,'w')
            
            # get number of samples for current city
            n_sam, n_line, n_col = fr[args.lkey][()].shape
            
            # create variables in fw
            ns_lab = fw.create_dataset('lab',
                                       shape=[n_sam,n_line,n_col],
                                       dtype=np.uint8)
            fw['ns_rate'] = np.mean(args.ns_rates)
            
            for si in range(n_sam):
                # load label data
                if args.lab_from_covarea:
                    if si==0: print(f'Generate binary labels from coverage variables ({args.lkey}, threshold={args.cth})')
                    gt = (fr[args.lkey][si]>=args.cth).astype(np.uint8)
                else:
                    if si==0: print(f'Load binary labels from files (key={args.lkey})')
                    gt = (fr[args.lkey][si]>0).astype(np.uint8)
                    
                # convert gt matrix to building indexes
                if args.gind: gt = count_and_index_building(gt)
                mask = np.zeros_like(gt)
                n_bd = gt.max()
                
                if n_bd>0: # buidlings exist in current patch
                    # 1> building list
                    bd_ilist = list(np.arange(n_bd)+1)
                    
                    # 2> select number and types of label noises
                    # number
                    shuffle(nsts)
                    n_nst = nsts[0]
                    # types
                    shuffle(nsts)
                    inds_nst = nsts[:n_nst]-1
                    
                    # 3> determine noise rate
                    shuffle(nsrs)
                    nsr = nsrs[0]
                    
                    # 4> insert label noises type by type
                    portion_rest = 1.
                    for ti in range(n_nst):
                        shuffle(bd_ilist)
                        # noise type (str) to insert
                        nst = args.ns_types[inds_nst[ti]]
                        # corresponding portion
                        p = portion_rest if ti == n_nst-1 else portion_rest*random()
                        portion_rest -= p
                        # number of buildings to modify
                        n_mbd = int(n_bd*nsr*p)
                        # insert label noises one building by one building
                        for bi in range(n_mbd):
                            bind = bd_ilist[0]
                            org_mask = (gt==bind).astype(np.uint8)
                            mask += insert_single_item_noise(org_mask,nst)
                            # remove current processed building id
                            bd_ilist.pop(0)
                    
                    # 5> for add_noise, update add_obj
                    if add_bd:
                        if random()>0.7:
                            add_obj_ind = randint(1,n_bd)
                            add_obj = np.zeros_like(gt,dtype=np.uint8)
                            add_obj[gt==add_obj_ind] = 1
                    
                    # 6> add untouched objects
                    for bi in bd_ilist:
                        mask += (gt==bi).astype(np.uint8)
                        
                else: # no buildings in current patch
                    if add_bd:    
                        if (add_obj is not None) and random()>0.5:
                            mask += add_noise(add_obj)
                
                # convert mask to binary map
                mask = (mask>0).astype(np.uint8)
                
                # save new modified mask
                ns_lab[si] = mask
            
            # close files
            fr.close()
            fw.close()
        
    # 2> concatenate data
    # 1 - saving path
    suff1, suff2, suff3 = args.save_name.split('_')[-3:]
    con_fname = os.path.join(args.save_dir,f'data_all_cities_ns_{suff1}_{suff2}_{suff3}')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # 2 - get number of samples from single files w.r.t. each city
    n_cities = len(all_cities)
    n_ps = np.zeros([n_cities]) 
    for i, city in enumerate(all_cities):
        fp = os.path.join(args.data_dir,city,args.data_name)
        with h5py.File(fp,'r') as fr:
            n_ps[i] = fr['dat'][()].shape[0]
    n_sam = np.sum(n_ps)
    
    # 3 - concatenate the data from all the cities
    with h5py.File(con_fname,'w') as cfw:
        for i, city in enumerate(all_cities):
            # save indexes
            si = int(n_ps[:i].sum())
            ei = int(n_ps[:i+1].sum())
            # open file of current city and resave
            fp = os.path.join(args.data_dir,city,args.save_name)
            with h5py.File(fp,'r') as fr:
                lab0 = fr['lab'][()]
            # create dataset
            if i==0:
                _, h, w = lab0.shape
                lab = cfw.create_dataset('lab',shape=(n_sam,h,w),dtype=lab0.dtype)
            # save
            lab[si:ei] = lab0
            del lab0
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    