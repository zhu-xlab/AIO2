# -*- coding: utf-8 -*-
"""
Noisy label injection for building extraction task 
    - For samples stored as single files and opened with cv2 (Massachussets dataset)

@author: liu_ch
"""

import os
import argparse
import cv2
import numpy as np

from random import seed, shuffle, randint, choice, random
    

def get_args():
    parser = argparse.ArgumentParser(description='Insert label noises into building extractiond data.')
    # data settings
    parser.add_argument('--data_dir', type=str, help='path-to-massachusetts-dataset',)
    parser.add_argument('--partition', type=str, choices=['train','test','val'], default='train')
    # saving directory
    parser.add_argument('--save_dir_name', type=str, default='ns_seg_rm_3')  
    # noise insertion settings
    parser.add_argument('--ns_types', nargs="+", default=['remove'], #['shift','erosion','dilation','rotation'], 
                        help='Candidate noise types.') 
    parser.add_argument('--ns_rates',  nargs="+", type=float, default=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], #, 0.8],
                        help='Candidate noise rates.')
    # generate index layer on the fly or directly load generated ones
    parser.add_argument('--generate_index', dest='gind', action='store_false')  
    # seed
    parser.add_argument('--seed',  type=int, default=86,
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
    mask = np.zeros(list(gt.shape)+[3], dtype=np.int_)
    for ci, cnt in enumerate(contours):
        cv2.drawContours(mask, [cnt], 0, (ci+1,ci+1,ci+1), thickness=cv2.FILLED)
    # convert from RGB to gray
    mask = mask[:,:,0] # cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask*gt  # to refine the shapes of objects in index file
    return mask




if __name__ == '__main__':
    # 0 - preparations
    args = get_args()
    seed(args.seed)
    nsrs = np.array(args.ns_rates)
    
    # directories
    # original data with building indexes
    if args.gind:
        ind_dir = os.path.join(args.data_dir,args.partition,'seg')
    else:
        ind_dir = os.path.join(args.data_dir,args.partition,'index')
    # saving directory for generated noisy labels
    save_dir = os.path.join(args.data_dir,args.partition, args.save_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # get all file names
    fnames = os.listdir(ind_dir)
    
    # 1 - check whether require adding noises for no-object patches
    if 'add' in args.ns_types:
        add_bd = True
        add_obj = None
        args.ns_types.remove('add')
    else:
        add_bd = False
    
    # candidates of numbers/types of label noise insertion into each patch
    nsts = np.arange(len(args.ns_types))+1
    
    # 2 - starting inserting label noises one by one
    for fname in fnames:
        fpath = os.path.join(ind_dir,fname)
        gt = cv2.imread(fpath,0)
        # convert gt matrix to building indexes
        if args.gind: gt = count_and_index_building(gt)
        mask = np.zeros_like(gt,dtype=np.uint8)
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
        mask[mask>1] = 1
        
        # save new modified mask
        psave = os.path.join(save_dir,fname)
        cv2.imwrite(psave, mask)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    