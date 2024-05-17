# -*- coding: utf-8 -*-
"""
Noisy label quality evaluation for Massachusetts dataset
(data are saved separately as png files for each patch)

@author: liu_ch
"""

import os
import cv2, argparse
import numpy as np
import matplotlib.pyplot as plt
from random import sample


def get_args():
    parser = argparse.ArgumentParser(description='Massachusetts dataset: Noisy label quality evaluation')
    # data settings
    parser.add_argument('--data_dir', type=str, help='Path-to-massachusetts-dataset')
    parser.add_argument('--partition', type=str, choices=['train','test','val'], default='train')
    # noisy label directory
    parser.add_argument('--ns_dir_name', type=str, default='ns_seg_rm_5')  
    parser.add_argument('--monte_carlo_runs', dest='mcrs', type=int, nargs="+", default=[1,2,3])  
    # evaluation options
    parser.add_argument('--show_eg', action='store_true', help='Whether to show some examples.') 
    parser.add_argument('--data_acc', action='store_false', help='Whether to calculate the accuracy of labels.') 
    return parser.parse_args()


def calculate_acc(gt,ns,eps=1e-6):
    TP = (gt*ns).sum()    
    iou = (TP+eps)/(gt.sum()+ns.sum()-TP+eps)
    oa = (gt-ns==0).sum()/gt.size
    precise = (TP+eps)/(ns.sum()+eps)
    recall = (TP+eps)/(gt.sum()+eps)
    return iou, oa, precise, recall


if __name__ == '__main__':
    args = get_args()
    
    # directories
    img_dir = os.path.join(args.data_dir,args.partition,'data')
    gt_dir = os.path.join(args.data_dir,args.partition,'seg')
    ind_dir = os.path.join(args.data_dir,args.partition,'index')
    ns_dir0 = os.path.join(args.data_dir,args.partition,args.ns_dir_name)
    # get all file names
    fnames = os.listdir(img_dir)
    # determine img ids for visualization
    if args.show_eg: seg_ids = sample(np.arange(len(fnames)).tolist(),2)
    
    # place holder for final results
    if args.data_acc:
        nm = len(args.mcrs)
        acc_dict = {}
        for k in ['iou_s','iou_a','rc_s','rc_a','pr_s','pr_a','oa_s','oa_a',
                  'n_org','n_ns','n_rm','rate_ns_s','rate_ns_a','rate_rm_s','rate_rm_a']:
            acc_dict[k] = np.zeros([nm])
    
    # loop with mcrs
    for mi, mcr in enumerate(args.mcrs):
        ns_dir = ns_dir0 + f'_{mcr}' 
        
        # place holders for accuracy calculation
        if args.data_acc:
            ious, oas, precises, recalls = [], [], [], []
            TPs, n_right, n_gt, n_ns = [], [], [], []
            nb_all, nb_remain = [], []
        
        # save index files for each gt
        for i,fname in enumerate(fnames):
            # fname = '23429020_15_5.png'   # test
            # fname = '22678930_15_25.png'   # train
            # read data
            img_path = os.path.join(img_dir,fname)
            gt_path = os.path.join(gt_dir,fname)
            ind_path = os.path.join(ind_dir,fname)
            ns_path = os.path.join(ns_dir,fname)
            
            img = cv2.imread(img_path)
            gt = cv2.imread(gt_path,0).astype(np.int8)
            inds = cv2.imread(ind_path,0)
            ns = cv2.imread(ns_path,0).astype(np.int8)
            
            if args.show_eg:
                if i in seg_ids:
                    # differences between gt and ns
                    diff = gt-ns
                    diff[diff<0] = 0
                    
                    # show images
                    # plt.close('all')
                    plt.figure(figsize=(16,4))
                    plt.subplot(141)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(f'{mcr}-{i}-optical')
                    plt.subplot(142)
                    plt.imshow(gt, interpolation='none')
                    plt.axis('off')
                    plt.title(f'{mcr}-{i}-gt')
                    plt.subplot(143)
                    plt.imshow(diff, interpolation='none')
                    plt.axis('off')
                    plt.title(f'{mcr}-{i}-removed')
                    plt.subplot(144)
                    plt.imshow(ns, interpolation='none')
                    plt.axis('off')
                    plt.title(f'{mcr}-{i}-noisy labels')
            
            if args.data_acc:
                # for single ones
                iou, oa, pr, rc = calculate_acc(gt,ns)
                ious.append(iou)
                oas.append(oa)
                precises.append(pr)
                recalls.append(rc)
                
                # for taking all as one
                TPs.append((gt*ns).sum())
                n_right.append((gt-ns==0).sum())
                n_gt.append(gt.sum())
                n_ns.append(ns.sum())
                
                # for counting individual buildings
                nb_all.append(inds.max())
                # count remianing buildings
                # find contours
                contours, hierarchy = cv2.findContours(ns.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                nb_remain.append(len(contours))
                
        # final results
        if args.data_acc:
            # for averaging single ones
            ious = np.array(ious)
            acc_dict['iou_s'][mi] = ious.mean()
            oas = np.array(oas)
            acc_dict['oa_s'][mi] = oas.mean()
            precises = np.array(precises)
            acc_dict['pr_s'][mi]  = precises.mean()
            recalls = np.array(recalls)
            acc_dict['rc_s'][mi]  = recalls.mean()
            
            # for taking all as one
            TPs = np.array(TPs)
            n_right = np.array(n_right)
            n_gt = np.array(n_gt)
            n_ns = np.array(n_ns)
            acc_dict['iou_a'][mi] = TPs.sum()/(n_gt.sum()+n_ns.sum()-TPs.sum())
            acc_dict['oa_a'][mi] = n_right.sum()/(len(fnames)*gt.size)
            acc_dict['pr_a'][mi] = TPs.sum()/n_ns.sum()
            acc_dict['rc_a'][mi] = TPs.sum()/n_gt.sum()
            
            # for counting
            nb_all = np.array(nb_all)
            acc_dict['n_org'][mi] = nb_all.sum()
            nb_remain = np.array(nb_remain)
            acc_dict['n_ns'][mi] = nb_remain.sum()
            acc_dict['n_rm'][mi] = acc_dict['n_org'][mi]-acc_dict['n_ns'][mi]
            nb_rate = (nb_remain+1e-8)/(nb_all+1e-8)
            acc_dict['rate_ns_s'][mi] = nb_rate.mean()
            acc_dict['rate_ns_a'][mi] = acc_dict['n_ns'][mi]/acc_dict['n_org'][mi]
            rm_rate = (nb_all-nb_remain+1e-8)/(nb_all+1e-8)
            acc_dict['rate_rm_s'][mi] = rm_rate.mean()
            acc_dict['rate_rm_a'][mi] = acc_dict['n_rm'][mi]/acc_dict['n_org'][mi]
    
    # get mean results
    if args.data_acc:
        if nm>1:
            for k in ['iou_a','oa_a','rate_rm_a']:
                accs = acc_dict[k]
                print(f"{k}:{np.mean(accs)*100:0.2f}$\pm${np.std(accs)*100:0.2f}")
            
        
        
        
    
    
    
        
        