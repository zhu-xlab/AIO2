# -*- coding: utf-8 -*-
"""
For counting and indexing buildings in gt in Massachussets dataset

@author: liu_ch
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# parent dir of data
data_dir = 'path-to-massachusetts-dataset'


if __name__ == '__main__':
    # options
    plt_show = True
    save_ind = False
    assert plt_show+save_ind==1, 'Only allowed to choose one from showing and saving options.'
    
    # partition
    partition = 'train'
    
    # directories
    img_dir = os.path.join(data_dir,partition,'data')
    gt_dir = os.path.join(data_dir,partition,'seg')
    save_dir = os.path.join(data_dir,partition,'index')
    
    # check whether dir exists
    os.makedirs(save_dir, exist_ok=True)
    
    # get all file names
    fnames = os.listdir(img_dir)
    
    # save index files for each gt
    for i,fname in enumerate(fnames):
        # read data
        img_path = os.path.join(img_dir,fname)
        gt_path = os.path.join(gt_dir,fname)
        ind_path = os.path.join(save_dir,fname)
        
        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path,0)
        
        # generate and save building indexes
        if save_ind:
            # find contours
            contours, hierarchy = cv2.findContours(gt,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            
            # index buildings
            mask = np.zeros(list(gt.shape)+[3], np.uint8)
            for ci, cnt in enumerate(contours):
                cv2.drawContours(mask, [cnt], 0, (ci+1,ci+1,ci+1), thickness=cv2.FILLED)
            # convert from RGB to gray
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            # save building indexes
            
            cv2.imwrite(ind_path, mask)
        else:
            mask = cv2.imread(ind_path,0)
        
        # show images
        if plt_show:
            plt.close('all')
            plt.figure(figsize=(12,4))
            plt.subplot(131)
            plt.imshow(img)
            plt.axis('off')
            plt.subplot(132)
            plt.imshow(gt)
            plt.axis('off')
            plt.subplot(133)
            plt.imshow(mask)
            plt.axis('off')


        
        