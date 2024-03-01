# -*- coding: utf-8 -*-
"""
clean data according to Nikolai's visual check results
validate indices are stored in .csv files

@author: liu_ch
"""

import argparse
import os, h5py
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='Data cleaning based on Nikolai\'s visual check results for the Germany dataset.')
    # data settings
    parser.add_argument('--data_dir', type=str, default='path-to-germany-dataset')
    parser.add_argument('--source_data_name', dest='srn', type=str, default='PatchSz320_LabPerc0_nonoverlap.h5')
    parser.add_argument('--target_data_name', dest='tgn', type=str, default='PatchSz320_LabPerc0_nov.h5')
    parser.add_argument('--valid_indices_name', dest='vin', type=str, default='valid_indices.csv')
    # coverage threshold for label generation
    parser.add_argument('--covarea_threshold', dest='cth', type=int, default=50)
    # patch size
    parser.add_argument('--patch_size', dest='psz', type=int, default=320,
                        help='Only be used when we need to generate source_data_name first.')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    all_cities = os.listdir(args.data_dir)
    
    for city in all_cities:
        # 1 - check source file 
        srd = os.path.join(args.data_dir,city,args.srn)
        # if source file doesn't exist, create one by removing overlap parts (cropping)
        if not os.path.exists(srd):
            fn0 = args.srn.rpartition('_')[0] + '.h5'
            srd0 = srd = os.path.join(args.data_dir,city,fn0)
            assert os.path.exists(srd0), 'Both source files don\'t exist.'
            
            with h5py.File(srd0,'r') as fr:
                with h5py.File(srd,'w') as fw:
                    fw['covarea'] = fr['covarea'][:,:args.psz,:args.psz]
                    fw['dat'] = fr['dat'][:,:,:args.psz,:args.psz]
                    fw['instances'] = fr['instances'][:,:args.psz,:args.psz]
                    fw['instances_from_labels'] = fr['instances_from_labels'][:,:args.psz,:args.psz]
                    fw['lab'] = fr['lab'][:,:args.psz,:args.psz]
                    fw['offset'] = fr['offset'][()]
                    fw['pos'] = fr['pos'][()]
                    fw['version'] = fr['version'][()]
    
        # 2 - remove bad samples and write target files
        # read valid indices
        vid = os.path.join(args.data_dir,city,args.vin)
        valid_index = np.genfromtxt(vid, dtype=int, skip_header=1, delimiter=',')
        valid_index = np.sort(valid_index)
        # remove and write target file
        tgd = os.path.join(args.data_dir,city,args.tgn)
        with h5py.File(srd,'r') as fr:
            with h5py.File(tgd,'w') as fw:
                fw['dat'] = fr['dat'][valid_index]
                fw['lab'] = (fr['covarea'][valid_index]>=args.cth).astype(np.uint8)
                fw['valid_indices'] = valid_index
                fw['covarea_threshold'] = args.cth
                
                
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    