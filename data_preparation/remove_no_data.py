# -*- coding: utf-8 -*-
"""
Data cleansing for Massachussets dataset: delete no-value patches

@author: liu_ch
"""

import os, argparse


def get_args():
    parser = argparse.ArgumentParser(description='Remove images with no data.')
    # data settings
    parser.add_argument('--data_dir', type=str, help='Paht-to-image-folder')
    parser.add_argument('--check_ref_dir', type=str, help='split-to-check')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    ref_dir = os.path.join(args.data_dir, args.check_ref_dir)
    fnames_ref = os.listdir(ref_dir)
    sub_dirs = list(set(os.listdir(args.data_dir))-set([args.check_ref_dir]))
    
    for ld in sub_dirs:    
        ck_dir = os.path.join(args.data_dir, ld)
        fname_rm = list(set(os.listdir(ck_dir))-set(fnames_ref))
        for f in fname_rm:
            fpath = os.path.join(ck_dir,f)
            os.remove(fpath)

    
    
        