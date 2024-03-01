'''
test script for the Germany dataset:
    test accuracy of the well-trained model on the test set
    load weights from checkpoints
    output accuracies (txt) and predicted maps (h5)
'''
import argparse
import datetime
import os, sys, h5py
import numpy as np
from pathlib import Path

# appending self-defined package path
sys.path.append('../AIO2')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


from dataset.data_loading_Building_h5 import BuildingDataset, DataSubset
# model
import segmentation_models_pytorch as smp
from utils.dice_score import multiclass_dice_coeff, dice_coeff, IoU, IoUs


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    # paths of input data
    parser.add_argument('--data_dir', type=str, help='path_to_data')
    parser.add_argument('--test_split', type=str, dest='split', choices=['train','test','val'], default='test')
    parser.add_argument('--data_name', type=str, dest='dname', default='data_all_cities_gt.h5')
    parser.add_argument('--noisy_label_name', type=str, dest='nsname', default='data_all_cities_ns_rm_50.h5')
    parser.add_argument('--partition_name', type=str, dest='pname', default='partitions_v50_ts250.h5')
    # model loading directory
    parser.add_argument('--save_dir', type=str, default='Results')    
    parser.add_argument('--model_state', dest='mstate', choices=['model_state_dict', 'model_state_dict_mit', 'model_state_dict_mep'], 
                        default='model_state_dict', help='Type of model state loaded from ckpt file.')
    parser.add_argument('--epoch', '-e', metavar='E', type=int, default=100, 
                        help='Epoch number from which model is loaded.')
    parser.add_argument('--monte_carlo_run', type=int, dest='mcr', default=1, help='Number of Monte Carlo runs')
    # model settings
    parser.add_argument('--model_type', dest='mt', type=str, default='xception',
                        help='Type of used models. It is mainly utilized to define the encoder part') 
    parser.add_argument('--n_channels', type=int, default=3,
                        help='Number of input channels.') 
    parser.add_argument('--n_classes', type=int, default=1,
                        help='Number of predefined classese.')
    # data settings 
    parser.add_argument('--subset_size', dest='subsize', type=int, default=None, 
                        help='Number of samples for test.')
    parser.add_argument('--batch_size', '-b', metavar='B', type=int, default=25, 
                        help='Batch size')
    parser.add_argument('--num_workers', '-n', metavar='NW', type=int, default=0, 
                        help='Number of workers for dataloaders')    

    return parser.parse_args()


def fix_random_seeds(seed=42):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def add_class_metrics_to_val_dict(val_metrics, caccs, mname='iou'):
    if type(caccs)==float:
        # for 1 class case
        val_metrics[mname] = caccs
    else:
        # for multi-class case
        for i in range(caccs.shape[0]-1):
            val_metrics[f'{mname}_{i}'] = caccs[i].item()
        val_metrics['m{mname}'] = caccs[-1].item()
    
    return val_metrics
    

def add_suffix_to_metric_dict(metrics_dict, suffix, new_dict=None):
    if new_dict is None: new_dict = {}
    for k in metrics_dict:
        new_dict[f'{suffix}_{k}'] = metrics_dict[k]
    
    return new_dict


def cal_precise(mask_true, mask_pred, eps=1e-8):
    precises = []
    recalls = []
    
    n_sam = mask_true.size(dim=0)
    judge_matr = ((mask_pred-mask_true)==0)
    for ni in range(n_sam):
        judge_matr_ = judge_matr[ni]
        mask_true_ = mask_true[ni]
        mask_pred_ = mask_pred[ni]
        pr = ((judge_matr_[mask_pred_!=0]).sum()+eps)/((mask_pred_!=0).sum()+eps)
        rc = ((judge_matr_[mask_true_!=0]).sum()+eps)/((mask_true_!=0).sum()+eps)
        precises.append(pr.item())
        recalls.append(rc.item())
    
    precise = np.array(precises).mean()
    recall = np.array(recalls).mean()
    return precise, recall


def cal_accs(mask_pred, mask_true, n_classes, device):
    eps = 1e-8
    # get labels from softmax outputs
    if n_classes>1:
        mask_pred_hl = mask_pred.argmax(dim=1).float()
    else:
        mask_pred_hl = (torch.sigmoid(mask_pred) > 0.5).float()

    # calculate OAs
    judge_matr = ((mask_pred_hl-mask_true)==0)
    oa_score = (judge_matr).sum()/mask_true.numel()
    if n_classes>1:
        ##### need to modify
        precise_score, recall_score = [], []
        for c in range(n_classes):
            precise_score.append((judge_matr[mask_pred_hl==c]).sum()/(mask_pred_hl==c).sum()+eps)
            recall_score.append((judge_matr[mask_true==c]).sum()/(mask_true==c).sum()+eps)
    else:
        precise_score, recall_score = cal_precise(mask_true, mask_pred_hl)
    
    # calculate ious and dice scores from each images
    if n_classes == 1:
        dice_score = dice_coeff(mask_pred_hl, mask_true, reduce_batch_first=False)
        iou_score = IoU(mask_pred_hl, mask_true, reduce_batch_first=False)
    else:
        mask_true = F.one_hot(mask_true, n_classes).permute(0, 3, 1, 2).float()
        mask_pred = F.one_hot(mask_pred.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
        dice_score = multiclass_dice_coeff(mask_pred[:,:, ...], mask_true[:,:, ...], reduce_batch_first=False)
        iou_score = IoUs(mask_pred[:,:, ...], mask_true[:,:, ...], device, reduce_batch_first=False)
    
    return oa_score, precise_score, recall_score, iou_score, dice_score
    
    

def evaluate(net, dataloader, device, n_classes, experiment='gt_labels', save=None):
    # set model status to validation
    net.eval()
    eps = 1e-8
    
    # place holder for validation metrics
    oa_score = 0.
    dice_score = 0.
    pr_score = torch.zeros([n_classes+1], device=device) if n_classes>1 else 0.
    rc_score = torch.zeros([n_classes+1], device=device) if n_classes>1 else 0.
    iou_score = torch.zeros([n_classes+1], device=device) if n_classes>1 else 0.
    
    # iterate over the validation set
    for i, batch in enumerate(dataloader):
        #image, mask_true = batch['image'], batch['mask']
        image = batch['img']
        ns_masks = torch.squeeze(batch['ns'],1)
        gt_masks = torch.squeeze(batch['gt'],1)
        # decide use which type of labels as reference data
        assert experiment in ['ns_labels','gt_labels'], 'Experiment label type for validation/test is wrong!'
        mask_true = ns_masks if experiment == 'ns_labels' else gt_masks
        
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)        

        # make predictions
        with torch.no_grad():
            mask_pred = net(image).squeeze(axis=1)
        
        # get labels from softmax outputs
        if n_classes>1:
            mask_pred_hl = mask_pred.argmax(dim=1).float()
        else:
            mask_pred_hl = (torch.sigmoid(mask_pred) > 0.5).float()

        # calculate accuracies
        judge_matr = ((mask_pred_hl-mask_true)==0)
        oa_score += (judge_matr).sum()/mask_true.numel()
        if n_classes==1:
            pr_score_, rc_score_ = cal_precise(mask_true, mask_pred_hl)
            pr_score += pr_score_
            rc_score += rc_score_
        else:
            ##### need to modify
            for c in range(n_classes):
                pr_score[c] += ((judge_matr[mask_pred_hl==c]).sum()+eps)/((mask_pred_hl==c).sum()+eps)
                rc_score[c] += ((judge_matr[mask_true==c]).sum()+eps)/((mask_true==c).sum()+eps)

        # calculate ious and dice scores from each images
        if n_classes == 1:
            dice_score += dice_coeff(mask_pred_hl, mask_true, reduce_batch_first=False)
            iou_score += IoU(mask_pred_hl, mask_true, reduce_batch_first=False)
        else:
            mask_true = F.one_hot(mask_true, n_classes).permute(0, 3, 1, 2).float()
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
            dice_score += multiclass_dice_coeff(mask_pred[:,:, ...], mask_true[:,:, ...], reduce_batch_first=False)
            iou_score += IoUs(mask_pred[:,:, ...], mask_true[:,:, ...], device, reduce_batch_first=False)
        
        # save predictions
        if os.path.exists(save):
            si = i*gt_masks.shape[0]
            ei = (i+1)*gt_masks.shape[0]
            with h5py.File(save, 'a') as fa:
                if device=='cpu':
                    fa['maps'][si:ei] = mask_pred_hl.numpy().astype(np.uint8)
                else:
                    fa['maps'][si:ei] = mask_pred_hl.cpu().numpy().astype(np.uint8)

    # congregate val results
    num_val_batches = i+1
    val_metrics = {'dice': dice_score.item()/num_val_batches,
                   'oa': oa_score.item()/num_val_batches}
    if n_classes>1:
        val_metrics = add_class_metrics_to_val_dict(val_metrics, 
                                                    iou_score/num_val_batches, 
                                                    mname='iou')
        val_metrics = add_class_metrics_to_val_dict(val_metrics, 
                                                    pr_score/num_val_batches, 
                                                    mname='precise')
        val_metrics = add_class_metrics_to_val_dict(val_metrics, 
                                                    rc_score/num_val_batches, 
                                                    mname='recall')
    else:
        val_metrics['iou'] = iou_score/num_val_batches
        val_metrics['precise'] = pr_score/num_val_batches
        val_metrics['recall'] = rc_score/num_val_batches
    
    return val_metrics


def modify_module_state_dict_keys(msd):
    prefix = 'module.'
    for k in list(msd.keys()):
        if k.startswith(prefix):
            newk = k[len(prefix):]
            msd[newk] = msd[k]
            del msd[k]
    return msd


def send_optim_tensor_gpu(optimizer,device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                if k == 'step': continue
                state[k] = v.to(device=device)
    return
        

# # # # # # # # # # main function # # # # # # # # # # 
def main():
    t0 = datetime.datetime.now().replace(microsecond=0)
    
    ###### 1 - parameter setting ######
    global args
    args = get_args()
    
    # checkpoint saving dir
    ckp_path = os.path.join(args.save_dir,'checkpoints', f'checkpoint_mcr_{args.mcr}_epoch_{args.epoch}.pth')
    assert os.path.isfile(ckp_path), 'Checkpoint file does not exist!'
    # Prediction saving dir
    pred_dir = os.path.join(args.save_dir,"predicts")
    Path(pred_dir).mkdir(parents=True, exist_ok=True)
    
    # get device information
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    ###### 2 - load data ######
    args.nsname = args.nsname.split('.')[0]+f'_{args.mcr}.'+args.nsname.split('.')[1]
    test_dataset = BuildingDataset(args.data_dir, args.dname, args.nsname, 
                                   args.pname, args.mcr, split=args.split, aug=False)
    # get subset
    n_sam = len(test_dataset)
    if args.subsize is not None:
        assert args.subsize>0 and args.subsize<len(test_dataset), "Size of subset is invalid!"
        ind = np.random.choice(list(range(len(test_dataset))), args.subsize, replace=False)
        test_subset = DataSubset(test_dataset,inds=ind)
        print(f'Test on subset consisting of {args.subsize} samples.')
        if ind.size >= 10:
            print(f'First 10 selected indexes are: {ind[:10]}.')
        else:
            print(f'Selected indexes are: {ind}.')
        n_sam = len(test_subset)
    
    # batch size
    if n_sam<args.batch_size:
        bs = n_sam
    else:
        bs = args.batch_size
        n_bs = n_sam//args.batch_size
        n_sam = n_bs*bs
    print(f'Actual number of tested samples is {n_sam} (bs={bs}, n_bs={n_bs}).')
    
    # construct dataloaders
    test_loader = DataLoader(test_dataset if args.subsize is None else test_subset,
                             batch_size=bs,                              
                             shuffle=False,
                             num_workers=args.num_workers,
                             drop_last=True)
    

    ###### 3 - create model ######
    net = smp.Unet(encoder_name=args.mt,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                   encoder_weights=None,           # use `imagenet` pre-trained weights for encoder initialization
                   in_channels=args.n_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                   classes=args.n_classes,                      # model output channels (number of classes in your dataset)
                   )
        
    
    ###### 4 - load well-trained model ######
    checkpoint = torch.load(ckp_path, map_location=torch.device('cpu'))
    msd = modify_module_state_dict_keys(checkpoint[args.mstate])
    net.load_state_dict(msd)
    print(f"Well-trained model have been loaded from '{ckp_path}' using '{args.mstate}'\n")
    
    # sent model to device
    net.to(device=device)
    
    
    ###### 5 - test the network ######
    print('Start test...')
    net.eval()
        
    # Test
    # create test file
    pred_map_path = os.path.join(pred_dir,f'Predmaps_ep{args.epoch}_{args.split}.h5')
    gt = test_dataset[0]['gt'].squeeze()
    with h5py.File(pred_map_path, 'w') as fw:
        fw.create_dataset('maps', shape=(n_sam, gt.shape[0], gt.shape[1]), 
                          dtype=np.uint8)
    # test
    ts_metrics = evaluate(net, test_loader, device, args.n_classes, 
                          experiment='gt_labels', save=pred_map_path)
    # save evaluation results
    for k in ts_metrics:
        print(f'{k}:{ts_metrics[k]}')
    
    # computing time
    t1 = datetime.datetime.now().replace(microsecond=0)
    print(f'Test is finished|Total spent time:{t1-t0}!' )
    
    
    
if __name__ == '__main__':
    main()
