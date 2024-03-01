import os, cv2
import numpy as np

import torch
import torch.nn.functional as F
from utils.dice_score import multiclass_dice_coeff, dice_coeff, IoU, IoUs


def pixel_wise_correct(net, n_classes, data_loader, ctype, K, device, sdir, 
                       adaptive_K=False, adaptive_type='all', more_or_less='more',
                       eps=1e-8):
    net.eval()
    
    # check parameter settings
    assert ctype=='confidence' or ctype=='uncertainty', \
        'Invalid input for sample selection criterion type setting (expected: confidence or uncertainty)!'
    if adaptive_K:
        assert adaptive_type=='all' or adaptive_type=='class', \
            "Please choose adaptive thresholding setting method for confidence criterion (expected: all or class)!"
        assert more_or_less=='more' or more_or_less=='less', \
            'Please clarify which kind of truncation is used for adaptive threshold setting (expected: more or less) '
    
    # place holder for acc metrics
    oa_score = 0.
    dice_score = 0.
    pr_score = torch.zeros([n_classes+1], device=device) if n_classes>1 else 0.
    rc_score = torch.zeros([n_classes+1], device=device) if n_classes>1 else 0.
    iou_score = torch.zeros([n_classes+1], device=device) if n_classes>1 else 0.
    
    # sample correction in a batch by batch fashion
    for bi, batch in enumerate(data_loader):
        # 1. load data
        images = batch['img'].to(device=device, dtype=torch.float32)   
        mask_true = torch.squeeze(batch['gt'],1).to(device=device, dtype=torch.float32)          
        update_labels = torch.squeeze(batch['ns'],1).to(device=device, dtype=torch.float32)
        fnames = batch['fname']
        
        # 2. make predictions
        with torch.no_grad():
            pred_outs = net(images).squeeze(axis=1)
            pred_logits = torch.sigmoid(pred_outs)
            pred_masks = (pred_logits>0.5).float()
        
        # 3. sample correction
        if ctype == 'uncertainty':
            # 1> calculate uncertainties
            uncertainties = -pred_logits*torch.log(pred_logits)
            uncertainties += -(1-pred_logits)*torch.log(1-pred_logits)
            # 2> threshold setting
            if adaptive_K:
                # # set threshold based on all the samples
                if adaptive_type=='all':
                    thrs = uncertainties.mean(dim=(1,2),keepdim=True)
                    if more_or_less=='more':
                        thrs[thrs<K] = K
                    elif more_or_less=='less':
                        thrs[thrs>K] = K
                elif adaptive_type=='class':
                    thrs_1 = (uncertainties*pred_masks).sum(dim=(1,2),keepdim=True)/(pred_masks.sum(dim=(1,2),keepdim=True)+eps)
                    thrs_0 = (uncertainties*(1-pred_masks)).sum(dim=(1,2),keepdim=True)/((1-pred_masks).sum(dim=(1,2),keepdim=True)+eps)
                    if more_or_less=='more':
                        thrs_1[thrs_1<K] = K
                        thrs_0[thrs_0<K] = K
                    elif more_or_less=='less':
                        thrs_1[thrs_1>K] = K
                        thrs_0[thrs_0>K] = K
                    thrs = thrs_1*pred_masks+thrs_0*(1-pred_masks)
            else:
                thrs = K
            # 3> correct sample labels whoes uncertainties are below threshold
            un_diff = uncertainties - thrs
            update_labels[un_diff<0] = pred_masks[un_diff<0]
        elif ctype == 'confidence':
            # 1> claculate confidence
            confs = pred_masks*pred_logits+(1-pred_masks)*(1-pred_logits)
            # 2> threshold setting
            if adaptive_K:
                # # set threshold based on all the samples
                if adaptive_type=='all':
                    thrs = confs.mean(dim=(1,2),keepdim=True)
                    if more_or_less=='more':
                        thrs[thrs>K] = K
                    elif more_or_less=='less':
                        thrs[thrs<K] = K
                # # set threshold in a class-wise fashion
                elif adaptive_type=='class':
                    thrs_1 = (confs*pred_masks).sum(dim=(1,2),keepdim=True)/(pred_masks.sum(dim=(1,2),keepdim=True)+eps)
                    thrs_0 = (confs*(1-pred_masks)).sum(dim=(1,2),keepdim=True)/((1-pred_masks).sum(dim=(1,2),keepdim=True)+eps)
                    if more_or_less=='more':
                        thrs_1[thrs_1>K] = K
                        thrs_0[thrs_0>K] = K
                    elif more_or_less=='less':
                        thrs_1[thrs_1<K] = K
                        thrs_0[thrs_0<K] = K
                    thrs = thrs_1*pred_masks+thrs_0*(1-pred_masks)
            else:
                thrs = K
            # 3> correct sample labels whoes confidences are greater than threshold
            conf_diff = confs - thrs
            update_labels[conf_diff>0] = pred_masks[conf_diff>0]
        
        # 4. save corrected samples
        for fi, f in enumerate(fnames):
            fpath = os.path.join(sdir,f)
            new_mask = update_labels[fi].squeeze()
            if device!=torch.device('cpu'):
                new_mask = new_mask.cpu()
            new_mask = new_mask.numpy().astype(np.uint8)
            cv2.imwrite(fpath, new_mask)
        
        # 5. calculate accuracies
        judge_matr = ((update_labels-mask_true)==0)
        oa_score += (judge_matr).sum()/mask_true.numel()
        pr_score_, rc_score_ = cal_precise(mask_true, update_labels)
        pr_score += pr_score_
        rc_score += rc_score_
        # calculate ious and dice scores from each images
        dice_score += dice_coeff(update_labels, mask_true, reduce_batch_first=False)
        iou_score += IoU(update_labels, mask_true, reduce_batch_first=False)

    # congregate val results
    num_val_batches = bi+1
    val_metrics = {'dice': dice_score.item()/num_val_batches,
                   'oa': oa_score.item()/num_val_batches}
    val_metrics['iou'] = iou_score/num_val_batches
    val_metrics['precise'] = pr_score/num_val_batches
    val_metrics['recall'] = rc_score/num_val_batches
    
    # reset model status to training
    net.train()
        
    return val_metrics


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

    # calculate OAs
    judge_matr = ((mask_pred-mask_true)==0)
    oa_score = (judge_matr).sum()/mask_true.numel()
    if n_classes>1:
        ##### need to modify
        precise_score, recall_score = [], []
        for c in range(n_classes):
            precise_score.append((judge_matr[mask_pred==c]).sum()/(mask_pred==c).sum()+eps)
            recall_score.append((judge_matr[mask_true==c]).sum()/(mask_true==c).sum()+eps)
    else:
        precise_score, recall_score = cal_precise(mask_true, mask_pred)
    
    # calculate ious and dice scores from each images
    if n_classes == 1:
        dice_score = dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
        iou_score = IoU(mask_pred, mask_true, reduce_batch_first=False)
    else:
        mask_true = F.one_hot(mask_true, n_classes).permute(0, 3, 1, 2).float()
        mask_pred = F.one_hot(mask_pred.to(torch.int64), n_classes).permute(0, 3, 1, 2).float()
        dice_score = multiclass_dice_coeff(mask_pred[:,:, ...], mask_true[:,:, ...], reduce_batch_first=False)
        iou_score = IoUs(mask_pred[:,:, ...], mask_true[:,:, ...], device, reduce_batch_first=False)
    
    return oa_score, precise_score, recall_score, iou_score, dice_score




