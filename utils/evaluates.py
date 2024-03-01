import sys
import numpy as np

# appending self-defined package path
sys.path.append('../AIO2')

import torch
import torch.nn.functional as F
from utils.dice_score import multiclass_dice_coeff, dice_coeff, IoU, IoUs # dice_loss


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
    
    

def evaluate(net, dataloader, device, experiment='gt_labels', n_classes=None):
    # set model status to validation
    net.eval()
    eps = 1e-8
    if n_classes is None: n_classes = net.n_classes
    
    # place holder for validation metrics
    oa_score = 0.
    dice_score = 0.
    pr_score = torch.zeros([n_classes+1], device=device) if n_classes>1 else 0.
    rc_score = torch.zeros([n_classes+1], device=device) if n_classes>1 else 0.
    iou_score = torch.zeros([n_classes+1], device=device) if n_classes>1 else 0.
    
    # iterate over the validation set
    for i,batch in enumerate(dataloader):
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
            mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
            dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            iou_score += IoU(mask_pred, mask_true, reduce_batch_first=False)
        else:
            mask_true = F.one_hot(mask_true, n_classes).permute(0, 3, 1, 2).float()
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
            dice_score += multiclass_dice_coeff(mask_pred[:,:, ...], mask_true[:,:, ...], reduce_batch_first=False)
            iou_score += IoUs(mask_pred[:,:, ...], mask_true[:,:, ...], device, reduce_batch_first=False)
    
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
    
    # reset model status to training
    net.train()
    
    return val_metrics


# # # # # training results within each batch # # # # #
def evaluate_batch_once(pred_masks, true_masks, n_classes, tr_accs, 
                        device, suffix, batch_to_wandb=False):
    tr_oa_, tr_pr_, tr_rc_, tr_iou_, tr_dice_ = cal_accs(pred_masks, 
                                                         true_masks, 
                                                         n_classes, 
                                                         device)
    # for calculating accumulated accuracies
    if len(tr_accs)>0:
        tr_accs['oa'] += tr_oa_
        tr_accs['pr']  += tr_pr_
        tr_accs['rc']  += tr_rc_
        tr_accs['iou']  += tr_iou_
        tr_accs['dice']  += tr_dice_
    
    # create training accuracy dict for recording to wandb
    if batch_to_wandb:
        # add accs into btr_metrics       
        btr_metrics = {'dice': tr_dice_,
                       'oa': tr_oa_}               
        if n_classes>1:
            btr_metrics = add_class_metrics_to_val_dict(btr_metrics, tr_iou_, mname='iou')
            btr_metrics = add_class_metrics_to_val_dict(btr_metrics, tr_pr_, mname='precise')
            btr_metrics = add_class_metrics_to_val_dict(btr_metrics, tr_rc_, mname='recall')
        else:
            btr_metrics['iou'] = tr_iou_
            btr_metrics['precise'] = tr_pr_
            btr_metrics['recall'] = tr_rc_
        btr_dict = add_suffix_to_metric_dict(btr_metrics, suffix=suffix)
    else:
        btr_dict = {}
        
    return tr_accs, btr_dict
        

def evaluate_batch(pred_masks, true_masks, n_classes, tr_accs, 
                   device, suffix,
                   batch_to_wandb=False,
                   gt_masks=None, trg_accs=None):
    tr_accs, btr_dict = \
        evaluate_batch_once(pred_masks, true_masks, n_classes, tr_accs, 
                            device, suffix=suffix+'b',
                            batch_to_wandb=batch_to_wandb)
    
    # calculte accs w.r.t. gt
    if gt_masks is not None:
        assert trg_accs is not None, "Please provide trg_accs!"
        tr_gt_accs, btr_gt_dict = \
            evaluate_batch_once(pred_masks, gt_masks, n_classes, trg_accs, 
                                device, suffix=suffix+'gb',
                                batch_to_wandb=batch_to_wandb)
        
        return tr_accs, btr_dict, tr_gt_accs, btr_gt_dict
    else:
        return tr_accs, btr_dict
# # # # # training results within each batch # # # # #


def epoch_log_dict(tr_dict, accs, batch_in_epoch, n_classes, suffix):
    acc_metrics = {'dice': accs['dice'].item()/batch_in_epoch,
                  'oa': accs['oa'].item()/batch_in_epoch}
    if n_classes>1:
        acc_metrics = add_class_metrics_to_val_dict(acc_metrics, accs['iou']/batch_in_epoch, mname='iou')
        acc_metrics = add_class_metrics_to_val_dict(acc_metrics, accs['pr']/batch_in_epoch, mname='precise')
        acc_metrics = add_class_metrics_to_val_dict(acc_metrics, accs['rc']/batch_in_epoch, mname='recall')
    else:
        acc_metrics['iou'] = accs['iou']/batch_in_epoch
        acc_metrics['precise'] = accs['pr']/batch_in_epoch
        acc_metrics['recall'] = accs['rc']/batch_in_epoch
    tr_dict = add_suffix_to_metric_dict(acc_metrics, suffix=suffix, new_dict=tr_dict)
    return tr_dict