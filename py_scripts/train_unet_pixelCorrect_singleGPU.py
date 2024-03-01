'''
Pixel-wise correction for the Massachusetts dataset:
    training UNet models with pixel-wise correction alongside the proposed ACT for early learning detection to adaptively trigger the sample selection procedure.
'''
import argparse
import datetime
import os, sys, json, shutil
import wandb
import numpy as np
from pathlib import Path

# appending self-defined package path
sys.path.append('../AIO2')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader#, random_split

from dataset.data_loading_Building import BuildingDataset
from models import UNet, ResNet18UNet
# from models.unet_resnet18 import ResNet18UNet
from utils.dice_score import dice_loss
import utils.self_ensembling as use
import utils.early_learning_detection as eld
import utils.pixel_wise_correct as pwc
import utils.evaluates as evl


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks (pixel-wise correction)')
    # paths of input data
    parser.add_argument('--data_path', type=str, help='path_to_data')
    parser.add_argument('--noise_dir_name', type=str, dest='ndn', default='seg_ns')
    parser.add_argument('--monte_carlo_run', type=int, dest='mcr', default=1, help='Number of Monte Carlo runs')
    # saving directory
    parser.add_argument('--save_dir', type=str, default='Results')  
    # model settings
    parser.add_argument('--model_type', dest='mt', type=str, choices=['unet','res18'], default='unet',
                        help='Type of used models. It is mainly utilized to define the encoder part') 
    parser.add_argument('--loss_type', type=str, choices=['c','d','cd'], default='cd',
                        help='Type of used segmentation loss function for training. Specifically, \'c\' represents CrossEntropy loss, \'d\' represents Dice loss, \'cd\' represents combined loss function of the two.') 
    parser.add_argument('--resume', type=str, default=False, 
                        help='Load model from a .pth file to continue training')
    parser.add_argument('--resume_ema', action='store_true',
                        help='Load ema models from a .pth file as well or not if resume=True')
    parser.add_argument('--resume_from_detection', dest='rs_fd', action='store_true',
                        help='Whether loading checkpoint from the detected point or not')
    # data settings 
    parser.add_argument('--batch_size', dest='bs', type=int, default=25, 
                        help='Batch size')
    parser.add_argument('--test_batch_size', dest='tbs', type=int, default=25, 
                        help='Test and val batch size')
    parser.add_argument('--num_workers', '-n', metavar='NW', type=int, default=0, 
                        help='Number of workers for dataloaders')    
    # training settings
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, 
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', '-l', dest='lr', metavar='LR', type=float, default=0.001, 
                        help='Initial learning rate')
    parser.add_argument('--amp', action='store_false', 
                        help='Use mixed precision or not')
    parser.add_argument('--lr_adjust', dest='schd', type=bool, default=False, 
                        help='Adjust lr according to validation accuracies or not')
    parser.add_argument('--cal_tr_acc', dest='cal_tr', action='store_true', 
                        help='Whether to calculate training accuracies or not')
    # wandb settings
    parser.add_argument('--project_name', dest='pjn', type=str, default='aio2', 
                        help='Name of the wandb project to record statistics.')
    parser.add_argument('--entity_name', dest='entity', type=str, default='v', 
                        help='Name of the entity of wandb.')
    parser.add_argument('--wandb_mode', dest='wmode', type=str, choices=["online", "offline", "disabled"], default='online',
                        help='Setting of wandb.init modes ("online", "offline" or "disabled")')
    # self-ensembling
    parser.add_argument('--ensembling_epoch', dest='esb_ep', action='store_true', 
                        help='Whether using self-ensembling model after each epoch if ensembling=True')
    parser.add_argument('--alpha_it', type=float, default=0.999, 
                        help='Alpha used for ema model updating after each iteration')
    parser.add_argument('--alpha_ep', type=float, default=0.99, 
                        help='Alpha used for ema model updating after each epoch')# other settings
    # early learning detection
    parser.add_argument('--el_window_sizes', dest='el_wsizes', type=int, nargs="+", 
                        help='List of sliding window sizes used for numerical gradient calculation')
    # sample correction
    parser.add_argument('--correct_model', dest='cmd', type=str, choices=['teacher','student'], default='teacher', 
                        help='Model type used for providing corrected labels.')
    parser.add_argument('--criterion_type', dest='ctype', type=str, choices=['confidence','uncertainty'], 
                        help='Type of sample selection criterion for pixel-wise label correction.')
    parser.add_argument('--alpha', type=float, default=0.2, 
                        help='Weight of the additional term (CE calculated w.r.t. original noisy labels) in the correction loss function')
    parser.add_argument('--K', type=float, default=0.1, 
                        help='For selection based on uncertainty, K is the lower bound for uncertainty threshold setting used for sample selection;'\
                            'For selection based on confidence, K is the lower treshold for sample selection.')
    parser.add_argument('--adaptive_K', dest='adp_K', action='store_true', 
                        help='If using adaptive K or not. ref. Dong et al.')
    parser.add_argument('--adaptive_type', dest='adp_Kt', type=str, choices=['all','class'], default='all',
                        help='How to calculate thresholds, i.e., from all the samples or in a class-wise fashion.')
    parser.add_argument('--adaptive_truncate_type', dest='adp_Ktct', type=str, choices=['more','less'], default='more',
                        help='How to truncate threshold using K. Keep more or less samples to correct.')
    # others
    parser.add_argument('--display_interval', dest='display_inv', type=int, default=1, 
                        help='Set the display interval during the training')
    parser.add_argument('--save_interval', dest='save_inv', type=int, default=1,
                        help='Set the save interval during the training')
    parser.add_argument("--print_to_log", action="store_false", help="If true, directs std-out to log file")
    parser.add_argument("--batch_to_wandb", action="store_true", help="If true, log batch-wise training results to wandb")
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def fix_random_seeds(seed=42):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# # # # for model loading # # # #
def modify_module_state_dict_keys(msd):
    prefix = 'module.'
    for k in list(msd.keys()):
        if k.startswith(prefix):
            newk = k[len(prefix):]
            msd[newk] = msd[k]
            del msd[k]
    return msd


def send_optim_tensor_device(optimizer,device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                if k == 'step': continue
                state[k] = v.to(device=device)
    return
# # # # for model loading # # # #
     

def create_acc_dict(acc_names=['oa', 'pr', 'rc', 'dice', 'iou']):
    acc_dict = {}
    for tra in acc_names:
        acc_dict[tra] = 0.
    return acc_dict


# # # # # # # # # # main function # # # # # # # # # # 
def main():
    t0 = datetime.datetime.now().replace(microsecond=0)
    
    ###### 1 - parameter setting ######
    global args
    args = get_args()
    fix_random_seeds(args.seed)
    os.environ["WANDB_MODE"] = args.wmode
        
    # set unet_type according to model settings: e.g., unet_ns_rm_5_pxl
    ndn_split = args.ndn.split('_')   # e.g., ndn=ns_seg_rm_5
    assert ndn_split[1]=='seg' and len(ndn_split)==4, "Directory name of noisy labels is invalid!"
    args.unet_type = f'{args.mt}_ns_{ndn_split[-2]}_{ndn_split[-1]}_pxl'   # e.g., unet_ns_rm_5_pxl
    args.ndn += f'_{args.mcr}'
    if args.adp_K:
        args.job_type = f'{args.cmd[0]}{args.ctype[0]}_{args.adp_Kt[0]}{args.adp_Ktct[0]}K{int(args.K*100)}_a{int(args.alpha*100)}'  
    else:
        args.job_type = f'{args.cmd[0]}{args.ctype[0]}_K{int(args.K*100)}_a{int(args.alpha*100)}'  
    # e.g., tce_K80_a10 -> teacher, confidence, epoch (correction interval), K=0.8, alpha=0.1
        
    # set name component for resuming cases
    last_epoch = 0
    if args.resume:
        assert os.path.isfile(args.resume), 'Model loading path is invalid!'
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        last_epoch = checkpoint['epoch']
        if not args.rs_fd:
            args.job_type += f'_rs{last_epoch}' # e.g., tce_K80_a10_rs70
            tr_iou_mit = checkpoint['tr_iou_mit'].tolist()
            ngs_dict = eld.cal_ngs_to_dict(tr_iou_mit, args.el_wsizes)
            detect_eps = np.zeros(len(args.el_wsizes))
    else:
        args.rs_fd = False
        tr_iou_mit = []
        ngs_dict = {b:[] for b in args.el_wsizes}
        detect_eps = np.zeros(len(args.el_wsizes))
    
    # saving dirs
    # parent dir
    pdir = os.path.join(args.save_dir, args.unet_type, args.job_type, f"mcr_{args.mcr}_seed_{args.seed}")
    Path(pdir).mkdir(parents=True, exist_ok=True)
    # checkpoint saving dir
    args.checkpoints_dir = os.path.join(pdir,"checkpoints")
    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    # corrected sample saving dir
    args.cs_dir = os.path.join(pdir,"corrected_labels")
    if os.path.exists(args.cs_dir): shutil.rmtree(args.cs_dir)
    shutil.copytree(os.path.join(args.data_path, 'train', args.ndn), args.cs_dir)

    # statistic recording in txt files
    # record params into txt file
    with open(f'{pdir}/params.txt', 'w') as ft:
        json.dump(args.__dict__, ft)
    # record system outputs to txt file
    if args.print_to_log:
        sys.stdout = open(f'{pdir}/log.txt', 'w')
    
    # output the model type
    print(f"Network architecture: {args.mt} ({args.unet_type})\n")
    
    # get device information
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    ###### 2 - load data ######
    train_dataset = BuildingDataset(args.data_path, noise_dir_name=args.cs_dir, split='train', 
                                    aug=True, origin_ns_dir=args.ndn)
    train_dataset_c = BuildingDataset(args.data_path, noise_dir_name=args.cs_dir, split='train', 
                                      aug=False)
    val_dataset = BuildingDataset(args.data_path, noise_dir_name=args.ndn, split='val')
    test_dataset = BuildingDataset(args.data_path, noise_dir_name=args.ndn, split='test')

    # construct dataloaders
    train_loader = DataLoader(train_dataset,
                              batch_size=args.bs,
                              shuffle=True,
                              num_workers=args.num_workers,
                              drop_last=True)
    train_loader_c = DataLoader(train_dataset_c,
                                batch_size=args.bs,
                                shuffle=True,
                                num_workers=args.num_workers,
                                drop_last=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.tbs,                              
                            shuffle=False,
                            num_workers=args.num_workers,
                            drop_last=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.tbs,                              
                             shuffle=False,
                             num_workers=args.num_workers,
                             drop_last=True)
    
    n_batch_tr = int(len(train_dataset)/args.bs)
    n_batch_val = int(len(val_dataset)/args.tbs)
    n_batch_ts = int(len(test_dataset)/args.tbs)
    print(f'len/n_batch of train, val, test:{len(train_dataset)}/{n_batch_tr}, {len(val_dataset)}/{n_batch_val}, {len(test_dataset)}/{n_batch_ts}')


    ###### 3 - create model ######
    if args.mt=='unet':
        net = UNet(n_channels=3, n_classes=1) 
    elif args.mt=='res18':
        net = ResNet18UNet(n_channels=3, n_classes=1)   
        

    ###### 4 - set the loss and optimizer ######
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-8)
    if args.schd: scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    # # # # segmentation loss # # # #
    # Cross Entropy
    if 'c' in args.loss_type:
        if net.n_classes>1:
            criterion = nn.CrossEntropyLoss()
            print("CrossEntropy loss is used!")
        else:
            criterion = nn.BCEWithLogitsLoss()
            print("BinaryCrossEntropy loss is used!")
    # Dice
    if 'd' in args.loss_type:
        print("Dice loss is used!")
    
    
    ###### 5 - load pre-trained model if have any ######
    correct = False
    if args.resume:
        msd = modify_module_state_dict_keys(checkpoint['model_state_dict'])
        net.load_state_dict(msd)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # send optimizer tensor to gpu
        send_optim_tensor_device(optimizer,device)
        
        # set correct = true if resumed from the detected epoch
        if args.rs_fd: correct = True 
        
        print(f"Pre-trained model and optimizer have been loaded from '{args.resume}'")
        print(f"Last epoch recorded in ckpt file is {checkpoint['epoch']} (loss={checkpoint['loss']}).")
    else:
        print('No pretrained weights, will start from scratch.')
        
    # sent model to device
    net.to(device=device)
    
    # set wandb
    wandb.init(project=args.pjn,#'Noisy label with reconstruction loss',
               entity=args.entity,
               config=args,
               group=args.unet_type,
               job_type=args.job_type,
               name=f'lt_{args.loss_type}_{args.mcr}_r{last_epoch}',
               dir=pdir,
               resume=False)
    # # Log the network weight histograms (optional)
    # wandb.watch(net)
    
    # # # # self-ensembling # # # #
    # create self-ensembling models
    # EMA-MIT
    net_ema_it = use.create_ema_model(net, device, name='mit',
                                      resume=args.resume if args.resume_ema else False)
    # EMA-MEP
    if args.esb_ep:
        net_ema_ep = use.create_ema_model(net, device, name='mep',
                                          resume=args.resume if args.resume_ema else False)
            
    
    ###### 5 - train the network ######
    print('\nStart training...')
    net.train()
    net_ema_it.train()
    if args.esb_ep: net_ema_ep.train()
    steps = 0
    # start training
    fepochs = args.epochs*2
    n_back = 0
    wp = 'bs_'
    # start training
    for epoch in range(last_epoch, fepochs):
        # update epoch index
        epoch -= n_back
        # place holders to record loss values during training
        epoch_loss = 0  # recording averaged (combined) loss for each epoch
        batch_in_epoch = 0 # recording number of batches in each epoch
        running_loss = 0 # recording sum of loss for showing temporily averaged loss values
        if args.loss_type=='cd':
            epoch_loss_c, epoch_loss_d = 0, 0
        # place holders for training accs
        if args.cal_tr:
            tr_accs = create_acc_dict()
            tr_gt_accs = create_acc_dict()
        # place holders for ensembling model after each iteration
        tr_accs_mit = create_acc_dict()
        tr_gt_accs_mit = create_acc_dict()
        
        # - CORRECTION
        if correct:
            ns_metrics = pwc.pixel_wise_correct(net_ema_it if args.cmd=='teacher' else net, 
                                                net.n_classes,
                                                train_loader_c, 
                                                args.ctype, 
                                                args.K, 
                                                device, 
                                                args.cs_dir,
                                                adaptive_K=args.adp_K,
                                                adaptive_type=args.adp_Kt, 
                                                more_or_less=args.adp_Ktct)
            print(f'[Epoch-{epoch}|Correct] IoU:{ns_metrics["iou"]}, Recall:{ns_metrics["recall"]}, Precision:{ns_metrics["precise"]}, OA:{ns_metrics["oa"]}')
        
        # - TRAINING in each batch
        for bi, batch in enumerate(train_loader):
            steps += 1
            # 1> load data
            images = batch['img'].to(device=device, dtype=torch.float32)            
            true_masks = torch.squeeze(batch['ns'],1).to(device=device, dtype=torch.float32)
            gt_masks = torch.squeeze(batch['gt'],1).to(device=device, dtype=torch.float32)
            ons_masks = torch.squeeze(batch['ons'],1).to(device=device, dtype=torch.float32)
            # check if number of input channels is compatible to model setting
            assert images.shape[1] == net.n_channels, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded images have {images.shape[1]} channels. Please check that the images are loaded correctly.'
            
            # 3> training student model
            # a - forward
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred_logits = net(images).squeeze(axis=1)
                dice_pred = torch.sigmoid(pred_logits)
                
                # calculate loss
                # - crossentropy
                loss_cc = criterion(pred_logits, true_masks) if 'c' in args.loss_type else torch.tensor(0.0)
                # - Dice
                if 'd' in args.loss_type:
                    if net.n_classes>1:
                        dice_true = F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float()
                        loss_cd = dice_loss(dice_pred, dice_true, multiclass=True)
                    else:
                        loss_cd = dice_loss(dice_pred, true_masks, multiclass=False)
                else: 
                    loss_cd = torch.tensor(0.0)
                # - final combined loss
                loss = loss_cc + loss_cd     
                
                # - loss regularization
                if correct:
                    if args.alpha>0:
                        reg = criterion(pred_logits, ons_masks) if 'c' in args.loss_type else torch.tensor(0.0)
                        reg += dice_loss(dice_pred, ons_masks, multiclass=False) if 'd' in args.loss_type else torch.tensor(0.0)
                        loss += args.alpha*reg
                        
            # b - backforward/optimization
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            # record loss
            running_loss += loss.item()
            epoch_loss += loss.item()
            if args.loss_type=='cd':
                epoch_loss_c += loss_cc.item()
                epoch_loss_d += loss_cd.item()
            
            # 4> training acc in each iteration
            if args.cal_tr:
                pred_masks = (torch.sigmoid(pred_logits) > 0.5).float()
                tr_accs, btr_dict, tr_gt_accs, btr_gt_dict = \
                    evl.evaluate_batch(pred_masks, ons_masks, net.n_classes, 
                                       tr_accs, device, suffix='tr',
                                       batch_to_wandb=args.batch_to_wandb,
                                       gt_masks=gt_masks,
                                       trg_accs=tr_gt_accs)
            
            # 5> self-ensembling after each iteration
            # update self-ensembling models after each iteration
            use.update_ema_variables(net, net_ema_it, args.alpha_it)
            
            # make predictions using ema model
            with torch.no_grad():
                pred_logits_ema_it = net_ema_it(images).squeeze(axis=1)
                pred_mit_masks = (torch.sigmoid(pred_logits_ema_it) > 0.5).float()
            
            tr_accs_mit, btr_dict_mit, tr_gt_accs_mit, btr_gt_dict_mit = \
                evl.evaluate_batch(pred_mit_masks, ons_masks, net.n_classes, 
                                   tr_accs_mit, device, suffix='mit_tr',
                                   batch_to_wandb=args.batch_to_wandb,
                                   gt_masks=gt_masks,
                                   trg_accs=tr_gt_accs_mit)
                                   
            # 6> Log batch-wise training metrics to wandb
            if args.batch_to_wandb:
                wandb.log({'step':steps, 'bloss':loss.item(),
                           'bloss_c':loss_cc.item(), 'bloss_d':loss_cd.item(),
                           **btr_dict,**btr_gt_dict,
                           **btr_dict_mit,**btr_gt_dict_mit})
            
            # update counters
            batch_in_epoch += 1
            # show intermedium training results
            if batch_in_epoch % args.display_inv == 0:    # print training results  
                t1 = datetime.datetime.now().replace(microsecond=0)  
                print('[%s|Epoch-%d|Batch-%d] lr: %.4f|current training loss: %.4f|average training loss: %.4f' 
                      % (t1-t0, epoch+1, batch_in_epoch, optimizer.param_groups[0]['lr'], loss.item(), running_loss/args.display_inv))
                running_loss = 0
                

        # - VALIDATE & TEST after each epoch
        # 1> Validation
        # - validate on noisy labels
        val_metrics = evl.evaluate(net, val_loader, device, experiment='ns_labels')
        # reframe val_metrics to val_dict
        val_dict = evl.add_suffix_to_metric_dict(val_metrics, suffix=f'{wp}v')
        # - validate on gt
        val_metrics_gt = evl.evaluate(net, val_loader, device, experiment='gt_labels')
        # merge val_metrics_gt into val_dict
        val_dict = evl.add_suffix_to_metric_dict(val_metrics_gt, suffix=f'{wp}vg',new_dict=val_dict)
        # adjust lr according to validation results 
        if args.schd: 
            if net.n_classes==1: 
                scheduler.step(val_metrics['iou']) 
            else:
                scheduler.step(val_metrics['miou']) 
        
        # 2> Test
        # - test on gt
        ts_metrics = evl.evaluate(net, test_loader, device, experiment='gt_labels')
        # reframe ts_metrics to ts_dict
        ts_dict = evl.add_suffix_to_metric_dict(ts_metrics, suffix=f'{wp}ts')
        # - test on noisy labels
        ts_metrics_ns = evl.evaluate(net, test_loader, device, experiment='ns_labels')
        # merge val_metrics_gt into val_dict
        ts_dict = evl.add_suffix_to_metric_dict(ts_metrics_ns, suffix=f'{wp}tsn',new_dict=ts_dict)
        
        
        # 3> create tr_dict for training results
        # loss
        tr_dict = {'epoch': epoch+1,
                   f'{wp}loss':epoch_loss/batch_in_epoch}
        if args.loss_type=='cd':
            tr_dict[f'{wp}loss_c'] = epoch_loss_c/batch_in_epoch
            tr_dict[f'{wp}loss_d'] = epoch_loss_d/batch_in_epoch
        # training acc
        if args.cal_tr:
            tr_dict = evl.epoch_log_dict(tr_dict, tr_accs, batch_in_epoch, 
                                         net.n_classes, suffix=f'{wp}tr')
            tr_dict = evl.epoch_log_dict(tr_dict, tr_gt_accs, batch_in_epoch, 
                                         net.n_classes, suffix=f'{wp}trg')
            # ensembling results
            tr_dict = evl.epoch_log_dict(tr_dict, tr_accs_mit, batch_in_epoch, 
                                         net.n_classes, suffix=f'{wp}mit_tr')
            tr_dict = evl.epoch_log_dict(tr_dict, tr_gt_accs_mit, batch_in_epoch, 
                                         net.n_classes, suffix=f'{wp}mit_trg')
        
            
        # 4> evaluate and test using self-ensembling models after each epoch
        # update self-ensembling models after each epoch
        # # a> self-ensembling model of each iteration
        # - validate on noisy labels
        val_metrics_mit = evl.evaluate(net_ema_it, val_loader, device, experiment='ns_labels')
        # reframe val_metrics to val_dict
        val_dict_mit = evl.add_suffix_to_metric_dict(val_metrics_mit, suffix=f'{wp}mit_v')
        # - validate on gt
        val_gt_metrics_mit = evl.evaluate(net_ema_it, val_loader, device, experiment='gt_labels')
        # merge val_metrics_gt into val_dict
        val_dict_mit = evl.add_suffix_to_metric_dict(val_gt_metrics_mit, suffix=f'{wp}mit_vg',new_dict=val_dict_mit)
        # - test on gt
        ts_metrics_mit = evl.evaluate(net_ema_it, test_loader, device, experiment='gt_labels')
        # reframe ts_metrics to ts_dict
        ts_dict_mit = evl.add_suffix_to_metric_dict(ts_metrics_mit, suffix=f'{wp}mit_ts')
        # - test on noisy labels
        ts_metrics_ns_mit = evl.evaluate(net_ema_it, test_loader, device, experiment='ns_labels')
        # merge val_metrics_gt into val_dict
        ts_dict_mit = evl.add_suffix_to_metric_dict(ts_metrics_ns_mit, suffix=f'{wp}mit_tsn',new_dict=ts_dict_mit) 
        # # b> self-ensembling model of each epoch
        if args.esb_ep:
            use.update_ema_variables(net, net_ema_ep, args.alpha_ep)
            # - validate on noisy labels
            val_metrics_mep = evl.evaluate(net_ema_ep, val_loader, device, experiment='ns_labels')
            # reframe val_metrics to val_dict
            val_dict_mep = evl.add_suffix_to_metric_dict(val_metrics_mep, suffix=f'{wp}mep_v')
            # - validate on gt
            val_gt_metrics_mep = evl.evaluate(net_ema_ep, val_loader, device, experiment='gt_labels')
            # merge val_metrics_gt into val_dict
            val_dict_mep = evl.add_suffix_to_metric_dict(val_gt_metrics_mep, suffix=f'{wp}mep_vg',new_dict=val_dict_mep)
            # - test on gt
            ts_metrics_mep = evl.evaluate(net_ema_ep, test_loader, device, experiment='gt_labels')
            # reframe ts_metrics to ts_dict
            ts_dict_mep = evl.add_suffix_to_metric_dict(ts_metrics_mep, suffix=f'{wp}mep_ts')
            # test on noisy labels
            ts_metrics_ns_mep = evl.evaluate(net_ema_ep, test_loader, device, experiment='ns_labels')
            # merge val_metrics_gt into val_dict
            ts_dict_mep = evl.add_suffix_to_metric_dict(ts_metrics_ns_mep, suffix=f'{wp}mep_tsn',new_dict=ts_dict_mep)
           
            # # c> test last training batch on self-ensembling model of each epoch
            # make predictions using ema_ep model
            with torch.no_grad():
                pred_logits_ema_ep = net_ema_ep(images).squeeze(axis=1)
                pred_mep_masks = (torch.sigmoid(pred_logits_ema_ep) > 0.5).float()
            # calculate training accs
            _, btr_dict_mep, _, btr_gt_dict_mep = \
                evl.evaluate_batch(pred_mep_masks, ons_masks, net.n_classes, 
                                   {}, device, suffix=f'{wp}mep_tr',
                                   batch_to_wandb=args.batch_to_wandb,
                                   gt_masks=gt_masks,
                                   trg_accs={})
        else:
            val_dict_mep, ts_dict_mep, btr_dict_mep, btr_gt_dict_mep = {}, {}, {}, {}
        
        
        # 5> check whether to start sample correction
        if correct:
            ns_dict = evl.add_suffix_to_metric_dict(ns_metrics, suffix='ns')
        else:
            ns_dict = {}
            # record mit_tr_iou and do early learning detection
            tr_iou_mit.append(tr_dict[f'{wp}mit_tr_iou'].item())
            # calculate numerical gradients and check trigger condition
            trip, ngs_dict, detect_eps = \
                eld.act_module(tr_iou_mit, ngs_dict, args.el_wsizes, detect_eps)
            # check whether to start sample correction in the next epoch
            if trip>0:
                # reload from trigger point
                # checkpoint file
                fmid = int(np.round(trip/args.save_inv)*args.save_inv)
                fckp_dep = args.checkpoints_dir + f'/checkpoint_mcr_{args.mcr}_epoch_{fmid}.pth'
                ckp_dep = torch.load(fckp_dep, map_location=torch.device('cpu'))
                # send models and optimizer back to cpu
                if device == torch.device('cuda'): 
                    net.to(torch.device('cpu'))
                    net_ema_it.to(torch.device('cpu'))
                    send_optim_tensor_device(optimizer,torch.device('cpu'))
                # reload student model
                smsd_dep = modify_module_state_dict_keys(ckp_dep['model_state_dict'])
                net.load_state_dict(smsd_dep)
                net.to(device)
                optimizer.load_state_dict(ckp_dep['optimizer_state_dict'])
                send_optim_tensor_device(optimizer,device)
                # reload teacher model
                tmsd_dep = modify_module_state_dict_keys(ckp_dep['model_state_dict_mit'])
                net_ema_it.load_state_dict(tmsd_dep)
                net_ema_it.to(device)
                # change correct variable to True
                correct = True
                n_back = epoch-fmid+1
                wp=''   # delete baseline prefix
                print(f"Correction starts! - After EPOCH {epoch+1} and resume from EPOCH {fmid}")
        

        # 6> Log train, validation, and test metrics to wandb
        wandb.log({'step':steps, **tr_dict, **val_dict, **ts_dict, **ns_dict,
                   **val_dict_mit, **val_dict_mep, 
                   **ts_dict_mit,  **ts_dict_mep, 
                   **btr_dict_mep, **btr_gt_dict_mep})
    
        
        # save checkpoints
        if (epoch+1)%args.save_inv==0:
            save_dict = {'epoch': epoch+1,
                         'model_state_dict': net.state_dict(),
                         'model_state_dict_mit': net_ema_it.state_dict(),
                         'optimizer_state_dict':optimizer.state_dict(),
                         'loss':epoch_loss/batch_in_epoch,}
            if not correct: save_dict['tr_iou_mit'] = np.array(tr_iou_mit)
            if args.esb_ep: save_dict['model_state_dict_mep'] = net_ema_ep.state_dict(),
            torch.save(save_dict, 
                       args.checkpoints_dir + '/checkpoint_mcr_{}_epoch_{}.pth'.format(args.mcr,epoch+1))
        
        
        # 7> end of training
        if epoch+1 >= args.epochs: break
    
    
    # Close wandb run 
    wandb.finish()
    
    t1 = datetime.datetime.now().replace(microsecond=0)
    print(f'Training is finished|Total spent time:{t1-t0}!' )
    
    
    
if __name__ == '__main__':
    main()
