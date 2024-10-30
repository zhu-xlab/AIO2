'''
baseline for the Germany dataset:
    training UNet models with noisy labels or ground truth labels without special considerations
Model definition with the segmentation-models-pytorch package
'''
import argparse
import datetime
import os, sys, json
import wandb
import numpy as np
from pathlib import Path

# appending self-defined package path
sys.path.append('./')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

# # dataset
from dataset.data_loading_Building_h5 import BuildingDataset
# # models
import segmentation_models_pytorch as smp
# # self-defined packages
from utils.dice_score import dice_loss
import utils.self_ensembling as use
import utils.evaluates as evl


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks (baseline)')
    # paths of input data
    parser.add_argument('--data_dir', type=str, help='path_to_data')
    parser.add_argument('--data_name', type=str, dest='dname', default='data_all_cities_gt.h5')
    parser.add_argument('--noisy_label_name', type=str, dest='nsname', default='data_all_cities_ns_rm_50.h5')
    parser.add_argument('--partition_name', type=str, dest='pname', default='partitions_v50_ts250.h5')
    parser.add_argument('--monte_carlo_run', type=int, dest='mcr', default=1, help='Number of Monte Carlo runs')
    # saving directory
    parser.add_argument('--save_dir', type=str, default='Results')  
    # model settings
    parser.add_argument('--model_type', dest='mt', type=str, default='xception',
                        help='Type of used models. It is mainly utilized to define the encoder part') 
    parser.add_argument('--n_channels', type=int, default=3,
                        help='Number of input channels.') 
    parser.add_argument('--n_classes', type=int, default=1,
                        help='Number of predefined classese.')
    parser.add_argument('--loss_type', type=str, choices=['c','d','cd'], default='cd',
                        help='Type of used segmentation loss function for training. Specifically, \'c\' represents CrossEntropy loss, \'d\' represents Dice loss, \'cd\' represents combined loss function of the two.') 
    parser.add_argument('--resume', type=str, default=False, 
                        help='Load model from a .pth file to continue training')
    parser.add_argument('--resume_ema', action='store_true',
                        help='Load ema models from a .pth file as well or not if resume=True')
    # data settings 
    parser.add_argument('--experiment', type=str, choices=['gt_labels','ns_labels'], default='gt_labels', 
                        help='Type of labels used for training')
    parser.add_argument('--batch_size', dest='bs', type=int, default=25, 
                        help='Batch size')
    parser.add_argument('--test_batch_size', dest='tbs', type=int, default=25, 
                        help='Batch size for test and validation sets')
    parser.add_argument('--num_workers', '-n', metavar='NW', type=int, default=0, 
                        help='Number of workers for dataloaders')    
    # training settings
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, 
                        help='Number of training epochs')
    parser.add_argument('--optimizer', dest='opt', type=str, default='adam', choices=['adam','sgd'],
                        help='Type of used optimizer.')
    parser.add_argument('--learning_rate', '-l', dest='lr', metavar='LR', type=float, default=0.001, 
                        help='Initial learning rate')
    parser.add_argument('--amp', action='store_false', 
                        help='Use mixed precision or not')
    parser.add_argument('--lr_adjust', dest='schd', type=bool, default=False, 
                        help='Adjust lr according to validation accuracies or not')
    parser.add_argument('--cal_tr_acc', dest='cal_tr', action='store_true', 
                        help='Whether to calculate training accuracies or not')
    parser.add_argument('--soft_filter_size', dest='sc_sfs', type=int, default=-1, 
                        help='Filter size for soft label generation from gt. If sc_sfs<=0, train with hard labels')
    # wandb settings
    parser.add_argument('--project_name', dest='pjn', type=str, default='aio2', 
                        help='Name of the wandb project to record statistics.')
    parser.add_argument('--entity_name', dest='entity', type=str, default='v', 
                        help='Name of the entity of wandb.')
    parser.add_argument('--wandb_mode', dest='wmode', type=str, choices=["online", "offline", "disabled"], default='offline',
                        help='Setting of wandb.init modes ("online", "offline" or "disabled")')
    # self-ensembling
    parser.add_argument('--ensembling', action='store_true', 
                        help='Whether using self-ensembling model to calculate statistics')
    parser.add_argument('--ensembling_epoch', dest='esb_ep', action='store_true', 
                        help='Whether using self-ensembling model after each epoch if ensembling=True')
    parser.add_argument('--alpha_it', type=float, default=0.999, 
                        help='Alpha used for ema model updating after each iteration')
    parser.add_argument('--alpha_ep', type=float, default=0.99, 
                        help='Alpha used for ema model updating after each epoch')# other settings
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


def send_optim_tensor_gpu(optimizer,device):
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
    
    # set unet_type according to model settings: unet_gt or unet_ns(_rm)
    args.unet_type = f'{args.mt}_{args.experiment[:2]}'  # e.g., unet_gt or unet_ns_rm
    if args.experiment == 'ns_labels':
        ndn_split = args.nsname[:-3].split('_')   # nsname = data_all_cities_ns_rm_50.h5
        ns_ind = ndn_split.index('ns')
        args.unet_type += f'_{ndn_split[ns_ind+1]}_{ndn_split[ns_ind+2]}'  # mt_ns_rm_50
    args.nsname = args.nsname.split('.')[0]+f'_{args.mcr}.'+args.nsname.split('.')[1]
    if args.sc_sfs>0: args.unet_type += '_soft'
    if args.opt=='sgd': args.unet_type += f'_{args.opt}'
    ea = int(np.ceil(np.log10(1/args.lr)))
    eb = int(args.lr*10**ea)
    args.job_type = f'baseline_lr{eb}e{ea}' # = 'baseline_ema' if args.ensembling else 'baseline'
    
    # saving dirs
    # parent dir
    pdir = os.path.join(args.save_dir, args.unet_type, args.job_type, f"mcr_{args.mcr}_seed_{args.seed}")
    Path(pdir).mkdir(parents=True, exist_ok=True)
    # checkpoint saving dir
    args.checkpoints_dir = os.path.join(pdir,"checkpoints")
    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    
    # statistic recording in txt files
    # record params into txt file
    with open(f'{pdir}/params.txt', 'w') as ft:
        json.dump(args.__dict__, ft)
    # record system outputs to txt file
    if args.print_to_log:
        sys.stdout = open(f'{pdir}/log.txt', 'w')
    
    # set wandb
    wandb.init(project=args.pjn,#'Noisy label with reconstruction loss',
               entity=args.entity,
               config=args,
               group=args.unet_type,
               job_type=args.job_type,
               name=f'lt_{args.loss_type}_{args.mcr}',
               dir=pdir,
               resume=False)
    
    # output the model type
    print(f"{args.mt} trained with {args.experiment} ({args.unet_type})\n")
    
    # get device information
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    ###### 2 - load data ######
    train_dataset = BuildingDataset(args.data_dir, args.dname, args.nsname, 
                                    args.pname, args.mcr, split='train', aug=True, only_load_gt=(args.experiment=='gt_labels'))
    val_dataset = BuildingDataset(args.data_dir, args.dname, args.nsname, 
                                  args.pname, args.mcr, split='val', aug=False, only_load_gt=(args.experiment=='gt_labels'))
    test_dataset = BuildingDataset(args.data_dir, args.dname, args.nsname, 
                                   args.pname, args.mcr, split='test', aug=False, only_load_gt=(args.experiment=='gt_labels'))

    # construct dataloaders
    train_loader = DataLoader(train_dataset,
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
    net = smp.Unet(encoder_name=args.mt,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                   encoder_weights=None,           # use `imagenet` pre-trained weights for encoder initialization
                   in_channels=args.n_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                   classes=args.n_classes,                      # model output channels (number of classes in your dataset)
                   )

    print(f"Construct UNet model with {args.mt} as backbone using smp package!\n")
        

    ###### 4 - set the loss and optimizer ######
    if args.opt=='adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-8)
        print('Adam is used as optimizer!')
    elif args.opt=='sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        print('SGD is used as optimizer!')
    else:
        raise ValueError("Please provide correct optimizer type (adam or sgd)!")
    if args.schd: scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    # # # # segmentation loss # # # #
    # Cross Entropy
    if 'c' in args.loss_type:
        if args.n_classes>1:
            criterion = nn.CrossEntropyLoss()
            print("CrossEntropy loss is used!")
        else:
            criterion = nn.BCEWithLogitsLoss()
            print("BinaryCrossEntropy loss is used!")
    # Dice
    if 'd' in args.loss_type:
        print("Dice loss is used!")

    
    ###### 5 - load pre-trained model if have any ######
    last_epoch = 0
    if args.resume:
        assert os.path.isfile(args.resume), 'Model loading path is invalid!'
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        msd = modify_module_state_dict_keys(checkpoint['model_state_dict'])
        net.load_state_dict(msd)
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            raise TypeError("The type of optimizer stored in ckpt file is not compitable to the used optimizer!")
        last_epoch = checkpoint['epoch']
        
        # send optimizer tensor to gpu
        if args.opt=='adam':
            send_optim_tensor_gpu(optimizer,device)
        
        print(f"Pre-trained model and optimizer have been loaded from '{args.resume}'")
    else:
        print('No pretrained weights, will start from scratch.')
        
    # sent model to device
    net.to(device=device)
    
    # Log the network weight histograms (optional)
    wandb.watch(net)
    
    # # # # self-ensembling # # # #
    # create self-ensembling models
    if args.ensembling:
        # EMA-MIT
        net_ema_it = use.create_ema_model(net, device, name='mit',
                                          resume=args.resume if args.resume_ema else False)
        # EMA-MEP
        if args.esb_ep:
            net_ema_ep = use.create_ema_model(net, device, name='mep',
                                              resume=args.resume if args.resume_ema else False)
    else:
        print('No self-ensembling will be included!')
               
    
    ###### 5 - train the network ######
    print('\nStart training...')
    net.train()
    if args.ensembling: 
        net_ema_it.train()
        if args.esb_ep: net_ema_ep.train()
    steps = 0
    for epoch in range(last_epoch, args.epochs):
        # place holders to record loss values during training
        epoch_loss = 0  # recording averaged (combined) loss for each epoch
        batch_in_epoch = 0 # recording number of batches in each epoch
        running_loss = 0 # recording sum of loss for showing temporily averaged loss values
        if args.loss_type=='cd':
            epoch_loss_c, epoch_loss_d = 0, 0
        # place holders for training accs
        if args.cal_tr:
            tr_accs = create_acc_dict()
            if args.experiment=='ns_labels':
                tr_gt_accs = create_acc_dict()
        if args.ensembling:
            tr_accs_mit = create_acc_dict()
            if args.experiment=='ns_labels':
                tr_gt_accs_mit = create_acc_dict()
                
        # - TRAINING in each batch
        for batch in train_loader:
            steps += 1
            # 1> load data
            images = batch['img']                
            gt_masks = torch.squeeze(batch['gt'],1)
            # decide training label type (exact or noisy ones)
            assert args.experiment in ['ns_labels','gt_labels'], 'Experiment label type is wrong!'
            if args.experiment == 'ns_labels':
                true_masks = torch.squeeze(batch['ns'],1)
            else:
                true_masks = gt_masks.clone()
            # check if number of input channels is compatible to model setting
            assert images.shape[1] == args.n_channels, \
                f'Network has been defined with {args.n_channels} input channels, ' \
                f'but loaded images have {images.shape[1]} channels. Please check that the images are loaded correctly.'
            # send data to device
            images = images.to(device=device, dtype=torch.float32)
            # true_masks = true_masks.to(device=device, dtype=torch.float32)
            gt_masks = gt_masks.to(device=device, dtype=torch.float32)
            
            # 2> forward
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred_logits = net(images).squeeze(axis=1) 
                if args.n_classes>1:
                    dice_pred = F.softmax(pred_logits, dim=1)  
                else:
                    dice_pred = torch.sigmoid(pred_logits)
                
                # calculate loss
                # generate soft labels around boundaries
                if args.sc_sfs>0:
                    true_masks = use.generate_soft_bounds(true_masks.float(), args.sc_sfs)
                # send true_masks to device
                true_masks = true_masks.to(device=device, dtype=torch.float32)    
                # - crossentropy
                loss_cc = criterion(pred_logits, true_masks) if 'c' in args.loss_type else torch.tensor(0.0)
                # - Dice
                if 'd' in args.loss_type:
                    if args.n_classes>1:
                        dice_true = F.one_hot(true_masks, args.n_classes).permute(0, 3, 1, 2).float()
                        loss_cd = dice_loss(dice_pred, dice_true, multiclass=True)
                    else:
                        loss_cd = dice_loss(dice_pred, true_masks, multiclass=False)
                else: 
                    loss_cd = torch.tensor(0.0)
                # - final combined loss
                loss = loss_cc + loss_cd
                
            # 3> backforward - optimization
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
                if args.experiment == 'ns_labels': 
                    tr_accs, btr_dict, tr_gt_accs, btr_gt_dict = \
                        evl.evaluate_batch(pred_masks, true_masks, args.n_classes, 
                                           tr_accs, device, suffix='tr',
                                           batch_to_wandb=args.batch_to_wandb,
                                           gt_masks=gt_masks,
                                           trg_accs=tr_gt_accs)
                else:
                    tr_accs, btr_dict = \
                        evl.evaluate_batch(pred_masks, true_masks, args.n_classes, 
                                           tr_accs, device, suffix='tr',
                                           batch_to_wandb=args.batch_to_wandb,
                                           gt_masks=None,
                                           trg_accs=None)
            
            # 5> self-ensembling after each iteration
            # update self-ensembling models after each iteration
            if args.ensembling:
                use.update_ema_variables(net, net_ema_it, args.alpha_it)
                
                # make predictions using ema model
                with torch.no_grad():
                    pred_logits_ema_it = net_ema_it(images).squeeze(axis=1)
                    pred_mit_masks = (torch.sigmoid(pred_logits_ema_it) > 0.5).float()
                
                if args.experiment == 'ns_labels':
                    tr_accs_mit, btr_dict_mit, tr_gt_accs_mit, btr_gt_dict_mit = \
                        evl.evaluate_batch(pred_mit_masks, true_masks, args.n_classes, 
                                           tr_accs_mit, device, suffix='mit_tr',
                                           batch_to_wandb=args.batch_to_wandb,
                                           gt_masks=gt_masks,
                                           trg_accs=tr_gt_accs_mit)
                else:
                    tr_accs_mit, btr_dict_mit = \
                        evl.evaluate_batch(pred_mit_masks, true_masks, args.n_classes, 
                                           tr_accs_mit, device, suffix='mit_tr',
                                           batch_to_wandb=args.batch_to_wandb,
                                           gt_masks=None,
                                           trg_accs=None)
                                   
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
        val_metrics = evl.evaluate(net, val_loader, device,
                                   experiment=args.experiment, n_classes=args.n_classes)
        # reframe val_metrics to val_dict
        val_dict = evl.add_suffix_to_metric_dict(val_metrics, suffix='v')
        # for training with noisy labels, also calculate validation results w.r.t. noisy labels
        if args.experiment=='ns_labels':
            val_metrics_gt = evl.evaluate(net, val_loader, device, 
                                          experiment='gt_labels', n_classes=args.n_classes)
            # merge val_metrics_gt into val_dict
            val_dict = evl.add_suffix_to_metric_dict(val_metrics_gt, suffix='vg',new_dict=val_dict)
        # adjust lr according to validation results 
        if args.schd: 
            if net.n_classes==1: 
                scheduler.step(val_metrics['iou']) 
            else:
                scheduler.step(val_metrics['miou']) 
        
        # 2> Test
        ts_metrics = evl.evaluate(net, test_loader, device, 
                                  experiment='gt_labels', n_classes=args.n_classes)
        # reframe ts_metrics to ts_dict
        ts_dict = evl.add_suffix_to_metric_dict(ts_metrics, suffix='ts')
        # for training with noisy labels, also calculate test results w.r.t. noisy labels
        if args.experiment=='ns_labels':
            ts_metrics_ns = evl.evaluate(net, test_loader, device, 
                                         experiment='ns_labels', n_classes=args.n_classes)
            # merge val_metrics_gt into val_dict
            ts_dict = evl.add_suffix_to_metric_dict(ts_metrics_ns, suffix='tsn',new_dict=ts_dict)
        
        
        # 3> create tr_dict for training results
        # loss
        tr_dict = {'epoch': epoch+1,
                   'loss':epoch_loss/batch_in_epoch}
        if args.loss_type=='cd':
            tr_dict['loss_c'] = epoch_loss_c/batch_in_epoch
            tr_dict['loss_d'] = epoch_loss_d/batch_in_epoch
        # training acc
        if args.cal_tr:
            tr_dict = evl.epoch_log_dict(tr_dict, tr_accs, batch_in_epoch, 
                                         args.n_classes, suffix='tr')
            if args.experiment=='ns_labels':
                tr_dict = evl.epoch_log_dict(tr_dict, tr_gt_accs, batch_in_epoch, 
                                             args.n_classes, suffix='trg')
            # ensembling results
            if args.ensembling:
                tr_dict = evl.epoch_log_dict(tr_dict, tr_accs_mit, batch_in_epoch, 
                                             args.n_classes, suffix='mit_tr')
                if args.experiment=='ns_labels':
                    tr_dict = evl.epoch_log_dict(tr_dict, tr_gt_accs_mit, batch_in_epoch, 
                                                 args.n_classes, suffix='mit_trg')
                
        # 4> evaluate and test using self-ensembling models after each epoch
        # update self-ensembling models after each epoch
        if args.ensembling:
            # # a> self-ensembling model of each iteration
            # - validation
            val_metrics_mit = evl.evaluate(net_ema_it, val_loader, device, 
                                           experiment=args.experiment, n_classes=args.n_classes)
            # reframe val_metrics to val_dict
            val_dict_mit = evl.add_suffix_to_metric_dict(val_metrics_mit, suffix='mit_v')
            # for training with noisy labels, also calculate validation results w.r.t. noisy labels
            if args.experiment=='ns_labels':
                val_gt_metrics_mit = evl.evaluate(net_ema_it, val_loader, device, 
                                                  experiment='gt_labels', n_classes=args.n_classes)
                # merge val_metrics_gt into val_dict
                val_dict_mit = evl.add_suffix_to_metric_dict(val_gt_metrics_mit, suffix='mit_vg',new_dict=val_dict_mit)
            # - test
            ts_metrics_mit = evl.evaluate(net_ema_it, test_loader, device, 
                                          experiment='gt_labels', n_classes=args.n_classes)
            # reframe ts_metrics to ts_dict
            ts_dict_mit = evl.add_suffix_to_metric_dict(ts_metrics_mit, suffix='mit_ts')
            # for training with noisy labels, also calculate test results w.r.t. noisy labels
            if args.experiment=='ns_labels':
                ts_metrics_ns_mit = evl.evaluate(net_ema_it, test_loader, device, 
                                                 experiment='ns_labels', n_classes=args.n_classes)
                # merge val_metrics_gt into val_dict
                ts_dict_mit = evl.add_suffix_to_metric_dict(ts_metrics_ns_mit, suffix='mit_tsn',new_dict=ts_dict_mit)
            
            # # b> self-ensembling model of each epoch
            if args.esb_ep:
                use.update_ema_variables(net, net_ema_ep, args.alpha_ep)
                # - validation
                val_metrics_mep = evl.evaluate(net_ema_ep, val_loader, device, 
                                               experiment=args.experiment, n_classes=args.n_classes)
                # reframe val_metrics to val_dict
                val_dict_mep = evl.add_suffix_to_metric_dict(val_metrics_mep, suffix='mep_v')
                # for training with noisy labels, also calculate validation results w.r.t. noisy labels
                if args.experiment=='ns_labels':
                    val_gt_metrics_mep = evl.evaluate(net_ema_ep, val_loader, device, 
                                                      experiment='gt_labels', n_classes=args.n_classes)
                    # merge val_metrics_gt into val_dict
                    val_dict_mep = evl.add_suffix_to_metric_dict(val_gt_metrics_mep, suffix='mep_vg',new_dict=val_dict_mep)
                # - test
                ts_metrics_mep = evl.evaluate(net_ema_ep, test_loader, device, 
                                              experiment='gt_labels', n_classes=args.n_classes)
                # reframe ts_metrics to ts_dict
                ts_dict_mep = evl.add_suffix_to_metric_dict(ts_metrics_mep, suffix='mep_ts')
                # for training with noisy labels, also calculate test results w.r.t. noisy labels
                if args.experiment=='ns_labels':
                    ts_metrics_ns_mep = evl.evaluate(net_ema_ep, test_loader, device,
                                                     experiment='ns_labels', n_classes=args.n_classes)
                    # merge val_metrics_gt into val_dict
                    ts_dict_mep = evl.add_suffix_to_metric_dict(ts_metrics_ns_mep, suffix='mep_tsn',new_dict=ts_dict_mep)
                
                # # c> test last training batch on self-ensembling model of each epoch
                # make predictions using ema_ep model
                with torch.no_grad():
                    pred_logits_ema_ep = net_ema_ep(images).squeeze(axis=1)
                # calculate training accs
                _, btr_dict_mep, _, btr_gt_dict_mep = \
                    evl.evaluate_batch(pred_logits_ema_ep, true_masks, args.n_classes, 
                                       {}, device, suffix='mep_tr',
                                       batch_to_wandb=args.batch_to_wandb,
                                       gt_masks=gt_masks if args.experiment == 'ns_labels' else None,
                                       trg_accs={} if args.experiment == 'ns_labels' else None)
            else:
                val_dict_mep, ts_dict_mep, btr_dict_mep, btr_gt_dict_mep = {}, {}, {}, {}
        else:
            val_dict_mit, ts_dict_mit, val_dict_mep, ts_dict_mep, btr_dict_mep, btr_gt_dict_mep = {}, {}, {}, {}, {}, {}
            
        # 5> Log train, validation, and test metrics to wandb
        wandb.log({'step':steps, **tr_dict,**val_dict,**ts_dict,
                   **val_dict_mit, **val_dict_mep, 
                   **ts_dict_mit,  **ts_dict_mep,
                   **btr_dict_mep, **btr_gt_dict_mep})
    
        
        # save checkpoints
        if (epoch+1)%args.save_inv==0 or epoch==0:
            save_dict = {'epoch': epoch+1,
                         'model_state_dict': net.state_dict(),
                         'optimizer_state_dict':optimizer.state_dict(),
                         'loss':epoch_loss/batch_in_epoch,}
            if args.ensembling: 
                save_dict['model_state_dict_mit'] = net_ema_it.state_dict()
                if args.esb_ep: save_dict['model_state_dict_mep'] = net_ema_ep.state_dict()
            torch.save(save_dict, 
                       args.checkpoints_dir + '/checkpoint_mcr_{}_epoch_{}.pth'.format(args.mcr,epoch+1))
        
    # Close wandb run 
    wandb.finish()
    
    t1 = datetime.datetime.now().replace(microsecond=0)
    print(f'Training is finished|Total spent time:{t1-t0}!' )
    
    
    
if __name__ == '__main__':
    main()
