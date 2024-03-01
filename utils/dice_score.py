import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)





def IoU(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of IoU for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = inter

        return (inter + epsilon) / (sets_sum -inter + epsilon)
    else:
        # compute and average metric for each batch element
        iou = 0
        for i in range(input.shape[0]):
            iou += IoU(input[i, ...], target[i, ...])
        return iou / input.shape[0]    
        
        
def mIoU(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # mIoU
    assert input.size() == target.size()
    miou = 0
    for channel in range(input.shape[1]):
        miou += IoU(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return miou / input.shape[1]


def IoUs(input: Tensor, target: Tensor, device, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    ious = torch.zeros([input.shape[1]+1], device=device, dtype=torch.float32)
    for channel in range(input.shape[1]):
        ious[channel] = IoU(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
    
    # miou
    ious[-1] = ious[:-1].sum()/ input.shape[1]
    
    return ious




