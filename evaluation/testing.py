import torch
import numpy as np
import torch.nn.functional as F
from scipy import ndimage
import nibabel as nib
import os
import numpy as np

def calculate_segmentation_dice(pred, label, clss):
    """
    calculate the dice between prediction and ground truth
    input:
        pred: predicted mask
        label: groud truth
        clss: eg. [0, 1] for binary class
    """
    Ncls = len(clss)
    dices = np.zeros(Ncls)
    [depth, height, width] = pred.shape
    print("pred shape: {}, label shape: {}".format(pred.shape, label.shape))
    for idx, cls in enumerate(clss):
        # binary map
        pred_cls = np.zeros([depth, height, width])
        pred_cls[np.where(pred == cls)] = 1
        label_cls = np.zeros([depth, height, width])
        label_cls[np.where(label == cls)] = 1

        # cal the inter & conv
        s = pred_cls + label_cls
        inter = len(np.where(s >= 2)[0])
        conv = len(np.where(s >= 1)[0]) + inter
        try:
            dice = 2.0 * inter / conv
        except:
            print("conv is zeros when dice = 2.0 * inter / conv")
            dice = -1

        dices[idx] = dice

    return dices

def test_3d_cnn(data_loader, model, test_img_names, no_cuda, data_folder):
    output_masks = []
    model.eval()
    for batch_id, batch_data in enumerate(data_loader):
        volume = batch_data
        if not no_cuda:
            volume = volume.cuda()
        with torch.no_grad():
            logits = model(volume)
            probs = F.softmax(logits, dim=1)

        # resize mask to original size
        [batchsize, num_classes, mask_d, mask_h, mask_w] = probs.shape
        original_data = nib.load(os.path.join(data_folder, test_img_names[batch_id]))
        original_data = original_data.get_fdata() # numpy
        [depth, height, width] = original_data.shape
        output_mask = probs[0] # shape = [num_classes, mask_d, mask_h, mask_w]
        scale = [1, depth*1.0/mask_d, height*1.0/mask_h, width*1.0/mask_w]
        output_mask = output_mask.cpu().numpy() # output_masks is numpy
        output_mask = ndimage.zoom(output_mask, scale, order=1)
        output_mask = np.argmax(output_mask, axis=0) # shape = [depth, height, width] with voxel values from 0 to num_classes-1; still numpy
        
        output_masks.append(output_mask)
 
    return output_masks
