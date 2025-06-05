from setting import parse_opts 
from datasets.brains18 import BrainS18Dataset
from model import generate_model
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy import ndimage
import nibabel as nib
import sys
import os
from utils.file_process import load_lines
import numpy as np


def seg_eval(pred, label, clss):
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

def test(data_loader, model, test_img_names, settings):
    output_masks = []
    model.eval()
    for batch_id, batch_data in enumerate(data_loader):
        volume = batch_data
        if not settings.no_cuda:
            volume = volume.cuda()
        with torch.no_grad():
            logits = model(volume)
            probs = F.softmax(logits, dim=1)

        # resize mask to original size
        [batchsize, num_classes, mask_d, mask_h, mask_w] = probs.shape
        original_data = nib.load(os.path.join(settings.data_root, test_img_names[batch_id]))
        original_data = original_data.get_fdata() # numpy
        [depth, height, width] = original_data.shape
        output_mask = probs[0] # shape = [num_classes, mask_d, mask_h, mask_w]
        scale = [1, depth*1.0/mask_d, height*1.0/mask_h, width*1.0/mask_w]
        output_mask = output_mask.cpu().numpy()
        output_mask = ndimage.zoom(output_mask, scale, order=1)
        output_mask = np.argmax(output_mask, axis=0) # shape = [depth, height, width] with voxel values from 0 to num_classes-1
        
        output_masks.append(output_mask)
 
    return output_masks


if __name__ == '__main__':
    # get setttings
    settings = parse_opts()
    settings.target_type = "normal"
    settings.phase = 'test'
    if not settings.no_cuda:
        print("Using GPU for testing")
    else:
        print("Using CPU for testing")

    # getting model
    training_checkpoint_path = torch.load(settings.resume_path) # selected training checkpoint
    net, _ = generate_model(settings)
    net.load_state_dict(training_checkpoint_path['state_dict'])

    # data tensor
    testing_data = BrainS18Dataset(settings.data_root, settings.img_list, settings)
    data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    # testing
    test_img_names = [info.split(" ")[0] for info in load_lines(settings.img_list)]
    output_masks = test(data_loader, net, test_img_names, settings)
    
    # evaluation: calculate dice 
    label_names = [info.split(" ")[1] for info in load_lines(settings.img_list)]
    Nimg = len(label_names)
    dices = np.zeros([Nimg, settings.n_seg_classes])
    for idx in range(Nimg):
        label = nib.load(os.path.join(settings.data_root, label_names[idx]))
        label = label.get_fdata()
        dices[idx, :] = seg_eval(output_masks[idx], label, range(settings.n_seg_classes))
    
    # print result
    for idx in range(0, settings.n_seg_classes):
        mean_dice_per_task = np.mean(dices[:, idx])
        print('mean dice for class-{} is {}'.format(idx, mean_dice_per_task))   
