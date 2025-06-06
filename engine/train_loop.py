import torch
import numpy as np
from torch import nn
import time
from utils.logger import log
from scipy import ndimage
import os

def train_3d_cnn(data_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, no_cuda):
    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    loss_seg = nn.CrossEntropyLoss(ignore_index=-1)
    if not no_cuda:
        loss_seg = loss_seg.cuda()
        
    train_time_sp = time.time()
    model.train()
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))
        
        scheduler.step()
        log.info('lr = {}'.format(scheduler.get_lr()))
        
        for batch_id, batch_data in enumerate(data_loader):
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes, label_masks = batch_data

            if not no_cuda: 
                volumes = volumes.cuda()

            optimizer.zero_grad()
            output_masks = model(volumes)
            # resize label
            [n, _, d, h, w] = output_masks.shape
            new_label_masks = np.zeros([n, d, h, w])
            for label_id in range(n):
                label_mask = label_masks[label_id]
                [ori_c, ori_d, ori_h, ori_w] = label_mask.shape 
                label_mask = np.reshape(label_mask, [ori_d, ori_h, ori_w])
                scale = [d*1.0/ori_d, h*1.0/ori_h, w*1.0/ori_w]
                label_mask = ndimage.interpolation.zoom(label_mask, scale, order=0)
                new_label_masks[label_id] = label_mask

            new_label_masks = torch.tensor(new_label_masks).to(torch.int64)
            if not no_cuda:
                new_label_masks = new_label_masks.cuda()

            # calculating loss
            loss_value_seg = loss_seg(output_masks, new_label_masks)
            loss = loss_value_seg
            loss.backward()                
            optimizer.step()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info('Batch: {}-{} ({}), loss = {:.3f}, loss_seg = {:.3f}, avg_batch_time = {:.3f}'\
                    .format(epoch, batch_id, batch_id_sp, loss.item(), loss_value_seg.item(), avg_batch_time))
          
            # save model
            if batch_id == 0 and batch_id_sp != 0 and batch_id_sp % save_interval == 0:
            #if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(save_folder, epoch, batch_id)
                model_save_dir = os.path.dirname(model_save_path)
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                
                log.info('Saving checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id)) 
                torch.save({
                            'ecpoch': epoch,
                            'batch_id': batch_id,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                            model_save_path)
                            
    print('Finished training')   

