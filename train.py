'''
Training code for MRBrainS18 datasets segmentation
Written by Whalechen
'''

from setting import parse_opts 
from datasets.brains18 import BrainS18Dataset 
from model import generate_model
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time
from utils.logger import log
from scipy import ndimage
import os

def train(data_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, settings):
    # settings
    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    loss_seg = nn.CrossEntropyLoss(ignore_index=-1)

    print("Current settings are:")
    print(settings)
    print("\n\n")     
    if not settings.no_cuda:
        loss_seg = loss_seg.cuda()
        
    model.train()
    train_time_sp = time.time()
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))
        
        scheduler.step()
        log.info('lr = {}'.format(scheduler.get_lr()))
        
        for batch_id, batch_data in enumerate(data_loader):
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes, label_masks = batch_data

            if not settings.no_cuda: 
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
            if not settings.no_cuda:
                new_label_masks = new_label_masks.cuda()

            # calculating loss
            loss_value_seg = loss_seg(output_masks, new_label_masks)
            loss = loss_value_seg
            loss.backward()                
            optimizer.step()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                    'Batch: {}-{} ({}), loss = {:.3f}, loss_seg = {:.3f}, avg_batch_time = {:.3f}'\
                    .format(epoch, batch_id, batch_id_sp, loss.item(), loss_value_seg.item(), avg_batch_time))
          
            if not settings.ci_test:
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
    if settings.ci_test:
        exit()


if __name__ == '__main__':
    # settting
    settings = parse_opts()   
    if settings.ci_test:
        settings.img_list = './toy_data/test_ci.txt' 
        settings.n_epochs = 1
        settings.no_cuda = True
        settings.data_root = './toy_data'
        settings.pretrain_path = ''
        settings.num_workers = 0
        settings.model_depth = 10
        settings.resnet_shortcut = 'A'
        settings.input_D = 14
        settings.input_H = 28
        settings.input_W = 28
       
     
    
    # getting model
    torch.manual_seed(settings.manual_seed)
    model, parameters = generate_model(settings) 
    print (model)
    # optimizer
    if settings.ci_test:
        params = [{'params': parameters, 'lr': settings.learning_rate}]
    else:
        params = [
                { 'params': parameters['base_parameters'], 'lr': settings.learning_rate }, 
                { 'params': parameters['new_parameters'], 'lr': settings.learning_rate*100 }
                ]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)   
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    # train from resume
    if settings.resume_path:
        if os.path.isfile(settings.resume_path):
            print("=> loading checkpoint '{}'".format(settings.resume_path))
            checkpoint = torch.load(settings.resume_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
              .format(settings.resume_path, checkpoint['epoch']))

    # getting data
    settings.phase = 'train'
    if settings.no_cuda:
        settings.pin_memory = False
    else:
        settings.pin_memory = True    
    training_dataset = BrainS18Dataset(settings.data_root, settings.img_list, settings)
    data_loader = DataLoader(training_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=settings.num_workers, pin_memory=settings.pin_memory)

    # training
    train(data_loader, model, optimizer, scheduler, total_epochs=settings.n_epochs, save_interval=settings.save_intervals, save_folder=settings.save_folder, settings=settings) 
