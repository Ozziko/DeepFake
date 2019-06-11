# -*- coding: utf-8 -*-
"""@author: oz.livneh@gmail.com

* All rights of this project and my code are reserved to me, Oz Livneh.
* Feel free to use - for personal use!
* Use at your own risk ;-)

<<<This script is for R&D, experiments, debugging!>>>
"""

#%% main parameters
#--------- general -------------
#debugging=True # executes the debugging (short) sections, prints results
debugging=False
torch_manual_seed=0 # integer or None for no seed; for torch reproducibility, as much as possible
#torch_manual_seed=None

#--------- data -------------
images_folder_path=r'D:\AI Data\DeepFake\ZioNLight Bibi.mp4 214x384 frames'

#random_transforms=True # soft data augmentation - color jitter (no random cropping or flipping - to keep all images aligned)
random_transforms=False
#max_dataset_length=100 # if positive: builds a dataset by sampling only max_dataset_length samples from all available data; requires user approval
max_dataset_length=0 # if non-positive: not restricting dataset length - using all available data
seed_for_dataset_downsampling=0 # integer or None for no seed; for sampling max_dataset_length samples from dataset

validation_ratio=0.3 # validation dataset ratio from total dataset length

#batch_size_int_or_ratio_float=1e-2 # if float: batch_size=round(batch_size_over_dataset_length*len(dataset_to_split))
batch_size_int_or_ratio_float=64 # if int: this is the batch size, should be 2**n
data_workers=0 # 0 means no multiprocessing in dataloaders
#data_workers='cpu cores' # sets data_workers=multiprocessing.cpu_count()

shuffle_dataset_indices_for_split=True # dataset indices for dataloaders are shuffled before splitting to train and validation indices
#shuffle_dataset_indices_for_split=False
dataset_shuffle_random_seed=0 # numpy seed for sampling the indices for the dataset, before splitting to train and val dataloaders
#dataset_shuffle_random_seed=None
dataloader_shuffle=True # samples are shuffled inside each dataloader, on each epoch
#dataloader_shuffle=False

#--------- net -------------
#net_architecture='simple auto-encoder'
net_architecture='experimental auto-encoder'

loss_name='MSE'

#--------- training -------------
train_model_else_load_weights=True
#train_model_else_load_weights=False # instead of training, loads a pre-trained model and uses it

epochs=3
learning_rate=1e-1
momentum=0.9

lr_scheduler_step_size=1
lr_scheduler_decay_factor=0.9

best_model_criterion='min val epoch MSE' # criterion for choosing best net weights during training as the final weights
return_to_best_weights_in_the_end=True # when training complets, loads weights of the best net, definied by best_model_criterion
#return_to_best_weights_in_the_end=False

training_progress_ratio_to_log_loss=0.25 # <=1, inter-epoch logging and reporting loss and metrics during training, period_in_batches_to_log_loss=round(training_progress_ratio_to_log_loss*dataset_samples_number['train']/batch_size)
#plot_realtime_stats_on_logging=True # incomplete implementation!
plot_realtime_stats_on_logging=False
#plot_realtime_stats_after_each_epoch=True
plot_realtime_stats_after_each_epoch=False
#plot_loss_in_log_scale=True
plot_loss_in_log_scale=False

#offer_mode_saving=True # offer model weights saving ui after training (only if train_model_else_load_weights=True)
offer_mode_saving=False
models_folder_path='D:\My Documents\Dropbox\Python\DatingAI\Data\Saved Models'

#%% initialization
import logging
logging.basicConfig(format='%(asctime)s %(funcName)s (%(levelname)s): %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger=logging.getLogger('data processing logger')
logger.setLevel(logging.INFO)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from time import time
import copy
import PIL
import multiprocessing
if data_workers=='cpu cores':
    data_workers=multiprocessing.cpu_count()

import torch
torch.manual_seed(torch_manual_seed)

import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

def plot_from_image_filenames_list(sample_indices_to_plot,image_filenames_list,
                                       images_folder_path,images_per_row=4):
    assert (isinstance(images_per_row,int) and images_per_row>0), 'images_per_row is invalid, must be a positive integer'
    plt.figure()
    columns_num=math.ceil(len(sample_indices_to_plot)/images_per_row)
    for i,sample_index in enumerate(sample_indices_to_plot):
        image_filename=image_filenames_list[sample_index]
        image_array=plt.imread(os.path.join(images_folder_path,image_filename))    
        
        plt.subplot(columns_num,images_per_row,i+1)
        plt.imshow(image_array)
        plt.title(image_filename)
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
    plt.show()

class images_torch_dataset(torch.utils.data.Dataset):
    def __init__(self,image_filenames,images_folder_path,transform_func=None):
        self.image_filenames=image_filenames
        self.images_folder_path=images_folder_path
        self.transform_func=transform_func

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self,idx):
        image_filename=self.image_filenames[idx]
        image_path=os.path.join(self.images_folder_path,image_filename)
        image_array=PIL.Image.open(image_path)
        
        if self.transform_func!=None:
            image_array=self.transform_func(image_array)
        
        sample={'image filename':image_filename,'image array':image_array}
        return sample

def plot_from_torch_dataset(sample_indices_to_plot,torch_dataset,
                            images_per_row=4,image_format='PIL->torch'):
    assert (isinstance(images_per_row,int) and images_per_row>0), 'images_per_row is invalid, must be a positive integer'
    plt.figure()
    columns_num=math.ceil(len(sample_indices_to_plot)/images_per_row)
    for i,sample_index in enumerate(sample_indices_to_plot):
        sample=torch_dataset[sample_index]
        image_filename=sample['image filename']
        image_array=sample['image array']
        if image_format=='np->torch': # to return from a torch format that reached from a np format, to a np for plotting. see # Helper function to show a batch from https://pytorch.org/tutorials/beginner/data_loading_tutorial
            image_array=image_array.transpose((1,2,0))
        elif image_format=='PIL->torch': # to return from a torch format that reached from a PIL format, to a np for plotting. see # Helper function to show a batch from https://pytorch.org/tutorials/beginner/data_loading_tutorial
            image_array=image_array.numpy().transpose((1,2,0))
        
        plt.subplot(columns_num,images_per_row,i+1)
        plt.imshow(image_array)
        plt.title(image_filename)
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
    plt.show()

def training_stats_plot(stats_dict,fig,loss_subplot,MSE_subplot,plot_loss_in_log_scale=False):
    running_stats_df=pd.DataFrame.from_dict(stats_dict['train']['running metrics'],orient='index')
    epoch_train_stats_df=pd.DataFrame.from_dict(stats_dict['train']['epoch metrics'],orient='index')
    epoch_val_stats_df=pd.DataFrame.from_dict(stats_dict['val']['epoch metrics'],orient='index')
    
    loss_subplot.clear() # clearing plot before plotting, to avoid over-plotting
    if len(running_stats_df)>0:
        loss_subplot.plot(running_stats_df['loss per sample'],'-x',label='running train')
    loss_subplot.plot(epoch_train_stats_df['loss per sample'],'k-o',label='epoch train')
    loss_subplot.plot(epoch_val_stats_df['loss per sample'],'r-o',label='epoch val')
    loss_subplot.set_ylabel('loss per sample')
    loss_subplot.set_xlabel('epoch')
    loss_subplot.grid()
    loss_subplot.legend(loc='best')
    if plot_loss_in_log_scale:
        loss_subplot.set_yscale('log')
        
    MSE_subplot.clear() # clearing plot before plotting, to avoid over-plotting
    if len(running_stats_df)>0:
        MSE_subplot.plot(running_stats_df['MSE']**0.5,'-x',label='running train')
    MSE_subplot.plot(epoch_train_stats_df['MSE']**0.5,'k-o',label='epoch train')
    MSE_subplot.plot(epoch_val_stats_df['MSE']**0.5,'r-o',label='epoch val')
    MSE_subplot.set_ylabel('sqrt(MSE)')
    MSE_subplot.set_xlabel('epoch')
    MSE_subplot.grid()
    MSE_subplot.legend(loc='best')
    if plot_loss_in_log_scale:
        MSE_subplot.set_yscale('log')
    fig.canvas.draw()

class remainder_time:
    def __init__(self,time_seconds):
        self.time_seconds=time_seconds
        self.hours=int(time_seconds/3600)
        self.remainder_minutes=int((time_seconds-self.hours*3600)/60)
        self.remainder_seconds=time_seconds-self.hours*3600-self.remainder_minutes*60

logger.info('script initialized')

#%% building a torch dataset of PIL images with torchvision transforms
"""torchvision.transforms accept PIL images, and not np images that are 
    created when using skimage as presented in 
    https://pytorch.org/tutorials/beginner/data_loading_tutorial

torchvision transforms: https://pytorch.org/docs/stable/torchvision/transforms.html
"""

if random_transforms:
    transform_func=torchvision.transforms.Compose([
    #            torchvision.transforms.Resize(400),
    #            torchvision.transforms.RandomCrop(390),
                torchvision.transforms.CenterCrop(200),
                torchvision.transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0,hue=0),
    #            torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ToTensor(),
                ])
else:
    transform_func=torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(200),
                torchvision.transforms.ToTensor(),
                ])
crop_approval=input('ATTENTION: using torchvision.transforms.CenterCrop(200), approve? y/[n] ')
if crop_approval!='y':
    raise RuntimeError('user did not approve torchvision.transforms.CenterCrop(200)!')
"""torchvision.transforms.ToTensor() Converts a PIL Image or numpy.ndarray (H x W x C) in the 
    range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the 
    range [0.0, 1.0] if the PIL Image belongs to one of the modes 
    (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has 
    dtype = np.uint8
In the other cases, tensors are returned without scaling
source: https://pytorch.org/docs/stable/torchvision/transforms.html
"""
image_filenames=os.listdir(images_folder_path)

if max_dataset_length>0:
    user_data_approval=input('ATTENTION: downsampling is chosen - building a dataset by sampling only max_dataset_length=%d samples from all available data! approve? y/[n] '%(round(max_dataset_length)))
    if user_data_approval!='y':
        raise RuntimeError('user did not approve dataset max_dataset_length sampling!')
    random.seed(seed_for_dataset_downsampling)
    image_filenames=random.sample(image_filenames,max_dataset_length)

images_dataset=images_torch_dataset(image_filenames,images_folder_path,transform_func=transform_func)
sample_size=images_dataset[0]['image array'].size()
sample_pixels_per_channel=sample_size[1]*sample_size[2]
sample_pixels_all_channels=sample_size[0]*sample_pixels_per_channel
logger.info('set a PyTorch dataset of length %.2e, input size (assuming it is constant): (%d,%d,%d)'%(
        len(image_filenames),sample_size[0],sample_size[1],sample_size[2]))
#%% (debugging) verifying dataset by plotting
samples_to_plot=20
#sampling_for_sample_verification='none' # plotting first samples_to_plot samples
sampling_for_sample_verification='random' # plotting randomly selected samples_to_plot samples, using seed_for_sample_verification seed
seed_for_sample_verification=0
images_per_row=4
# end of inputs ---------------------------------------------------------------
if sampling_for_sample_verification=='none':
    sample_indices_to_plot=range(samples_to_plot)
elif sampling_for_sample_verification=='random':
    random.seed(seed_for_sample_verification)
    sample_indices_to_plot=random.sample(range(len(image_filenames)),samples_to_plot)
else:
    raise RuntimeError('unsupported sampling_for_sample_verification input!')

if debugging:
    plot_from_image_filenames_list(sample_indices_to_plot,image_filenames,
                                   images_folder_path,images_per_row)
    plt.suptitle('plotting images directly from disk')
    
    plot_from_torch_dataset(sample_indices_to_plot,images_dataset,
                            images_per_row,image_format='PIL->torch')
    plt.suptitle('plotting images from PyTorch dataset')
#%% splitting to train and val datsets and dataloaders
dataset_to_split=images_dataset

if isinstance(batch_size_int_or_ratio_float,int):
    batch_size=batch_size_int_or_ratio_float
elif isinstance(batch_size_int_or_ratio_float,float):
    batch_size=round(batch_size_int_or_ratio_float*len(dataset_to_split))
else:
    raise RuntimeError('unsupported batch_size input!')
if batch_size<1:
    batch_size=1
    logger.warning('batch_size=round(batch_size_over_dataset_length*len(dataset_to_split))<1 so batch_size=1 was set')
if batch_size==1:
    user_batch_size=input('batch_size=1 should cause errors since batch_size>1 is generally assumed! enter a new batch size equal or larger than 1, or smaller than 1 to abort: ')
    if user_batch_size<1:
        raise RuntimeError('aborted by user batch size decision')
    else:
        batch_size=round(user_batch_size)

dataset_length=len(dataset_to_split)
dataset_indices=list(range(dataset_length))
split_index=int((1-validation_ratio)*dataset_length)
if shuffle_dataset_indices_for_split:
    np.random.seed(dataset_shuffle_random_seed)
    np.random.shuffle(dataset_indices)
train_indices=dataset_indices[:split_index]
val_indices=dataset_indices[split_index:]

# splitting the dataset to train and val
train_dataset=torch.utils.data.Subset(dataset_to_split,train_indices)
val_dataset=torch.utils.data.Subset(dataset_to_split,val_indices)

# creating the train and val dataloaders
train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,
                        num_workers=data_workers,shuffle=dataloader_shuffle)
val_dataloader=torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,
                        num_workers=data_workers,shuffle=dataloader_shuffle)

# structuring
dataset_indices={'train':train_indices,'val':val_indices}
datasets={'train':train_dataset,'val':val_dataset}
dataset_samples_number={'train':len(train_dataset),'val':len(val_dataset)}

dataloaders={'train':train_dataloader,'val':val_dataloader}
dataloader_batches_number={'train':len(train_dataloader),'val':len(val_dataloader)}

logger.info('dataset split to training and validation datasets and dataloaders with validation_ratio=%.1f, lengths: (train,val)=(%d,%d)'%(
        validation_ratio,dataset_samples_number['train'],dataset_samples_number['val']))

#%% (debugging) verifying dataloaders
images_per_row=4
# end of inputs ---------------------------------------------------------------

if debugging:
    if __name__=='__main__' or data_workers==0: # required in Windows for multi-processing
        samples_batches={}
        for phase in ['train','val']:
            samples_batch=next(iter(dataloaders[phase]))
            samples_batches.update({phase:samples_batch})
    else:
        raise RuntimeError('cannot use multiprocessing (data_workers>0 in dataloaders) in Windows when executed not as main!')
        
    columns_num=math.ceil(batch_size/images_per_row)
    for phase in ['train','val']:
        plt.figure()
        for i in range(batch_size):
            samples_batch=samples_batches[phase]
            image_array=samples_batch['image array'][i].numpy().transpose((1,2,0))
            image_filename=samples_batch['image filename'][i]
            
            plt.subplot(columns_num,images_per_row,i+1)
            plt.imshow(image_array)
            plt.title(image_filename)
            plt.xticks(ticks=[])
            plt.yticks(ticks=[])
        plt.suptitle('plotting a batch from the %s dataloader'%phase)

#%% setting the NN
if training_progress_ratio_to_log_loss>1:
    raise RuntimeError('invalid training_progress_ratio_to_log_loss=%.2f, must be <=1'%training_progress_ratio_to_log_loss)
period_in_batches_to_log_loss=round(training_progress_ratio_to_log_loss*dataset_samples_number['train']/batch_size) # logging only during training

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if plot_realtime_stats_on_logging or plot_realtime_stats_after_each_epoch:
    logger.warning('plotting from inside the net loop is not working, should be debugged...')

if net_architecture=='simple auto-encoder':
    # inspired by https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    class autoencoder(nn.Module):
        def __init__(self):
            super(autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3,16,20,stride=4,bias=False),
                nn.ReLU(True),
                nn.BatchNorm2d(16),
                
                nn.Conv2d(16,3,8,stride=2,bias=False),
                nn.ReLU(True),
                nn.BatchNorm2d(3),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(3,16,8,stride=2,bias=False),
                nn.ReLU(True),
                nn.BatchNorm2d(16),
                
                nn.ConvTranspose2d(16,3,20,stride=4,bias=False),
                nn.ReLU(True),
                nn.BatchNorm2d(3),
            )
    
        def forward(self,x):
            x=self.encoder(x)
            x=self.decoder(x)
            return x
    model=autoencoder()
elif net_architecture=='experimental auto-encoder':
    class autoencoder(nn.Module):
        def __init__(self):
            super(autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3,16,4,stride=2,bias=False),
                nn.ReLU(True),
                nn.BatchNorm2d(16),
                
                nn.Conv2d(16,8,4,stride=1,bias=False),
                nn.ReLU(True),
                nn.BatchNorm2d(8),
                
                nn.Conv2d(8,4,4,stride=2,bias=False),
                nn.ReLU(True),
                nn.BatchNorm2d(4),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(4,8,4,stride=2,bias=False),
                nn.ReLU(True),
                nn.BatchNorm2d(8),
                
                nn.ConvTranspose2d(8,16,4,stride=1,bias=False),
                nn.ReLU(True),
                nn.BatchNorm2d(16),
                
                nn.ConvTranspose2d(16,3,4,stride=2,bias=False),
                nn.ReLU(True),
                nn.BatchNorm2d(3),
            )
    
        def forward(self,x):
            x=self.encoder(x)
            x=self.decoder(x)
            return x
    model=autoencoder()
else:
    raise RuntimeError('untreated net_architecture!')

model=model.to(device)
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum)

if loss_name=='MSE':
    loss_fn=nn.MSELoss(reduction='mean').to(device)
else:
    raise RuntimeError('untreated loss_name input')
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,
    step_size=lr_scheduler_step_size,gamma=lr_scheduler_decay_factor)

#%% (debugging) verifying model outputs and loss
if debugging:
    if __name__=='__main__' or data_workers==0:
        batch=next(iter(dataloaders['train']))
        input_images=batch['image array']
        input_images=input_images.to(device)
        print('input shape:',input_images.shape)
        model.eval()
        output_images=model(input_images)
        print('output shape:',output_images.shape)
        print('nn.MSELoss(input_images,output_images):',
              nn.MSELoss(reduction='mean').to(device)(input_images,output_images))
        print('((input_images-output_images)**2).mean():',
              ((input_images-output_images)**2).mean())
    else:
        raise RuntimeError('cannot use multiprocessing (data_workers>0 in dataloaders) in Windows when executed not as main!')

#%% training the net
if train_model_else_load_weights and (__name__=='__main__' or data_workers==0):
    stats_dict={'train':{'epoch metrics':{},
                     'running metrics':{}}, # running = measuerd on samples only since the last log
                     'val':{'epoch metrics':{}}}
    
    total_batches=epochs*(dataloader_batches_number['train']+dataloader_batches_number['val'])
    
    pytorch_total_wts=sum(p.numel() for p in model.parameters())
    pytorch_trainable_wts=sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("started training '%s' net on %s, trainable/total weigths: %d/%d"%(
        net_architecture,device,pytorch_trainable_wts,pytorch_total_wts))
    tic=time()
    for epoch in range(epochs):
        for phase in ['train','val']:
            if phase == 'train':
                scheduler.step()
                model.train() # set model to training mode
            else:
                model.eval() # set model to evaluate mode
            
            epoch_loss=0.0 # must be a float
            epoch_squared_error=0.0
            samples_processed_since_last_log=0
            loss_since_last_log=0.0 # must be a float
            squared_error_since_last_log=0.0
            
            for i_batch,batch in enumerate(dataloaders[phase]):
                input_images=batch['image array'].to(device)
                
                optimizer.zero_grad() # zero the parameter gradients
                
                # forward
                with torch.set_grad_enabled(phase=='train'): # if phase=='train' it tracks tensor history for grad calc
                    output_images=model(input_images)
                    loss=loss_fn(output_images,input_images)
                    if torch.isnan(loss):
                        raise RuntimeError('reached NaN loss - aborting training!')
                    # backward + optimize if training
                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                
                # accumulating stats
                samples_number=len(input_images)
                samples_processed_since_last_log+=samples_number
                
                current_loss=loss.item()*samples_number*sample_pixels_all_channels # the loss is averaged across samples and pixels in each minibatch, so it is multiplied to return to a total
                epoch_loss+=current_loss
                loss_since_last_log+=current_loss
                
                with torch.set_grad_enabled(False):
                    batch_squared_error=((output_images-input_images)**2).sum()
                    batch_squared_error=batch_squared_error.item()
                epoch_squared_error+=batch_squared_error
                squared_error_since_last_log+=batch_squared_error
                
                if phase=='train' and i_batch%period_in_batches_to_log_loss==(period_in_batches_to_log_loss-1):
                    loss_since_last_log_per_sample=loss_since_last_log/samples_processed_since_last_log
                    MSE_since_last_log=squared_error_since_last_log/samples_processed_since_last_log
            
                    completed_batches=epoch*(dataloader_batches_number['train']+dataloader_batches_number['val'])+(i_batch+1)
                    completed_batches_progress=completed_batches/total_batches
                    passed_seconds=time()-tic
                    expected_seconds=passed_seconds/completed_batches_progress*(1-completed_batches_progress)
                    expected_remainder_time=remainder_time(expected_seconds)
                    
                    logger.info('(epoch %d/%d, batch %d/%d, %s) running loss per sample (since last log): %.3e, running sqrt(MSE) (since last log): sqrt(%.3e)=%.3e\n\tETA: %dh:%dm:%.0fs'%(
                                epoch+1,epochs,i_batch+1,dataloader_batches_number[phase],phase,
                                loss_since_last_log_per_sample,
                                MSE_since_last_log,MSE_since_last_log**0.5,
                                expected_remainder_time.hours,expected_remainder_time.remainder_minutes,expected_remainder_time.remainder_seconds))
                    
                    partial_epoch=epoch+completed_batches_progress
                    stats_dict[phase]['running metrics'].update({partial_epoch:
                        {'batch':i_batch+1,'loss per sample':loss_since_last_log_per_sample,
                         'MSE':MSE_since_last_log}})
                    
                    loss_since_last_log=0.0 # must be a float
                    squared_error_since_last_log=0.0
                    samples_processed_since_last_log=0
            
            # epoch stats
            epoch_loss_per_sample=epoch_loss/dataset_samples_number[phase]
            epoch_MSE=epoch_squared_error/dataset_samples_number[phase]
            
            stats_dict[phase]['epoch metrics'].update({epoch:
                        {'loss per sample':epoch_loss_per_sample,
                         'MSE':epoch_MSE}})
            if phase=='val':
                if best_model_criterion=='min val epoch MSE':
                    best_criterion_current_value=epoch_MSE
                    if epoch==0:
                        best_criterion_best_value=best_criterion_current_value
                        best_model_wts=copy.deepcopy(model.state_dict())
                        best_epoch=epoch
                    else:
                        if best_criterion_current_value<best_criterion_best_value:
                            best_criterion_best_value=best_criterion_current_value
                            best_model_wts=copy.deepcopy(model.state_dict())
                            best_epoch=epoch
                
                completed_epochs_progress=(epoch+1)/epochs
                passed_seconds=time()-tic
                expected_seconds=passed_seconds/completed_epochs_progress*(1-completed_epochs_progress)
                expected_remainder_time=remainder_time(expected_seconds)
                
                # not printing epoch stats for training, since in this phase they are being measured while the weights are being updated, unlike in validation where stats are measured with no update
                logger.info('(epoch %d, %s) epoch loss per sample: %.3e, epoch sqrt(MSE): sqrt(%.3e)=%.3e\n\tETA: %dh:%dm:%.0fs'%(
                                    epoch+1,phase,
                                    epoch_loss_per_sample,
                                    epoch_MSE,epoch_MSE**0.5,
                                    expected_remainder_time.hours,
                                    expected_remainder_time.remainder_minutes,
                                    expected_remainder_time.remainder_seconds))
                print('-'*10)
    toc=time()
    elapsed_sec=toc-tic

    logger.info('finished training %d epochs in %dm:%.1fs'%(
            epochs,elapsed_sec//60,elapsed_sec%60))
    if return_to_best_weights_in_the_end:
        model.load_state_dict(best_model_wts)
        logger.info("loaded weights of best model according to '%s' criterion: best value %.3f achieved in epoch %d"%(
                best_model_criterion,best_criterion_best_value,best_epoch+1))
    if not (plot_realtime_stats_on_logging or plot_realtime_stats_after_each_epoch):
        fig=plt.figure()
        plt.suptitle('model stats')
        loss_subplot=plt.subplot(1,2,1)
        MSE_subplot=plt.subplot(1,2,2)
    training_stats_plot(stats_dict,fig,loss_subplot,MSE_subplot)
else: # train_model_else_load_weights==False
    ui_model_name=input('model weights file name to load: ')
    model_weights_file_path=os.path.join(models_folder_path,ui_model_name)
    if not os.path.isfile(model_weights_file_path):
        raise RuntimeError('%model_weights_path does not exist!')
    model_weights=torch.load(model_weights_file_path)
    model.load_state_dict(model_weights)
    logger.info('model weights from %s were loaded'%model_weights_file_path)

#%% post-training model evaluation
"""the validation class_metrics_df measured here in the model evaluation must be identical to those measured during the
    last/best epoch, UNLIKE the training metrics - since the train phase metrics measured during training were being 
    measured while the weights were being updated in batches (!), not after the train phase epoch completed (which 
    would require another iteration on the train dataloader to measure metrics, as is done here without training)
"""
logger.info('started model evaluation')
model.eval() # set model to evaluate mode
for phase in ['train','val']:
    epoch_loss=0.0 # must be a float
    epoch_squared_error=0.0
    
    for i_batch,batch in enumerate(dataloaders[phase]):
        input_images=batch['image array'].to(device)
                    
        # forward
        with torch.set_grad_enabled(False): # if phase=='train' it tracks tensor history for grad calc
            output_images=model(input_images)
            loss=loss_fn(output_images,input_images)
        
        # accumulating stats
        samples_number=len(input_images)            
        current_loss=loss.item()*sample_pixels_all_channels # the loss is averaged across samples and pixels in each minibatch, so it is multiplied to return to a total
        epoch_loss+=current_loss
        
        with torch.set_grad_enabled(False):
            batch_squared_error=((output_images-input_images)**2).sum()
            batch_squared_error=batch_squared_error.item()
        epoch_squared_error+=batch_squared_error
    
    # epoch stats
    epoch_loss_per_sample=epoch_loss/dataset_samples_number[phase]
    epoch_MSE=epoch_squared_error/dataset_samples_number[phase]
    
    logger.info('(post-training, %s) loss per sample: %.3e, sqrt(MSE): sqrt(%.3e)=%.3e'%(
                    phase,epoch_loss_per_sample,epoch_MSE,epoch_MSE**0.5))
          
logger.info('completed model evaluation')

##%% inspecting auto-encoding by batches
#images_per_row=4
## end of inputs ---------------------------------------------------------------
#
#if __name__=='__main__' or data_workers==0: # required in Windows for multi-processing
#    samples_batches={}
#    for phase in ['train','val']:
#        samples_batch=next(iter(dataloaders[phase]))
#        samples_batches.update({phase:samples_batch})
#else:
#    raise RuntimeError('cannot use multiprocessing (data_workers>0 in dataloaders) in Windows when executed not as main!')
#
#model.eval() # set model to evaluate mode
#for phase in ['train','val']:
#    samples_batch=samples_batches[phase]
#    input_images=samples_batch['image array'].to(device)
#    with torch.set_grad_enabled(False):
#        output_images=model(input_images)
#    
#    # see https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
#    input_images_grid=np.transpose(vutils.make_grid(
#            input_images,nrow=images_per_row,padding=5,normalize=True).cpu(),(1,2,0))
#    output_images_grid=np.transpose(vutils.make_grid(
#            output_images,nrow=images_per_row,padding=5,normalize=True).cpu(),(1,2,0))
#    
#    plt.figure()
#    plt.subplot(1,2,1)
#    plt.axis('off')
#    plt.title('original images')
#    plt.imshow(input_images_grid)
#    
#    plt.subplot(1,2,2)
#    plt.axis('off')
#    plt.title('reconstructed images')
#    plt.imshow(output_images_grid)
#
#    plt.suptitle('%s batch'%phase)

#%% inspecting auto-encoding by plotting (building the batches from samples)
samples_to_plot=20
#sampling_for_sample_verification='none' # plotting first samples_to_plot samples from datasets (train, val)
sampling_for_sample_verification='random' # plotting randomly selected samples_to_plot samples from datasets (train, val), using seed_for_sample_verification seed
seed_for_sample_verification=0
images_per_row=4
# end of inputs ---------------------------------------------------------------

# concatenating the samples to inspect into batches
image_batches={}
for phase in ['train','val']:
    if sampling_for_sample_verification=='none':
        sample_indices_to_plot=range(samples_to_plot)
    elif sampling_for_sample_verification=='random':
        random.seed(seed_for_sample_verification)
        sample_indices_to_plot=random.sample(range(len(datasets[phase])),samples_to_plot)
    else:
        raise RuntimeError('unsupported sampling_for_sample_verification input!')
    
    image_tensors_list=[]
    for i,i_sample in enumerate(sample_indices_to_plot):
        image_array=datasets[phase][i_sample]['image array'].unsqueeze(0)
        image_tensors_list.append(image_array)
    image_batches.update({phase:torch.cat(image_tensors_list,0)})

# applying the model, plotting results
model.eval() # set model to evaluate mode
for phase in ['train','val']:
    input_images=image_batches[phase].to(device)
    with torch.set_grad_enabled(False):
        output_images=model(input_images)
    
    # see https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    input_images_grid=np.transpose(vutils.make_grid(
            input_images,nrow=images_per_row,padding=5,scale_each=True,normalize=True).cpu(),(1,2,0))
    output_images_grid=np.transpose(vutils.make_grid(
            output_images,nrow=images_per_row,padding=5,scale_each=True,normalize=True).cpu(),(1,2,0))
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.title('original images')
    plt.imshow(input_images_grid)
    
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.title('reconstructed images')
    plt.imshow(output_images_grid)

    plt.suptitle('%s batch'%phase)

#%% saving the model
if offer_mode_saving and train_model_else_load_weights:
    try: os.mkdir(models_folder_path)
    except FileExistsError: pass # if the folder exists already - do nothing
    
    saving_decision=input('save model weights? [y]/n ')
    if saving_decision!='n':
        ui_model_name=input('name model weights file: ')
        model_weights_file_path=os.path.join(models_folder_path,ui_model_name+'.ptweights')
        if os.path.isfile(model_weights_file_path):
            alternative_filename=input('%s already exists, give a different file name to save, the same file name to over-write, or hit enter to abort: '%model_weights_file_path)
            if alternative_filename=='':
                raise RuntimeError('aborted by user')
            else:
                model_weights_file_path=os.path.join(models_folder_path,alternative_filename+'.ptweights')
        torch.save(model.state_dict(),model_weights_file_path)       
        logger.info('%s saved'%model_weights_file_path)
#%%
logger.info('script completed')
