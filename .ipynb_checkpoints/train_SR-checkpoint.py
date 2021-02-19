"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import wandb
import numpy as np
import matplotlib.pyplot as plt
wandb.init(project="translation_compare")
from models.MLPFL_model import MLPFLModel
from data.sr_dataset import SRDataset
import torch
from collections import OrderedDict 



def plot_only_depth(img_dict, global_step, depth=True,  is_epoch=False, stage='train'):

        image = img_dict['image'].cpu().detach()
        hr_bibcubic_depth = img_dict['hr_bibcubic_depth'].cpu().detach()
#         pred_syn_depth = img_dict['lr_depth'].cpu().detach()
        hr_gt_depth = img_dict['hr_gt_depth'].cpu().detach()
        prediction = img_dict['prediction'].cpu().detach()
        edges_hr = img_dict['edges_hr'].cpu().detach()
        edges_pred = img_dict['edges_pred'].cpu().detach()
        edges_bibc = img_dict['edges_bibc'].cpu().detach()

        
        n_col = 2
        n_row = 4
        fig, axes = plt.subplots(nrows = n_row, ncols = n_col, figsize=(40, 60))
        fig.subplots_adjust(hspace=0.0, wspace=0.01)
        
        for ax in axes.flatten():
            ax.axis('off')
            
            
        pr_d = lambda img: np.clip((img[0].permute(1,2,0).numpy()+1)/2,0,1)[:,:,0]
        pr = lambda img: np.clip((img[0].permute(1,2,0).numpy()+1)/2,0,1)

        axes[0,0].set_title('image')
        axes[0,1].set_title('hr_gt_depth')


        axes[1,0].set_title('hr_bibcubic_depth')
        axes[1,1].set_title('prediction')
        
        axes[2,0].set_title('edges_hr')
        axes[2,1].set_title('edges_pred')
        
        
        axes[3,0].set_title('edges_bibc')
        axes[3,1].set_title('nothing')
        
            
        axes[0,0].imshow(pr(image))
        axes[0,1].imshow(pr_d(hr_gt_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
            
        axes[1,0].imshow(pr_d(hr_bibcubic_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[1,1].imshow(pr_d(prediction), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        
        
        axes[2,0].imshow(edges_hr[0].permute(1,2,0).numpy()[:,:,0]*100)
        axes[2,1].imshow(edges_pred[0].permute(1,2,0).numpy()[:,:,0]*100)
        
        
        axes[3,0].imshow(edges_bibc[0].permute(1,2,0).numpy()[:,:,0]*100)
        axes[3,1].imshow(edges_bibc[0].permute(1,2,0).numpy()[:,:,0]*0)
        
        wandb.log({"chart": fig}, step=global_step)
        plt.close(fig)

     
        
def sum_of_dicts(dict1, dict2, l):
    output = OrderedDict([(key, dict1[key]+dict2[key]/l) for key in dict1.keys()])
    return output

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    wandb.config.update(opt)
    plot_function = plot_only_depth
#     dataset = create_dataset(opt, SRDataset) 
#     test_dataset = create_dataset(opt, SRDataset, stage='test')
    
    dataset = torch.utils.data.DataLoader(
            SRDataset(opt, stage='train'),
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))
    
    test_dataset = torch.utils.data.DataLoader(
            SRDataset(opt, stage='test'),
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))
    
    print(len(dataset), len(test_dataset))
    
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = MLPFLModel(opt)
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    total_iters = opt.start_iter                # the total number of training iterations
    test_iter=0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        model._train()
        stage = 'train'
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters(total_iters, opt.update_ratio)   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                image_dict = model.get_current_visuals()
                depth = opt.input_nc == 1
                plot_function(image_dict, total_iters, depth=depth, stage=stage)
#                 visualizer.display_current_results(image_dict, epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                wandb.log(losses, step = total_iters)   


            if total_iters % 10000 == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)



        model.eval()
        stage = 'test'
        
        with torch.no_grad():
            mean_losses = OrderedDict([('L1', 0.0), ('edge', 0.0), ('ssim', 0.0)])
            l = len(test_dataset)
            for i, data in enumerate(test_dataset):  # inner loop within one epoch
                test_iter += opt.batch_size
                model.set_input(data)
                model.calculate()
                if test_iter % 4500 == 0:   
                    model.compute_visuals()
                    image_dict = model.get_current_visuals()
                    depth = opt.input_nc == 1
                    plot_function(image_dict, total_iters, depth=depth)
                    
                    
                losses = model.get_current_losses()
                mean_losses = sum_of_dicts(mean_losses, losses,  l/opt.batch_size*4)  
            wandb.log({stage:mean_losses}, step = total_iters)       
            print(mean_losses)               
            


            
        
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
#         model.update_learning_rate()                     # update learning rates at the end of every epoch.

        
        
        
        
