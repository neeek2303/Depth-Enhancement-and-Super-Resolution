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
import imageio
wandb.init(project="translation_compare")
from models.cycle_depth import CycleGANModel_depth
from data.my_dataset import MyUnalignedDataset




        
def plot_part2(img_dict, global_step, is_epoch=False, depth=True, name='_'):

        syn_image = img_dict['syn_image'].cpu().detach()
        syn_depth = img_dict['syn_depth'].cpu().detach()
        syn2real_depth = img_dict['syn2real_depth'].cpu().detach()
        mask = img_dict['mask'].cpu().detach()
        pred_syn_depth = img_dict['pred_syn_depth'].cpu().detach()
        
        
        real_image = img_dict['real_image'].cpu().detach()
        real_depth = img_dict['real_depth'].cpu().detach()
        pred_real_depth = img_dict['pred_real_depth'].cpu().detach()

        
        n_col = 4
        n_row = 2
        fig, axes = plt.subplots(nrows = n_row, ncols = n_col, figsize=(45, 25))
        fig.subplots_adjust(hspace=0.0, wspace=0.01)
        
        for ax in axes.flatten():
            ax.axis('off')
            
            
        pr_d = lambda img: np.clip((img[0].permute(1,2,0).numpy()+1)/2,0,1)[:,:,0]
        pr = lambda img: np.clip((img[0].permute(1,2,0).numpy()+1)/2,0,1)

        axes[0,0].set_title('syn_image')
        axes[0,1].set_title('syn_depth')
        axes[0,2].set_title('syn2real_depth')
        axes[0,3].set_title('pred_syn_depth')

        axes[1,0].set_title('mask')
        
        axes[1,1].set_title('real_image')
        axes[1,2].set_title('real_depth')
        axes[1,3].set_title('pred_real_depth')

            
        axes[0,0].imshow(pr(syn_image))
        axes[0,1].imshow(pr_d(syn_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,2].imshow(pr_d(syn2real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,3].imshow(pr_d(pred_syn_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)

            
        axes[1,0].imshow(pr_d(mask) , cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[1,1].imshow(pr(real_image))
        axes[1,2].imshow(pr_d(real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[1,3].imshow(pr_d(pred_real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        
        
        imageio.imwrite(f'/mnt/hdd/un_depth/results/nikita/scheme_8_20/{name}.png' , (np.clip((pred_real_depth[0].permute(1,2,0).numpy()+1)/2,0,1)[:,:,0]*8000).astype(np.uint16))
        
        wandb.log({"chart": fig}, step=global_step)
        plt.close(fig)   
   
        
def plot_part3(img_dict, global_step, is_epoch=False, depth=True):

        syn_image = img_dict['syn_image'].cpu().detach()
        syn_depth = img_dict['syn_depth'].cpu().detach()
        syn2real_depth = img_dict['syn2real_depth'].cpu().detach()
        mask = img_dict['mask'].cpu().detach()
        pred_syn_depth = img_dict['pred_syn_depth'].cpu().detach()
        
        
        real_image = img_dict['real_image'].cpu().detach()
        real_depth = img_dict['real_depth'].cpu().detach()
        pred_real_depth = img_dict['pred_real_depth'].cpu().detach()
        
        sr2syn = img_dict['sr2syn_depth'].cpu().detach()
        real2syn = img_dict['real2syn_depth'].cpu().detach()
        
        n_col = 5
        n_row = 2
        fig, axes = plt.subplots(nrows = n_row, ncols = n_col, figsize=(45, 30))
        fig.subplots_adjust(hspace=0.0, wspace=0.01)
        
        for ax in axes.flatten():
            ax.axis('off')
            
            
        pr_d = lambda img: np.clip((img[0].permute(1,2,0).numpy()+1)/2,0,1)[:,:,0]
        pr = lambda img: np.clip((img[0].permute(1,2,0).numpy()+1)/2,0,1)

        axes[0,0].set_title('syn_image')
        axes[0,1].set_title('syn_depth')
        axes[0,2].set_title('syn2real_depth')
        axes[0,3].set_title('pred_syn_depth')
        axes[0,4].set_title('sr2syn')

        axes[1,0].set_title('mask')
        
        axes[1,1].set_title('real_image')
        axes[1,2].set_title('real_depth')
        axes[1,3].set_title('pred_real_depth')
        axes[1,3].set_title('real2syn')
            
        axes[0,0].imshow(pr(syn_image))
        axes[0,1].imshow(pr_d(syn_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,2].imshow(pr_d(syn2real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,3].imshow(pr_d(pred_syn_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,4].imshow(pr_d(sr2syn), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
            
        axes[1,0].imshow(pr_d(mask) , cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        
        axes[1,1].imshow(pr(real_image))
        axes[1,2].imshow(pr_d(real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[1,3].imshow(pr_d(pred_real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[1,4].imshow(pr_d(real2syn), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        
        wandb.log({"chart": fig}, step=global_step)
        plt.close(fig)      


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    wandb.config.update(opt)
    plot_function = plot_part2
#     dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt, MyUnalignedDataset) 
    
    
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

#     model = create_model(opt)      # create a model given opt.model and other options
    model = CycleGANModel_depth(opt)
    
    model.setup(opt)               # regular setup: load and print networks; create schedulers
#     visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for i, data in enumerate(dataset):  # inner loop within one epoch
        iter_start_time = time.time()  # timer for computation per iteration
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        total_iters += opt.batch_size
        if (i+1)%10==0:
            print(f'{i+1} samples tested')
        if opt.eval:
             model.eval()

        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()

        model.compute_visuals()
        image_dict = model.get_current_visuals()
        depth = opt.input_nc == 1
        plot_function(image_dict, total_iters, depth=depth, name = i)

        losses = model.get_current_losses()
        wandb.log(losses, step = total_iters)   

#             if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
#                 print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
#                 save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
#                 model.save_networks(save_suffix)

#             iter_data_time = time.time()
#         if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
#             print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
#             model.save_networks('latest')
#             model.save_networks(epoch)

#         print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
#         model.update_learning_rate()                     # update learning rates at the end of every epoch.

        
        
        
        
