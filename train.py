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
from models.cycle_depth import CycleGANModel_depth
from models.cycle_cycle import CycleGANModel
from data.my_dataset import MyUnalignedDataset
import torch
from collections import OrderedDict 


def plot_img(img_dict, global_step, is_epoch=False, depth=True):

        img_t = img_dict['real_A'].cpu().detach()
        lab_t = img_dict['fake_B'].cpu().detach()
        img_t2t = img_dict['rec_A'].cpu().detach()
        img_t2t_d = img_dict['idt_B'].cpu().detach()

        
        img_s = img_dict['real_B'].cpu().detach()
        lab_s = img_dict['fake_A'].cpu().detach()
        img_s2t = img_dict['rec_B'].cpu().detach()
        img_s2t_d = img_dict['idt_A'].cpu().detach()

        
        n_col = 4
        n_row = 2
        fig, axes = plt.subplots(nrows = n_row, ncols = n_col, figsize=(30, 15))
        fig.subplots_adjust(hspace=0.0, wspace=0.01)
        
        for ax in axes.flatten():
            ax.axis('off')
            
        if depth:    
            pr = lambda img: np.clip((img[0].permute(1,2,0).numpy()+1)/2,0,1)
        else:
            pr = lambda img: np.clip((img[0].permute(1,2,0).numpy()+1)/2,0,1)

        axes[0,0].set_title('Syn RGB ')
        axes[0,1].set_title('Fake real RGB')
        axes[0,2].set_title('Reconstructed real')
        axes[0,3].set_title('Identity fake')

            
        axes[1,0].set_title('Real RGB')
        axes[1,1].set_title('Fake syn Depth')
        axes[1,2].set_title('Reconstructed Fake')
        axes[1,3].set_title('S-R Depth')

            
        axes[0,0].imshow(pr(img_t), cmap=plt.get_cmap('RdYlBu'))
        axes[0,1].imshow(pr(lab_t), cmap=plt.get_cmap('RdYlBu'))
        axes[0,2].imshow(pr(img_t2t), cmap=plt.get_cmap('RdYlBu'))
        axes[0,3].imshow(pr(img_t2t_d), cmap=plt.get_cmap('RdYlBu'))

            
        axes[1,0].imshow(pr(img_s), cmap=plt.get_cmap('RdYlBu'))
        axes[1,1].imshow(pr(lab_s), cmap=plt.get_cmap('RdYlBu'))
        axes[1,2].imshow(pr(img_s2t), cmap=plt.get_cmap('RdYlBu'))
        axes[1,3].imshow(pr(img_s2t_d), cmap=plt.get_cmap('RdYlBu'))
        
        wandb.log({"chart": fig}, step=global_step)
        plt.close(fig)   
        
def plot_part2(img_dict, global_step, is_epoch=False, depth=True):

        syn_image = img_dict['syn_image'].cpu().detach()
        syn_depth = img_dict['syn_depth'].cpu().detach()
        syn2real_depth = img_dict['syn2real_depth'].cpu().detach()
#         syn2real_image = img_dict['syn2real_image'].cpu().detach()
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

        axes[1,0].set_title('nothing')
        
        axes[1,1].set_title('real_image')
        axes[1,2].set_title('real_depth')
        axes[1,3].set_title('pred_real_depth')

            
        axes[0,0].imshow(pr(syn_image))
        axes[0,1].imshow(pr_d(syn_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,2].imshow(pr_d(syn2real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,3].imshow(pr_d(pred_syn_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)

            
        axes[1,0].imshow(pr(real_image*0)) # nothing
        
        axes[1,1].imshow(pr(real_image))
        axes[1,2].imshow(pr_d(real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[1,3].imshow(pr_d(pred_real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        
        wandb.log({"chart": fig}, step=global_step)
        plt.close(fig)   
        
def plot_part3(img_dict, global_step, is_epoch=False, depth=True):

        syn_image = img_dict['syn_image'].cpu().detach()
        syn_depth = img_dict['syn_depth'].cpu().detach()
        syn2real_depth = img_dict['syn2real_depth'].cpu().detach()
#         syn2real_image = img_dict['syn2real_image'].cpu().detach()
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

        axes[1,0].set_title('nothing')
        
        axes[1,1].set_title('real_image')
        axes[1,2].set_title('real_depth')
        axes[1,3].set_title('pred_real_depth')
        axes[1,3].set_title('real2syn')
            
        axes[0,0].imshow(pr(syn_image))
        axes[0,1].imshow(pr_d(syn_depth), cmap=plt.get_cmap('RdYlBu'))
        axes[0,2].imshow(pr_d(syn2real_depth), cmap=plt.get_cmap('RdYlBu'))
        axes[0,3].imshow(pr_d(pred_syn_depth), cmap=plt.get_cmap('RdYlBu'))
        axes[0,4].imshow(pr_d(sr2syn), cmap=plt.get_cmap('RdYlBu'))
            
        axes[1,0].imshow(pr(real_image*0)) # nothing
        
        axes[1,1].imshow(pr(real_image))
        axes[1,2].imshow(pr_d(real_depth), cmap=plt.get_cmap('RdYlBu'))
        axes[1,3].imshow(pr_d(pred_real_depth), cmap=plt.get_cmap('RdYlBu'))
        axes[1,4].imshow(pr_d(real2syn), cmap=plt.get_cmap('RdYlBu'))
        
        wandb.log({"chart": fig}, step=global_step)
        plt.close(fig)
        
def plot_part4(img_dict, global_step, is_epoch=False, depth=True, stage='train'):

        syn_image = img_dict['syn_image'].cpu().detach()
        syn_depth = img_dict['syn_depth'].cpu().detach()
        fake_B = img_dict['fake_B'].cpu().detach()
        rec_A = img_dict['rec_A'].cpu().detach()
        idt_B = img_dict['idt_B'].cpu().detach()
        
        
        real_image = img_dict['real_image'].cpu().detach()
        real_depth = img_dict['real_depth'].cpu().detach()
        fake_A = img_dict['fake_A'].cpu().detach()
        rec_B = img_dict['rec_B'].cpu().detach()
        idt_A = img_dict['idt_A'].cpu().detach()

        
        n_col = 5
        n_row = 2
        fig, axes = plt.subplots(nrows = n_row, ncols = n_col, figsize=(45, 25))
        fig.subplots_adjust(hspace=0.0, wspace=0.01)
        
        for ax in axes.flatten():
            ax.axis('off')
                      
        pr_d = lambda img: np.clip((img[0].permute(1,2,0).numpy()+1)/2,0,1)[:,:,0]
        pr = lambda img: np.clip((img[0].permute(1,2,0).numpy()+1)/2,0,1)

        axes[0,0].set_title('syn_image')
        axes[0,1].set_title('syn_depth')
        axes[0,2].set_title('fake_B')
        axes[0,3].set_title('rec_A')
        axes[0,4].set_title('idt_B')

        axes[1,0].set_title('real_image')
        axes[1,1].set_title('real_depth')
        axes[1,2].set_title('fake_A')
        axes[1,3].set_title('rec_B')
        axes[1,4].set_title('idt_A')

            
        axes[0,0].imshow(pr(syn_image))
        axes[0,1].imshow(pr_d(syn_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,2].imshow(pr_d(fake_B), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,3].imshow(pr_d(rec_A), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,4].imshow(pr_d(idt_B), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)

            
        axes[1,0].imshow(pr(real_image))
        axes[1,1].imshow(pr_d(real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[1,2].imshow(pr_d(fake_A), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[1,3].imshow(pr_d(rec_B), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[1,4].imshow(pr_d(idt_A), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        
#         wandb.log({f"{stage}": fig}, step=global_step)
        wandb.log({f"chart": fig}, step=global_step)
        plt.close(fig)           
        
def sum_of_dicts(dict1, dict2, l):
    output = OrderedDict([(key, dict1[key]+dict2[key]/l) for key in dict1.keys()])
    return output

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    wandb.config.update(opt)
    plot_function = plot_part4
#     dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt, MyUnalignedDataset) 
    test_dataset = create_dataset(opt, MyUnalignedDataset, stage='test')
    print(len(dataset), len(test_dataset))
    
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = CycleGANModel(opt)
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    total_iters = opt.start_iter                # the total number of training iterations
    test_iter=0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
#         visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        
    



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
#                 visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
#                 if opt.display_id > 0:
#                     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % 10000 == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

        model.eval()
        stage = 'test'
        
        with torch.no_grad():
            mean_losses = OrderedDict([('D_A', 0.0), ('G_A', 0.0), ('cycle_A', 0.0), ('idt_A', 0.0), ('D_B', 0.0), ('G_B', 0.0), ('cycle_B', 0.0), ('idt_B', 0.0)])
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
                mean_losses = sum_of_dicts(mean_losses, losses, l)  
            wandb.log({stage:mean_losses}, step = total_iters)       
            print(mean_losses)
            
            
            


            
        
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

        
        
        
        
