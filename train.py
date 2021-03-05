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
# from models.main_network_model import MainNetworkModel
# from models.main_network_best_model import MainNetworkBestModel
# from models.main_network_best_sr1_model import MainNetworkBestSR1Model
from models.main_network_best_sr2_model import MainNetworkBestSR2Model
# from models.translation_model import TranslationModel
# from models.I2D_model import I2DModel
# from models.depth_by_image import Depth_by_Image
# from data.my_dataset import MyUnalignedDataset
from data.my_up_dataset import MyUnalignedDataset
# from data.my_naive_sr_dataset import MyUnalignedDataset
# from data.my_I2D_dataset import MyUnalignedDataset
# from data.my_translation_dataset import MyUnalignedDataset
import torch
from collections import OrderedDict 
# from .metrics import call_it


def plot_main_new_norm(img_dict, global_step, depth=True,  is_epoch=False, stage='train'):

        syn_image = img_dict['syn_image'].cpu().detach()
        syn_depth = img_dict['syn_depth'].cpu().detach()
        syn2real_depth = img_dict['syn2real_depth'].cpu().detach()
        pred_syn_depth = img_dict['pred_syn_depth'].cpu().detach()
        mask_syn_add_holes = img_dict['mask_syn_add_holes'].cpu().detach()
        syn_mask = img_dict['syn_mask'].cpu().detach()
        syn2real_depth_masked = img_dict['syn2real_depth_masked'].cpu().detach()
        
        syn_norm = img_dict['norm_syn'].cpu().detach()
        norm_syn2real = img_dict['norm_syn2real'].cpu().detach()
        syn_norm_pred = img_dict['norm_syn_pred'].cpu().detach()
        syn_depth_by_image = img_dict['syn_depth_by_image'].cpu().detach()
        
        real_image = img_dict['real_image'].cpu().detach()
        real_depth = img_dict['real_depth'].cpu().detach()
        real_depth_by_image = img_dict['real_depth_by_image'].cpu().detach()
        pred_real_depth = img_dict['pred_real_depth'].cpu().detach()
        mask_real_add_holes = img_dict['mask_real_add_holes'].cpu().detach()
        real_mask = img_dict['real_mask'].cpu().detach()

        depth_masked  = img_dict['depth_masked'].cpu().detach()
        norm_real = img_dict['norm_real'].cpu().detach()
        norm_real_pred = img_dict['norm_real_pred'].cpu().detach()
#         norm_real_rec = img_dict['norm_real_rec'].cpu().detach()
        
        
        n_col = 5
        n_row = 4
        fig, axes = plt.subplots(nrows = n_row, ncols = n_col, figsize=(45, 30))
        fig.subplots_adjust(hspace=0.0, wspace=0.01)
        
        for ax in axes.flatten():
            ax.axis('off')
            
            
        pr_d = lambda img: np.clip((img[0].permute(1,2,0).numpy()+1)/2,0,1)[:,:,0]
        pr = lambda img: np.clip((img[0].permute(1,2,0).numpy()+1)/2,0,1)

        axes[0,0].set_title('syn_image')
        axes[0,1].set_title('syn_depth')
        axes[0,2].set_title('syn2real_depth')
        axes[0,3].set_title('syn2real_depth_masked')
        axes[0,4].set_title('depth_masked')
        
        axes[1,0].set_title('syn_mask')
        axes[1,1].set_title('norm_syn')
        axes[1,2].set_title('norm_syn2real')
        axes[1,3].set_title('syn_norm_pred') 
        axes[1,4].set_title('mask_syn_add_holes') 


        axes[2,0].set_title('real_mask')
        axes[2,1].set_title('real_depth')
        axes[2,2].set_title('real_depth_by_image')
        axes[2,3].set_title('depth_masked')
        axes[2,4].set_title('mask_real_add_holes')
        
        axes[3,0].set_title('real_depth_by_image')
        axes[3,1].set_title('norm_real')
        axes[3,2].set_title('norm_real_pred')
        axes[3,3].set_title('norm_real_rec') 
        axes[3,4].set_title('mask_real_add_holes')  

            
        axes[0,0].imshow(pr(syn_image))
        axes[0,1].imshow(pr_d(syn_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,2].imshow(pr_d(syn2real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
#         print(syn2real_depth_masked.shape)
        axes[0,3].imshow(pr_d(syn2real_depth_masked), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,4].imshow(pr_d(pred_syn_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        
        axes[1,0].imshow(pr_d(syn_mask), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[1,1].imshow(pr(syn_norm*100))
        axes[1,2].imshow(pr(norm_syn2real*100))
        axes[1,3].imshow(pr(syn_norm_pred*100))
        axes[1,4].imshow(pr_d(mask_syn_add_holes), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
            
        axes[2,0].imshow(pr(real_image))
        axes[2,1].imshow(pr_d(real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[2,2].imshow(pr_d(real_depth_by_image), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[2,3].imshow(pr_d(depth_masked), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[2,4].imshow(pr_d(pred_real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        
        axes[3,0].imshow(pr_d(real_mask), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[3,1].imshow(pr(norm_real*100))
        axes[3,2].imshow(pr(norm_real_pred*100))
        axes[3,3].imshow(pr(norm_real_pred*100))
        axes[3,4].imshow(pr_d(mask_real_add_holes), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        
        wandb.log({"chart": fig}, step=global_step)
        plt.close(fig)  


def plot_cycle(img_dict, global_step, is_epoch=False, depth=True, stage='train'):

        syn_image = img_dict['syn_image'].cpu().detach()
        syn_depth = img_dict['syn_depth'].cpu().detach()
        fake_B = img_dict['fake_B'].cpu().detach()
        rec_A = img_dict['rec_A'].cpu().detach()
        idt_B = img_dict['idt_B'].cpu().detach()
        
        norm_syn = img_dict['norm_syn'].cpu().detach()
        norm_fake_B = img_dict['norm_fake_B'].cpu().detach()
        norm_rec_A = img_dict['norm_rec_A'].cpu().detach()
        norm_idt_B = img_dict['norm_idt_B'].cpu().detach()
        
        
        real_image = img_dict['real_image'].cpu().detach()
        real_depth = img_dict['real_depth'].cpu().detach()
        fake_A = img_dict['fake_A'].cpu().detach()
        rec_B = img_dict['rec_B'].cpu().detach()
        idt_A = img_dict['idt_A'].cpu().detach()
        
        norm_real = img_dict['norm_real'].cpu().detach()
        norm_fake_A = img_dict['norm_fake_A'].cpu().detach()
        norm_rec_B = img_dict['norm_rec_B'].cpu().detach()
        norm_idt_A = img_dict['norm_idt_A'].cpu().detach()
        
        n_col = 5
        n_row = 4
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
        
        axes[1,0].set_title('nothing')
        axes[1,1].set_title('syn_depth')
        axes[1,2].set_title('fake_B')
        axes[1,3].set_title('rec_A')
        axes[1,4].set_title('idt_B')

        axes[2,0].set_title('real_image')
        axes[2,1].set_title('real_depth')
        axes[2,2].set_title('fake_A')
        axes[2,3].set_title('rec_B')
        axes[2,4].set_title('idt_A')
        
        
        axes[3,0].set_title('real_image')
        axes[3,1].set_title('real_depth')
        axes[3,2].set_title('fake_A')
        axes[3,3].set_title('rec_B')
        axes[3,4].set_title('idt_A')

            
        axes[0,0].imshow(pr(syn_image))
        axes[0,1].imshow(pr_d(syn_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,2].imshow(pr_d(fake_B), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,3].imshow(pr_d(rec_A), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,4].imshow(pr_d(idt_B), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        
        axes[1,0].imshow(pr(syn_image*0))
        axes[1,1].imshow(pr(norm_syn*1000))
        axes[1,2].imshow(pr(norm_fake_B*1000))
        axes[1,3].imshow(pr(norm_rec_A*1000))
        axes[1,4].imshow(pr(norm_idt_B*1000))
            
        axes[2,0].imshow(pr(real_image))
        axes[2,1].imshow(pr_d(real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[2,2].imshow(pr_d(fake_A), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[2,3].imshow(pr_d(rec_B), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[2,4].imshow(pr_d(idt_A), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        
        
        axes[3,0].imshow(pr(syn_image*0))
        axes[3,1].imshow(pr(norm_real*1000))
        axes[3,2].imshow(pr(norm_fake_A*1000))
        axes[3,3].imshow(pr(norm_rec_B*1000))
        axes[3,4].imshow(pr(norm_idt_A*1000))
        
#         wandb.log({f"{stage}": fig}, step=global_step)
        wandb.log({f"chart": fig}, step=global_step)
        plt.close(fig)           

        
        
def plot_I2D(img_dict, global_step, depth=True,  is_epoch=False, stage='train'):

        syn_image = img_dict['syn_image'].cpu().detach()
        syn_depth = img_dict['syn_depth'].cpu().detach()
        pred_syn_depth = img_dict['pred_syn_depth'].cpu().detach()
        
        
        real_image = img_dict['real_image'].cpu().detach()
        real_depth = img_dict['real_depth'].cpu().detach()
        pred_real_depth = img_dict['pred_real_depth'].cpu().detach()

        
        
        syn_norm = img_dict['norm_syn'].cpu().detach()
        syn_norm_pred = img_dict['norm_syn_pred'].cpu().detach()
        
        
        real_norm = img_dict['norm_real'].cpu().detach()
        real_norm_pred = img_dict['norm_real_pred'].cpu().detach()       
        
        
        n_col = 3
        n_row = 4
        fig, axes = plt.subplots(nrows = n_row, ncols = n_col, figsize=(45, 30))
        fig.subplots_adjust(hspace=0.0, wspace=0.01)
        
        for ax in axes.flatten():
            ax.axis('off')
            
            
        pr_d = lambda img: np.clip((img[0].permute(1,2,0).numpy()+1)/2,0,1)[:,:,0]
        pr = lambda img: np.clip((img[0].permute(1,2,0).numpy()+1)/2,0,1)

        axes[0,0].set_title('syn_image')
        axes[0,1].set_title('syn_depth')
        axes[0,2].set_title('pred_syn_depth')

        axes[1,0].set_title('nothing')
        axes[1,1].set_title('syn_norm')
        axes[1,2].set_title('syn_norm_pred')       
        
        axes[2,0].set_title('real_image')
        axes[2,1].set_title('real_depth')
        axes[2,2].set_title('pred_real_depth')
        
        
        axes[3,0].set_title('nothing')
        axes[3,1].set_title('real_norm')
        axes[3,2].set_title('real_norm_pred')

            
        axes[0,0].imshow(pr(syn_image))
        axes[0,1].imshow(pr_d(syn_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,2].imshow(pr_d(pred_syn_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
            
        axes[1,0].imshow(pr(syn_norm*0))
        axes[1,1].imshow(pr(syn_norm*1000))
        axes[1,2].imshow(pr(syn_norm_pred*1000))
        
        axes[2,0].imshow(pr(real_image))
        axes[2,1].imshow(pr_d(real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[2,2].imshow(pr_d(pred_real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)

        axes[3,0].imshow(pr(real_norm*0))
        axes[3,1].imshow(pr(real_norm*1000))
        axes[3,2].imshow(pr(real_norm_pred*1000))
        
        wandb.log({"chart": fig}, step=global_step)
        plt.close(fig)
        
        
def sum_of_dicts(dict1, dict2, l):
    
    output = OrderedDict([(key, dict1[key]+dict2[key]/l) for key in dict1.keys()])
    return output




import os
from glob import glob
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import imageio
from scipy import signal
import tqdm
from glob import glob
import multiprocessing
import functools
import albumentations as A

def fspecial_gauss(size, sigma):
    """
        Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def _ssim(img1, img2, L=1):
    """
        L = 1 for depth maps in [0, 1] range
    
        Return the Structural Similarity Map corresponding to input images img1 
        and img2 (images are assumed to be uint8)
        This function attempts to mimic precisely the functionality of ssim.m a 
        MATLAB provided by the author's of SSIM
        https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    return np.mean(((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)))
    

def _rmse(pred, target):
    return np.sqrt(np.mean(np.power(target - pred, 2)))

def _psnr(pred, target, max_value=1):
    mse = np.mean(np.power(target - pred, 2))
    if mse == 0:
        raise NotImplementedError('Same img')
    else:
        return 20. * np.log10(max_value) - 10 * np.log10(mse)

def _mae(pred, target):
    return np.mean(np.abs(target - pred))

def _rmse_h(pred, target, hole_map, render_hole_map):
    no_original_holes_mask = np.logical_not(hole_map) + render_hole_map
    if np.any(hole_map):
        diff2 = np.square(pred[~no_original_holes_mask] - target[~no_original_holes_mask])
        return np.sqrt(np.mean(diff2))
    else:
        return np.nan

def _rmse_d(pred, target, hole_map, render_hole_map):
    union_hole_map = hole_map+render_hole_map
    diff2 = np.square(pred[~union_hole_map]- target[~union_hole_map])
    return np.sqrt(np.mean(diff2))

def calc_rmse(pred, target, hole_map, render_hole_map):
    return _rmse(pred[~render_hole_map], target[~render_hole_map])

def calc_psnr(pred, target, hole_map, render_hole_map):
    return _psnr(pred[~render_hole_map]/5100, target[~render_hole_map]/5100) # TODO: pass args.max_depth

def calc_mae(pred, target, hole_map, render_hole_map):
#     print(pred[~render_hole_map].shape)
#     print(target[~render_hole_map].shape)
    return _mae(pred[~render_hole_map], target[~render_hole_map])

def calc_ssim(pred, target, hole_map, render_hole_map):
    return _ssim(pred/5100, target/5100) # TODO: pass args.max_depth

metric_by_name = {
    "mae": calc_mae, #_mae,
    "rmse": calc_rmse, #_rmse,
    "psnr": calc_psnr, #_psnr,
    "ssim": calc_ssim, #_ssim,
    "rmse_h": _rmse_h,
    "rmse_d": _rmse_d
}

def apply_transformer(transformations,  depth):

        res = A.Compose(transformations, p=1)(image=depth)
        return res

def calc_metrics(pred, target, hole_map,render_hole_map, metric_names):
    out = {}
    
    for metric_name in metric_names:
        metric_func = metric_by_name[metric_name]
        out[metric_name] = metric_func(pred, target, hole_map, render_hole_map)

    return out

def calc_metrics_for_path(path_args, metric_names):
    input_path, pred_path, target_path = path_args
    input_orig = imageio.imread(input_path).astype(np.float64)
    pred = imageio.imread(pred_path).astype(np.float64).clip(0, 5100) # TODO: pass args.max_depth
    target = imageio.imread(target_path).astype(np.float64).clip(0, 5100) # TODO: pass args.max_depth
    transform_list = []
    transform_list.append(A.Resize(height=480, width=640, interpolation=4, p=1))
    transformed = apply_transformer(transform_list, target)
    target = transformed['image']
    hole_map = input_orig < 50
    render_hole_map = target < 50
#     print(render_hole_map.shape, hole_map.shape)
    
    return calc_metrics(pred, target, hole_map,render_hole_map, metric_names)

def calculate_given_paths(input_dir_path, pred_dir_path, target_dir_path, metric_names, n_cpus):
    input_names = sorted(glob(os.path.join(input_dir_path,'*.png')))
    pred_names = sorted(glob(os.path.join(pred_dir_path,'*.png')))
    target_names = sorted(glob(os.path.join(target_dir_path,'*.png')))

    _calc_metrics_for_path = functools.partial(calc_metrics_for_path, metric_names=metric_names)
    paths = zip(input_names, pred_names, target_names)
    with multiprocessing.Pool(n_cpus) as p:
        res = list(tqdm.tqdm(p.imap(func=_calc_metrics_for_path, iterable=paths), total=len(input_names)))

    out = {}
    for metric_name in metric_names:
        out[metric_name] = np.asarray([x[metric_name] for x in res])
        out[metric_name] = np.mean(out[metric_name][~np.isnan(out[metric_name])])

    return out



def call_it():
    input_path = '/root/datasets/un_depth/Scannet_ssim/testA/full_size/depth'
    pred_path = '/root/callisto/depth_SR/test_pred'
    target_path= '/root/datasets/un_depth/Scannet_ssim/testB/full_size/depth'
    max_depth=5100
    n_cpus=10
    list_of_metrics = ["rmse", "mae", "rmse_h", "rmse_d", "psnr", "ssim"]
    out = calculate_given_paths(input_path, pred_path, target_path, list_of_metrics, n_cpus)
    return out 




if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    wandb.config.update(opt)
    plot_function = plot_main_new_norm
#     plot_function = plot_cycle
#     plot_function = plot_I2D
# 
    dataset = create_dataset(opt, MyUnalignedDataset) 
    test_dataset = create_dataset(opt, MyUnalignedDataset, stage='test')
    print(len(dataset), len(test_dataset))
    
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

#     model = TranslationModel(opt)
#     model = MainNetworkModel(opt)
#     model = MainNetworkBestModel(opt)
#     model = MainNetworkBestSR1Model(opt)
    model = MainNetworkBestSR2Model(opt)
#     model = I2DModel(opt)

    model.setup(opt)               # regular setup: load and print networks; create schedulers

    total_iters = opt.start_iter                # the total number of training iterations
    test_iter=0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
#         visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        
#         if total_iters==0:
#             print('metrics')
#             metrics = call_it()
#             wandb.log(metrics, step = total_iters)
#             print('metrics end')
    
    
         
      

        
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

            if (total_iters-opt.start_iter)  % 1000*opt.batch_size == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
                print('metrics')

      
                # update learning rates at the end of every epoch.

        
        
        model.eval()
        stage = 'test'
        
        with torch.no_grad():
#             mean_losses = OrderedDict([('D_A', 0.0), ('G_A', 0.0), ('cycle_A', 0.0), ('idt_A', 0.0), ('D_B', 0.0), ('G_B', 0.0), ('cycle_B', 0.0), ('idt_B', 0.0)])
#             mean_losses = OrderedDict([('G_A', 0.0), ('cycle_A', 0.0), ('idt_A', 0.0),  ('G_B', 0.0), ('cycle_B', 0.0), ('idt_B', 0.0)])
#             mean_losses = OrderedDict([('task_syn', 0.0), ('task_real', 0.0)])
            mean_losses = OrderedDict([('task_syn', 0.0), ('holes_syn', 0.0), ('task_real_by_depth', 0.0), ('holes_real', 0.0), ('syn_norms', 0.0) ])
            l = len(test_dataset)
            for i, data in enumerate(test_dataset):  # inner loop within one epoch
                test_iter += opt.batch_size
                model.set_input(data)
                model.calculate(stage = stage)
#                 if test_iter % 1 == 0:   
#                     model.compute_visuals()
#                     image_dict = model.get_current_visuals()
#                     depth = opt.input_nc == 1
#                     plot_function(image_dict, test_iter, depth=depth)
                    
                    
                losses = model.get_current_losses()

                mean_losses = sum_of_dicts(mean_losses, losses,  l/max(opt.batch_size//4, 1))
                
            wandb.log({stage:mean_losses}, step = total_iters)
#             print(mean_losses) 
#             metrics = call_it()
#             wandb.log(metrics, step = total_iters)
#             print(mean_losses)          
        
        
        
            
        
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate() 