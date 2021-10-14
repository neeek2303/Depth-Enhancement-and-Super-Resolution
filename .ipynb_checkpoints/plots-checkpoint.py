import numpy as np
import matplotlib.pyplot as plt
import torch
from util.util import data_to_meters

def get_normals(depth):
    norm = np.zeros(( depth.shape[0], depth.shape[1], depth.shape[2], 3))
    dzdx = np.gradient(depth, 1, axis=1)
    dzdy = np.gradient(depth, 1, axis=2)
    norm[:, :, :, 0] = -dzdx
    norm[:, :, :, 1] = -dzdy
    norm[:, :, :, 2] = np.ones_like(depth)
    n = np.linalg.norm(norm, axis = 3, ord=2, keepdims=True)
    norm = norm/(n + 1e-15)
    return norm

def plot_main(wandb, img_dict, global_step, depth=True,  is_epoch=False, stage='train'):

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
        axes[0,3].imshow(pr_d(syn2real_depth_masked), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,4].imshow(pr_d(pred_syn_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        
        axes[1,0].imshow(pr_d(syn_mask), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1) 
        axes[1,1].imshow(pr(syn_norm))
        axes[1,2].imshow(pr(norm_syn2real))
        axes[1,3].imshow(pr(syn_norm_pred))
        axes[1,4].imshow(pr_d(mask_syn_add_holes), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
            
        axes[2,0].imshow(pr(real_image))
        axes[2,1].imshow(pr_d(real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[2,2].imshow(pr_d(real_depth_by_image), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[2,3].imshow(pr_d(depth_masked), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[2,4].imshow(pr_d(pred_real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        
        axes[3,0].imshow(pr_d(real_mask), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[3,1].imshow(pr(norm_real))
        axes[3,2].imshow(pr(norm_real))
        axes[3,3].imshow(pr(norm_real_pred))
        axes[3,4].imshow(pr_d(mask_real_add_holes), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        
        wandb.log({"chart": fig}, step=global_step)
        plt.close(fig)  


        
        
def plot_I2D(wandb, img_dict, global_step, depth=True,  is_epoch=False, stage='train'):

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
        
        
        
def tensor2im(input, isDepth = True):
    """"Converts a Tensor array into a numpy image array in meters.
    Parameters:
        input_image (tensor) --  the input image tensor array
    """
    if not isinstance(input, np.ndarray):
        if isinstance(input, torch.Tensor):  # get the data from a variable
            tensor = input.data
        else:
            return input
        if isDepth:
            tensor = data_to_meters(tensor, 5100)
            numpy = tensor.cpu().permute(0,2,3,1).numpy()[:,:,:,0]
        else:
            tensor = tensor * 127.5 + 127.5
            numpy = tensor.cpu().permute(0,2,3,1).numpy().astype(np.uint8)
    else:  # if it is a numpy array, do nothing
        numpy = input
    return numpy

def plot_translation(wandb, img_dict, global_step, use_cycle_A=False, depth=True,  is_epoch=False, stage='train'):
    A_imgs = tensor2im(img_dict['real_img_A'], isDepth=False)
    A_depth = tensor2im(img_dict['real_depth_A'], isDepth=True)
    A_norm = get_normals(A_depth * 1000)
    B_depth_fake = tensor2im(img_dict['fake_depth_B'], isDepth=True)
    B_norm_fake = get_normals(B_depth_fake * 1000)

    if use_cycle_A:
        A_depth_rec = tensor2im(img_dict['rec_depth_A'], isDepth=True)
        A_norm_rec = get_normals(A_depth_rec * 1000)
    else:
        A_depth_rec = np.zeros_like(B_depth_fake)
        A_norm_rec = np.zeros_like(B_norm_fake)

    B_imgs = tensor2im(img_dict['real_img_B'], isDepth=False)
    B_depth = tensor2im(img_dict['real_depth_B'], isDepth=True)
    B_norm = get_normals(B_depth * 1000)
    A_depth_fake = tensor2im(img_dict['fake_depth_A'], isDepth=True)
    A_norm_fake = get_normals(A_depth_fake * 1000)
    B_depth_rec = tensor2im(img_dict['rec_depth_B'], isDepth=True)
    B_norm_rec = get_normals(B_depth_rec * 1000)

    max_dist = 5100 / 1000 #opt.max_distance=5100
    batch_size = A_imgs.shape[0]
    n_pic = min(batch_size, 3)
    n_col = 8
    fig_size = (40,30)
    n_row = 2 * n_pic
    fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=fig_size)
    fig.subplots_adjust(hspace=0.0, wspace=0.1)
    for i,ax in enumerate(axes.flatten()):
        ax.axis('off')
        if (i+1) % 8 == 0:
            ax.axis('on')

    for i in range(n_pic):
        axes[2*i,0].set_title('Real RGB')
        axes[2*i,1].set_title('Real Depth')
        axes[2*i,2].set_title('R-S Depth')
        axes[2*i,3].set_title('Cycle Depth A')
        axes[2*i,4].set_title('Real Norm')
        axes[2*i,5].set_title('R-S Norm')
        axes[2*i,6].set_title('Cycle Norm A')
        axes[2*i,7].set_title('Graph')

        axes[2*i+1,0].set_title('Syn RGB')
        axes[2*i+1,1].set_title('Syn Depth')
        axes[2*i+1,2].set_title('S-R Depth')
        axes[2*i+1,3].set_title('Cycle Depth B')
        axes[2*i+1,4].set_title('Syn Norm')
        axes[2*i+1,5].set_title('S-R Norm')
        axes[2*i+1,6].set_title('Cycle Norm B')
        axes[2*i+1,7].set_title('Graph')

        axes[2*i,0].imshow(A_imgs[i])
        axes[2*i,1].imshow(A_depth[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
        axes[2*i,2].imshow(B_depth_fake[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
        axes[2*i,3].imshow(A_depth_rec[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
        axes[2*i,4].imshow(A_norm[i])
        axes[2*i,5].imshow(B_norm_fake[i])
        axes[2*i,6].imshow(A_norm_rec[i])
        axes[2*i,7].plot(A_depth[i][100], label = 'Real Depth')
        axes[2*i,7].plot(B_depth_fake[i][100], label = 'R-S Depth')
        axes[2*i,7].legend()

        axes[2*i+1,0].imshow(B_imgs[i])
        axes[2*i+1,1].imshow(B_depth[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
        axes[2*i+1,2].imshow(A_depth_fake[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
        axes[2*i+1,3].imshow(B_depth_rec[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
        axes[2*i+1,4].imshow(B_norm[i])
        axes[2*i+1,5].imshow(A_norm_fake[i])
        axes[2*i+1,6].imshow(B_norm_rec[i])
        axes[2*i+1,7].plot(B_depth[i][100], label = 'Syn Depth')
        axes[2*i+1,7].plot(A_depth_fake[i][100], label = 'S-R Depth')
        axes[2*i+1,7].legend()

    wandb.log({"chart": fig}, step=global_step)
    plt.close(fig)        