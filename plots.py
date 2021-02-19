import numpy as np
import matplotlib.pyplot as plt
import wandb

def plot_only_depth(img_dict, global_step, depth=True,  is_epoch=False, stage='train'):

        syn_image = img_dict['syn_image'].cpu().detach()
        syn_depth = img_dict['syn_depth'].cpu().detach()
        pred_syn_depth = img_dict['pred_syn_depth'].cpu().detach()
        
        
        real_image = img_dict['real_image'].cpu().detach()
        real_depth = img_dict['real_depth'].cpu().detach()
        pred_real_depth = img_dict['pred_real_depth'].cpu().detach()

        
        n_col = 3
        n_row = 2
        fig, axes = plt.subplots(nrows = n_row, ncols = n_col, figsize=(45, 30))
        fig.subplots_adjust(hspace=0.0, wspace=0.01)
        
        for ax in axes.flatten():
            ax.axis('off')
            
            
        pr_d = lambda img: np.clip((img[0].permute(1,2,0).numpy()+1)/2,0,1)[:,:,0]
        pr = lambda img: np.clip((img[0].permute(1,2,0).numpy()+1)/2,0,1)

        axes[0,0].set_title('syn_image')
        axes[0,1].set_title('syn_depth')
        axes[0,2].set_title('pred_syn_depth')

        axes[1,0].set_title('real_image')
        axes[1,1].set_title('real_depth')
        axes[1,2].set_title('pred_real_depth')

            
        axes[0,0].imshow(pr(syn_image))
        axes[0,1].imshow(pr_d(syn_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,2].imshow(pr_d(pred_syn_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
            
        
        axes[1,0].imshow(pr(real_image))
        axes[1,1].imshow(pr_d(real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[1,2].imshow(pr_d(pred_real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        
        wandb.log({"chart": fig}, step=global_step)
        plt.close(fig)

def plot_main_new(img_dict, global_step, depth=True,  is_epoch=False, stage='train'):

        syn_image = img_dict['syn_image'].cpu().detach()
        syn_depth = img_dict['syn_depth'].cpu().detach()
        syn2real_depth = img_dict['syn2real_depth'].cpu().detach()
#         syn2real_image = img_dict['syn2real_image'].cpu().detach()
        pred_syn_depth = img_dict['pred_syn_depth'].cpu().detach()
        syn_depth_by_image = img_dict['syn_depth_by_image'].cpu().detach()
        
        real_image = img_dict['real_image'].cpu().detach()
        real_depth = img_dict['real_depth'].cpu().detach()
        pred_real_depth = img_dict['pred_real_depth'].cpu().detach()
        real_depth_by_image = img_dict['real_depth_by_image'].cpu().detach()
        mask  = img_dict['mask'].cpu().detach()
#         sr2syn = img_dict['sr2syn_depth'].cpu().detach()
#         real2syn = img_dict['real2syn_depth'].cpu().detach()
        
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
        axes[0,4].set_title('syn_depth_by_image')
#         axes[0,4].set_title('sr2syn')

        axes[1,0].set_title('mask')
        axes[1,1].set_title('real_image')
        axes[1,2].set_title('real_depth')
        axes[1,3].set_title('pred_real_depth')
        axes[1,4].set_title('real_depth_by_image')
#         axes[1,3].set_title('real2syn')
            
        axes[0,0].imshow(pr(syn_image))
        axes[0,1].imshow(pr_d(syn_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,2].imshow(pr_d(syn2real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,3].imshow(pr_d(pred_syn_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,4].imshow(pr_d(syn_depth_by_image), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
#         axes[0,4].imshow(pr_d(pred_syn_depth*0), cmap=plt.get_cmap('RdYlBu'))
            
        axes[1,0].imshow(pr_d(mask)) # nothing
        
        axes[1,1].imshow(pr(real_image))
        axes[1,2].imshow(pr_d(real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[1,3].imshow(pr_d(pred_real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[1,4].imshow(pr_d(real_depth_by_image), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
#         axes[1,4].imshow(pr_d(real2syn), cmap=plt.get_cmap('RdYlBu'))
        
        wandb.log({"chart": fig}, step=global_step)
        plt.close(fig)
        
        
def plot_main_new_norm(img_dict, global_step, depth=True,  is_epoch=False, stage='train'):

        syn_image = img_dict['syn_image'].cpu().detach()
        syn_depth = img_dict['syn_depth'].cpu().detach()
        syn2real_depth = img_dict['syn2real_depth'].cpu().detach()
#         syn2real_depth = img_dict['pred_syn_impr'].cpu().detach()
        pred_syn_depth = img_dict['pred_syn_depth'].cpu().detach()
        syn_depth_by_image = img_dict['syn_depth_by_image'].cpu().detach()
        
        syn_norm = img_dict['norm_syn'].cpu().detach()
        norm_syn2real = img_dict['norm_syn2real'].cpu().detach()
        syn_norm_pred = img_dict['norm_syn_pred'].cpu().detach()
        
        real_image = img_dict['real_image'].cpu().detach()
        real_depth = img_dict['real_depth'].cpu().detach()
        pred_real_depth = img_dict['pred_real_depth'].cpu().detach()
        real_depth_by_image = img_dict['real_depth_by_image'].cpu().detach()
        mask  = img_dict['mask'].cpu().detach()
        gt_mask  = img_dict['depth_masked'].cpu().detach()
#         gt_mask  = img_dict['pred_real_impr'].cpu().detach()
    
        real_norm = img_dict['norm_real'].cpu().detach()
        real_norm_pred = img_dict['norm_real_pred'].cpu().detach()

        
        n_col = 4
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
        axes[0,3].set_title('pred_syn_depth')
        axes[1,0].set_title('syn_depth_by_image')
        axes[1,1].set_title('norm_syn')
        axes[1,2].set_title('norm_syn_pred')
        axes[1,3].set_title('nothing') 


        axes[2,0].set_title('mask')
        axes[2,1].set_title('real_image')
        axes[2,2].set_title('real_depth')
        axes[2,3].set_title('pred_real_depth')
        axes[3,0].set_title('real_depth_by_image')
        axes[3,1].set_title('norm_real')
        axes[3,2].set_title('norm_real_pred')
        axes[3,3].set_title('gt_mask')         

            
        axes[0,0].imshow(pr(syn_image))
        axes[0,1].imshow(pr_d(syn_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,2].imshow(pr_d(syn2real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,3].imshow(pr_d(pred_syn_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[1,0].imshow(pr_d(syn_depth_by_image), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[1,1].imshow(pr(syn_norm*1000))
        axes[1,2].imshow(pr(norm_syn2real*1000))
        axes[1,3].imshow(pr(syn_norm_pred*1000))
            
        axes[2,0].imshow(pr_d(mask))
        axes[2,1].imshow(pr(real_image))
        axes[2,2].imshow(pr_d(real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[2,3].imshow(pr_d(pred_real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[3,0].imshow(pr_d(real_depth_by_image), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[3,2].imshow(pr(real_norm*1000))
        axes[3,3].imshow(pr(real_norm_pred*1000))
#         axes[3,1].imshow(pr(real_norm_pred*0))
        axes[3,1].imshow(pr_d(gt_mask), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        
        wandb.log({"chart": fig}, step=global_step)
        plt.close(fig)        
        
def plot_main(img_dict, global_step, depth=True,  is_epoch=False, stage='train'):

        syn_image = img_dict['syn_image'].cpu().detach()
        syn_depth = img_dict['syn_depth'].cpu().detach()
        syn2real_depth = img_dict['syn2real_depth'].cpu().detach()
#         syn2real_image = img_dict['syn2real_image'].cpu().detach()
        pred_syn_depth = img_dict['pred_syn_depth'].cpu().detach()
        
        
        real_image = img_dict['real_image'].cpu().detach()
        real_depth = img_dict['real_depth'].cpu().detach()
        pred_real_depth = img_dict['pred_real_depth'].cpu().detach()
        mask  = img_dict['mask'].cpu().detach()
#         sr2syn = img_dict['sr2syn_depth'].cpu().detach()
#         real2syn = img_dict['real2syn_depth'].cpu().detach()
        
        n_col = 4
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
#         axes[0,4].set_title('sr2syn')

        axes[1,0].set_title('mask')
        
        axes[1,1].set_title('real_image')
        axes[1,2].set_title('real_depth')
        axes[1,3].set_title('pred_real_depth')
#         axes[1,3].set_title('real2syn')
            
        axes[0,0].imshow(pr(syn_image))
        axes[0,1].imshow(pr_d(syn_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,2].imshow(pr_d(syn2real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[0,3].imshow(pr_d(pred_syn_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
#         axes[0,4].imshow(pr_d(pred_syn_depth*0), cmap=plt.get_cmap('RdYlBu'))
            
        axes[1,0].imshow(pr_d(mask)) # nothing
        
        axes[1,1].imshow(pr(real_image))
        axes[1,2].imshow(pr_d(real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
        axes[1,3].imshow(pr_d(pred_real_depth), cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=1)
#         axes[1,4].imshow(pr_d(real2syn), cmap=plt.get_cmap('RdYlBu'))
        
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