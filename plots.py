def plot_cycle(img_dict, global_step, is_epoch=False, depth=True, stage='train'):

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