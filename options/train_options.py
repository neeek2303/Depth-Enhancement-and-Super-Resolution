from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=2346, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=500, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--start_iter', type=int, default=0, help='')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--max_distance', type=int, default=10, help='max_distance')
        
        parser.add_argument('--save_image_folder', type=str, default='/root/code_for_article/depth_SR_git/int2s_sr_test', help='max_distance')
        
        
        parser.add_argument('--update_ratio', type=int, default=1, help='update_ratio G vs D')
        parser.add_argument('--replace_transpose', action='store_true', help='replace transpose convolution')
        parser.add_argument('--uint16', action='store_true', help='uint16 or not')
        parser.add_argument('--take', type=int, default=0, help='how much images take, if 0 take all')
        parser.add_argument('--custom_pathes', action='store_true', help='custom_pathes to data or not')
        
        #Paths 
        parser.add_argument('--path_to_intr', type=str, default='/root/data/un_depth/Scannet/', help='max_distance')
        parser.add_argument('--path_A', type=str, default='/root/data/un_depth/InteriorNet_5.1m/trainB/depth', help='path_A')
        parser.add_argument('--path_B', type=str, default='/root/data/un_depth/Scannet_ssim/Scannet_ssim/trainA/full_size/depth', help='path_B')
        
        parser.add_argument('--path_A_test', type=str, default='/root/data/un_depth/Scannet_ssim/Scannet_ssim/testB/full_size/depth', help='path_B_test')
        parser.add_argument('--path_B_test', type=str, default='/root/data/un_depth/Scannet_ssim/Scannet_ssim/testA/full_size/depth', help='path_B_test')
        
        parser.add_argument('--image_and_depth', action='store_true', help='image and depth')
        parser.add_argument('--A_add_paths', type=str, default='/root/data/un_depth/InteriorNet_5.1m/trainB/img', help='path_A_test')
        parser.add_argument('--B_add_paths', type=str, default='/root/data/un_depth/Scannet_ssim/Scannet_ssim/trainA/full_size/img', help='path_B_test')     
        parser.add_argument('--A_add_paths_test', type=str, default='/root/data/un_depth/Scannet_ssim/Scannet_ssim/testB/full_size/img', help='path_A_test')
        parser.add_argument('--B_add_paths_test', type=str, default='/root/data/un_depth/Scannet_ssim/Scannet_ssim/testA/full_size/img', help='path_B_test')
        
        
        parser.add_argument('--num_test', type=int, default=5000)
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--print_mean', action='store_true', help='')
        parser.add_argument('--cat', action='store_true', help='')
        parser.add_argument('--use_rec_masks', action='store_true', help='')
        parser.add_argument('--use_idt_masks', action='store_true', help='')
        parser.add_argument('--save_all', action='store_true', help='')
        
        parser.add_argument('--use_rec_iou_error', action='store_true', help='')
        parser.add_argument('--SR', action='store_true', help='')
        parser.add_argument('--use_wandb', action='store_true', help='')
        parser.add_argument('--back_rec_iou_error', action='store_true', help='')
        parser.add_argument('--iou_error_weight', type=float, default=0.5, help='iou_error_weight')
        
        
        

        
        
        
        
        
        
                # Main
        parser.add_argument('--Imagef_ndown', type=int, default=2)
        parser.add_argument('--Depthf_ndown', type=int, default=2)
        parser.add_argument('--Task_ndown', type=int, default=2)
        
        parser.add_argument('--Imagef_basef', type=int, default=32)
        parser.add_argument('--Depthf_basef', type=int, default=32)
        parser.add_argument('--Task_basef', type=int, default=64)
        
        parser.add_argument('--Imagef_outf', type=int, default=16)
        parser.add_argument('--Depthf_outf', type=int, default=128)
#         parser.add_argument('--Task_outf', type=int, default=32)
        
        
        parser.add_argument('--Imagef_type', type=str, default='resnet_6blocks')
        parser.add_argument('--Depthf_type', type=str, default='resnet_6blocks')
        parser.add_argument('--Task_type', type=str, default='unet_128')
        
        parser.add_argument('--use_rec_as_real_input', action='store_true', help='use_rec_as_real_input')
        parser.add_argument('--use_image_for_trans', action='store_true', help='use_rec_as_real_input')
        parser.add_argument('--norm_loss', action='store_true', help='use_rec_as_real_input')
        parser.add_argument('--use_smooth_loss', action='store_true', help='use_rec_as_real_input')
        
        parser.add_argument('--w_syn_adv', type=float, default=0.5, help='initial learning rate for adam')
        parser.add_argument('--w_real_l1', type=float, default=0.1, help='initial learning rate for adam')
        parser.add_argument('--w_holles', type=float, default=0.0, help='initial learning rate for adam')
        parser.add_argument('--w_syn_norm', type=float, default=0.0, help='initial learning rate for adam')
        parser.add_argument('--w_real_norm', type=float, default=0.0, help='initial learning rate for adam')
        parser.add_argument('--w_edge_s', type=float, default=0.0, help='initial learning rate for adam')
        parser.add_argument('--w_edge_r', type=float, default=0.0, help='initial learning rate for adam')        
        
        parser.add_argument('--w_rec_holles', type=float, default=0.0, help='initial learning rate for adam')
        
        parser.add_argument('--w_syn_l1', type=float, default=1, help='initial learning rate for adam')
        parser.add_argument('--w_syn_holes', type=float, default=2, help='initial learning rate for adam')
        parser.add_argument('--w_real_holes', type=float, default=5, help='initial learning rate for adam')
        parser.add_argument('--w_real_l1_d', type=float, default=1, help='initial learning rate for adam')   
        parser.add_argument('--w_real_l1_i', type=float, default=0.1, help='initial learning rate for adam') 
        parser.add_argument('--w_smooth', type=float, default=0.1, help='initial learning rate for adam') 
        parser.add_argument('--w_tv', type=float, default=0.1, help='initial learning rate for adam')
        parser.add_argument('--w_norm_idt', type=float, default=0, help='initial learning rate for adam') 
        parser.add_argument('--w_norm_cycle', type=float, default=0, help='initial learning rate for adam') 
        
        parser.add_argument('--w_loss_l1', type=float, default=0.1, help='initial learning rate for adam')
        parser.add_argument('--w_edge_l1', type=float, default=1, help='initial learning rate for adam')
        parser.add_argument('--w_ssim', type=float, default=1, help='initial learning rate for adam')
        
        parser.add_argument('--ImageDepthf_outf', type=int, default=128)
        parser.add_argument('--ImageDepthf_basef', type=int, default=32)
        parser.add_argument('--ImageDepthf_type', type=str, default='resnet_6blocks')
        
        parser.add_argument('--I2D_base', type=int, default=64)
        parser.add_argument('--I2D_type', type=str, default='unet_128')
        
        
        
        parser.add_argument('--scale_G', type=float, default=1.0, help='initial learning rate for adam')
        
        
        
        parser.add_argument('--notscannet', type=bool, default=True, help='initial learning rate for adam') 
        parser.add_argument('--use_D', action='store_true', help='')
        parser.add_argument('--use_edge', action='store_true', help='')
        parser.add_argument('--use_masked', action='store_true', help='')
        parser.add_argument('--use_scannet', action='store_true', help='')
        parser.add_argument('--use_tv', action='store_true', help='')
        parser.add_argument('--do_train', action='store_true', help='')
        parser.add_argument('--do_test', action='store_true', help='')
        parser.add_argument('--interiornet', action='store_true', help='')
        parser.add_argument('--no_aug', action='store_true', help='')
        self.isTrain = True        
        
        
        parser.add_argument('--load_size_h', type=int, default=480, help='scale images to this size')
        parser.add_argument('--load_size_w', type=int, default=640, help='scale images to this size')
        parser.add_argument('--crop_size_h', type=int, default=384, help='then crop to this size')
        parser.add_argument('--crop_size_w', type=int, default=512, help='then crop to this size')

        
        parser.add_argument('--use_i2d_in_input', action='store_true', help='')
        
        self.isTrain = True
        return parser
