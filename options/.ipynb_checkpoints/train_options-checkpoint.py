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
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=20, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
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
        
        parser.add_argument('--update_ratio', type=int, default=1, help='update_ratio G vs D')
        parser.add_argument('--replace_transpose', action='store_true', help='replace transpose convolution')
        parser.add_argument('--uint16', action='store_true', help='uint16 or not')
        parser.add_argument('--take', type=int, default=0, help='how much images take, if 0 take all')
        parser.add_argument('--custom_pathes', action='store_true', help='custom_pathes to data or not')
        
        parser.add_argument('--path_A', type=str, default='/mnt/hdd/un_depth/semi/sample/trainB/depth', help='path_A')
        parser.add_argument('--path_B', type=str, default='/mnt/hdd/un_depth/NYUv2/depth', help='path_B')
#         parser.add_argument('--path_B', type=str, default='/mnt/hdd/un_depth/semi/sample/trainA/depth', help='path_A')
        
        parser.add_argument('--path_A_test', type=str, default='/mnt/hdd/un_depth/semi/sample/testB/depth', help='path_A_test')
        parser.add_argument('--path_B_test', type=str, default='/mnt/hdd/un_depth/NYUv2/depth_test', help='path_B_test')
#         parser.add_argument('--path_B_test', type=str, default='/mnt/hdd/un_depth/semi/sample/valA/depth', help='path_A_test')
        
        parser.add_argument('--image_and_depth', action='store_true', help='image and depth')
        parser.add_argument('--A_add_paths', type=str, default='/mnt/hdd/un_depth/semi/sample/trainB/img', help='path_A_test')
        parser.add_argument('--B_add_paths', type=str, default='/mnt/hdd/un_depth/NYUv2/img', help='path_B_test')
#         parser.add_argument('--B_add_paths', type=str, default='/mnt/hdd/un_depth/semi/sample/trainA/img', help='path_B_test')     
        
        parser.add_argument('--A_add_paths_test', type=str, default='/mnt/hdd/un_depth/semi/sample/testB/img', help='path_A_test')
        parser.add_argument('--B_add_paths_test', type=str, default='/mnt/hdd/un_depth/NYUv2/img_test', help='path_B_test')
#         parser.add_argument('--B_add_paths_test', type=str, default='/mnt/hdd/un_depth/semi/sample/valA/img', help='path_A_test')
        
        
        parser.add_argument('--num_test', type=int, default=5000)
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--print_mean', action='store_true', help='')

        

        self.isTrain = True
        return parser
