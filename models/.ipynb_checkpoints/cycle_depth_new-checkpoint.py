import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
import cv2

def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D

def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi *sigma **2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D

def get_thin_kernels(start=0, end=360, step=45):
        k_thin = 3  # actual size of the directional kernel
        # increase for a while to avoid interpolation when rotating
        k_increased = k_thin + 2

        # get 0° angle directional kernel
        thin_kernel_0 = np.zeros((k_increased, k_increased))
        thin_kernel_0[k_increased // 2, k_increased // 2] = 1
        thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

        # rotate the 0° angle directional kernel to get the other ones
        thin_kernels = []
        for angle in range(start, end, step):
            (h, w) = thin_kernel_0.shape
            # get the center to not rotate around the (0, 0) coord point
            center = (w // 2, h // 2)
            # apply rotation
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

            # get the k=3 kerne
            kernel_angle = kernel_angle_increased[1:-1, 1:-1]
            is_diag = (abs(kernel_angle) == 1)      # because of the interpolation
            kernel_angle = kernel_angle * is_diag   # because of the interpolation
            thin_kernels.append(kernel_angle)
        return thin_kernels


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss    
    
class CannyFilter(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 use_cuda=False):
        super(CannyFilter, self).__init__()
        # device
        self.device = 'cuda' if use_cuda else 'cpu'
        
        # gaussian

        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        
#         self.gaussian_filter = torch.nn.functional.conv2d(in_channels=1,
#                                          out_channels=1,
#                                          kernel_size=k_gaussian,
#                                          padding=k_gaussian // 2,
#                                          bias=False)
#         self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D).to(self.device)
        gaussian_filter_weight= torch.from_numpy(gaussian_2D).to(self.device)
    
        self.gaussian_filter = lambda x: torch.nn.functional.conv2d(x, gaussian_filter_weight,  padding=k_gaussian // 2, bias=False)
        self.gaussian_filter = self.gaussian_filter.to(self.device)
        
        # sobel

        sobel_2D = get_sobel_kernel(k_sobel)
        self.sobel_filter_x = torch.nn.functional.conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_x.weight[:] = torch.from_numpy(sobel_2D).to(self.device)
        self.sobel_filter_x = self.sobel_filter_x.to(self.device)


        self.sobel_filter_y = torch.nn.functional.conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_y.weight[:] = torch.from_numpy(sobel_2D.T).to(self.device)
        self.sobel_filter_y = self.sobel_filter_y.to(self.device)

        # thin

        thin_kernels = get_thin_kernels()
        directional_kernels = np.stack(thin_kernels)

        self.directional_filter = torch.nn.functional.conv2d(in_channels=1,
                                            out_channels=8,
                                            kernel_size=thin_kernels[0].shape,
                                            padding=thin_kernels[0].shape[-1] // 2,
                                            bias=False)
        self.directional_filter.weight[:, 0] = torch.from_numpy(directional_kernels)
        
        self.directional_filter = self.directional_filter.to(self.device)
        # hysteresis

        hysteresis = np.ones((3, 3)) + 0.25
        self.hysteresis = torch.nn.functional.conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False)
        self.hysteresis.weight[:] = torch.from_numpy(hysteresis)
        self.hysteresis = self.hysteresis.to(self.device)


    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=False):
        # set the setps tensors
        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).to(self.device)
        grad_x = torch.zeros((B, 1, H, W)).to(self.device)
        grad_y = torch.zeros((B, 1, H, W)).to(self.device)
        grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)
        grad_orientation = torch.zeros((B, 1, H, W)).to(self.device)

        # gaussian

        for c in range(C):
            blurred[:, c:c+1] = self.gaussian_filter(img[:, c:c+1])

            grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c+1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c+1])

        # thick edges

        grad_x, grad_y = grad_x / C, grad_y / C
        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
        grad_orientation = torch.atan(grad_y / grad_x)
        grad_orientation = grad_orientation * (360 / np.pi) + 180 # convert to degree
        grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45

        # thin edges

        directional = self.directional_filter(grad_magnitude)
        # get indices of positive and negative directions
        positive_idx = (grad_orientation / 45) % 8
        negative_idx = ((grad_orientation / 45) + 4) % 8
        thin_edges = grad_magnitude.clone()
        # non maximum suppression direction by direction
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) * 1
            is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])

            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)

            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
            thin_edges[to_remove] = 0.0

        # thresholds

        if low_threshold is not None:
            low = thin_edges > low_threshold

            if high_threshold is not None:
                high = thin_edges > high_threshold
                # get black/gray/white only
                thin_edges = low * 0.5 + high * 0.5

                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (thin_edges == 0.5) * 1
                    weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                    thin_edges = high * 1 + weak_is_high * 1
            else:
                thin_edges = low * 1


        return blurred, grad_x, grad_y, grad_magnitude, grad_orientation, thin_edges

class SurfaceNormals(nn.Module):
    
    def __init__(self):
        super(SurfaceNormals, self).__init__()
    
    def forward(self, depth):
        dzdx = -self.gradient_for_normals(depth, axis=2)
        dzdy = -self.gradient_for_normals(depth, axis=3)
        norm = torch.cat((dzdx, dzdy, torch.ones_like(depth)), dim=1)
        n = torch.norm(norm, p=2, dim=1, keepdim=True)
        return norm / (n + 1e-6)
    
    def gradient_for_normals(self, f, axis=None):
        N = f.ndim  # number of dimensions
        dx = 1.0
    
        # use central differences on interior and one-sided differences on the
        # endpoints. This preserves second order-accuracy over the full domain.
        # create slice objects --- initially all are [:, :, ..., :]
        slice1 = [slice(None)]*N
        slice2 = [slice(None)]*N
        slice3 = [slice(None)]*N
        slice4 = [slice(None)]*N
    
        otype = f.dtype
        if otype is torch.float32:
            pass
        else:
            raise TypeError('Input shold be torch.float32')
    
        # result allocation
        out = torch.empty_like(f, dtype=otype)
    
        # Numerical differentiation: 2nd order interior
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(None, -2)
        slice3[axis] = slice(1, -1)
        slice4[axis] = slice(2, None)
    
        out[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / (2. * dx)
    
        # Numerical differentiation: 1st order edges
        slice1[axis] = 0
        slice2[axis] = 1
        slice3[axis] = 0
        dx_0 = dx 
        # 1D equivalent -- out[0] = (f[1] - f[0]) / (x[1] - x[0])
        out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_0

        slice1[axis] = -1
        slice2[axis] = -1
        slice3[axis] = -2
        dx_n = dx 
        # 1D equivalent -- out[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
        out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_n
        return out


def scale_pyramid(img, num_scales):
    scaled_imgs = [img]

    s = img.size()

    h = s[2]
    w = s[3]

    for i in range(1, num_scales):
        ratio = 2**i
        nh = h // ratio
        nw = w // ratio
        scaled_img = F.upsample(img, size=(nh, nw), mode='bilinear', align_corners=True)
        scaled_imgs.append(scaled_img)

    scaled_imgs.reverse()
    return scaled_imgs   
    
    
def gradient_x(img):
    gx = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gx


def gradient_y(img):
    gy = img[:, :, :, :-1] - img[:, :, :, 1:]
    return gy


# calculate the gradient loss
def get_smooth_weight(depth, Image, num_scales):
    
    Images = scale_pyramid(Image, num_scales)
    
    depths = scale_pyramid(depth, num_scales)
    
    depth_gradient_x = [gradient_x(d) for d in depths]
    depth_gradient_y = [gradient_y(d) for d in depths]

    Image_gradient_x = [gradient_x(img) for img in Images]
    Image_gradient_y = [gradient_y(img) for img in Images]

    weight_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in Image_gradient_x]
    weight_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in Image_gradient_y]

    smoothness_x = [depth_gradient_x[i] * weight_x[i] for i in range(num_scales)]
    smoothness_y = [depth_gradient_y[i] * weight_y[i] for i in range(num_scales)]

    loss_x = [torch.mean(torch.abs(smoothness_x[i]))/2**i for i in range(num_scales)]
    loss_y = [torch.mean(torch.abs(smoothness_y[i]))/2**i for i in range(num_scales)]

    return sum(loss_x+loss_y)
    
  
    
class CycleGANModel_depth_new(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['task_syn','holes_syn', 'task_real_by_depth','task_real_by_image', 'syn_mean_diff', 'real_mean_diff']
        if opt.norm_loss:
            self.loss_names+=['syn_norms']
        
        if opt.use_D:
            self.loss_names+=['G_pred', 'D_depth']
        
        if opt.use_smooth_loss:
            self.loss_names+=['smooth']
            
        if opt.use_tv:
            self.loss_names+=['tv']    
            
            
            
            
        if opt.print_mean:
            self.loss_names = ['syn_mean_diff', 'real_mean_diff', 'mean_of_abs_diff_syn', 'mean_of_abs_diff_real', 'L1_syn', 'L1_real']
            
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['syn_image', 'syn_depth', 'syn2real_depth', 'syn_depth_by_image',  'pred_syn_depth']
        visual_names_B = ['real_image', 'real_depth', 'real_depth_by_image', 'pred_real_depth', 'mask']
        
        if opt.norm_loss:
            visual_names_A += ['norm_syn','norm_syn_pred', 'norm_syn2real']
            visual_names_B += [ 'norm_real','norm_real_pred']
        
        if self.opt.use_masked:
            visual_names_B+=['depth_masked']
            self.loss_names+=['holes_real']

        if opt.use_edge:
            self.loss_names+=['edge_real', 'edge_syn']
            visual_names_B+=['edges_real', 'edges_real_pred']
            visual_names_A+=['edges_syn', 'edges_syn_pred']
            
            
        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        
        
        self.model_names = ['Image2Depth','I2D_features_real', 'I2D_features_syn',  'Task', 'Depth_f_syn', 'Depth_f_real']
        
        
        # Define networks 
        
        if opt.use_D:
            self.model_names +=['D_depth']
        
        self.border = -0.95

        if opt.use_D:
            self.netD_depth = networks.define_D(3, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)         
        ### Image2Depth 
        self.netI2D_features_real = networks.define_G(3, opt.ImageDepthf_outf, opt.ImageDepthf_basef, opt.ImageDepthf_type, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.replace_transpose)
        
        self.netI2D_features_syn = networks.define_G(3, opt.ImageDepthf_outf, opt.ImageDepthf_basef, opt.ImageDepthf_type, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.replace_transpose)
        
        I2D_input_features = opt.ImageDepthf_outf
        self.netImage2Depth = networks.define_G(I2D_input_features, 1, opt.I2D_base, opt.I2D_type, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.replace_transpose)  
        
        ### Translation 
        translation_input_d = 4 if opt.use_image_for_trans else 1 
#         self.netG_A_d = networks.define_G(translation_input_d, 1, opt.ngf, opt.netG, opt.norm,
#                                         not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.replace_transpose)
        
        if opt.use_rec_as_real_input or opt.use_image_for_trans:
            self.netG_B_d = networks.define_G(translation_input_d, 1, opt.ngf, opt.netG, opt.norm,
                                                not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.replace_transpose)
        
        ## Main Part 
        self.netDepth_f_syn = networks.define_G(2, opt.Depthf_outf, opt.Depthf_basef, opt.Depthf_type, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.replace_transpose, n_down = opt.Depthf_ndown)
        
        self.netDepth_f_real = networks.define_G(2, opt.Depthf_outf, opt.Depthf_basef, opt.Depthf_type, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.replace_transpose, n_down = opt.Depthf_ndown)  
        
        task_input_features = opt.ImageDepthf_outf + opt.Depthf_outf + 5
        self.netTask = networks.define_G(task_input_features, 1, opt.Task_basef, opt.Task_type, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.replace_transpose, n_down = opt.Task_ndown)
        

        self.loss_L1_syn=0
        self.loss_L1_real=0

        
        if self.isTrain:
            if opt.use_D:
                self.fake_depth_pool = ImagePool(opt.pool_size) 
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
                
            # define loss functions

            self.criterion_task = torch.nn.L1Loss()
            self.criterion_task_2 = torch.nn.MSELoss()
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netDepth_f_syn.parameters(), self.netDepth_f_real.parameters(), self.netTask.parameters()), lr=opt.lr)
#             self.optimizer_G = torch.optim.Adam(itertools.chain( self.netTask.parameters()), lr=opt.lr)
#             self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_depth.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if self.opt.use_D:
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_depth.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
#             self.optimizers.append(self.optimizer_D)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.syn_image = input['A_i' if AtoB else 'B_i'].to(self.device)
        self.real_image = input['B_i' if AtoB else 'A_i'].to(self.device)
        
        self.syn_depth = input['A_d' if AtoB else 'B_d'].to(self.device)
        self.real_depth = input['B_d' if AtoB else 'A_d'].to(self.device)
        
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        
        
        holl_mask = torch.where(self.real_depth<=self.border, torch.tensor(1).float().to(self.real_depth.device), torch.tensor(0).float().to(self.real_depth.device))
        right_mask = holl_mask.clone()
        right_mask[:,:,:-1,:]+=right_mask[:,:,1:,:].clone()
#         right_mask[:,:,:-1,:]+=right_mask[:,:,1:,:].clone()
        
        right_mask[:,:,1:,:]+=right_mask[:,:,:-1,:].clone()
#         right_mask[:,:,1:,:]+=right_mask[:,:,:-1,:].clone()
        
        right_mask[:,:,:,:-1]+=right_mask[:,:,:,1:].clone()
#         right_mask[:,:,:,:-1]+=right_mask[:,:,:,1:].clone()
        
        right_mask[:,:,:,1:]+=right_mask[:,:,:,:-1].clone()
#         right_mask[:,:,:,1:]+=right_mask[:,:,:,:-1].clone()
        
        right_mask = torch.where(right_mask<1, torch.tensor(1).float().to(self.real_depth.device), torch.tensor(0).float().to(right_mask.device))
        mask = right_mask 
        self.mask = right_mask
        
        
        
        
        
        
#         if self.opt.use_image_for_trans:
#             self.syn2real_depth = self.netG_A_d(torch.cat([self.syn_depth, self.syn_image], dim=1) )
#             syn_depth = self.syn2real_depth
#             real_depth = self.real_depth
#             if self.opt.use_rec_as_real_input:
#                 r2s = self.netG_B_d(torch.cat([self.real_depth, self.real_image], dim=1))
#                 self.real_rec = self.netG_A_d(torch.cat([r2s, self.real_image], dim=1))
#                 real_depth = self.real_rec
#         else:
#             self.syn2real_depth = self.netG_A_d(self.syn_depth)
#             syn_depth = self.syn2real_depth
#             real_depth = self.real_depth
            
#             if opt.use_rec_as_real_input:
#                 self.real_rec = self.netG_A_d(self.netG_B_d(self.real_depth))
#                 real_depth = self.real_rec
        self.syn2real_depth =  self.syn_depth   
        syn_depth = self.syn_depth
        real_depth = self.real_depth
        
        image_features_syn = self.netI2D_features_syn(self.syn_image)
        image_features_real = self.netI2D_features_real(self.real_image)
        self.syn_depth_by_image = self.netImage2Depth(image_features_syn)
        self.real_depth_by_image = self.netImage2Depth(image_features_real)
        
        
        if self.opt.use_masked:
            out = []
            for i in range(self.mask.shape[0]):
                number = np.random.randint(6,11)
                xs = np.random.choice(self.mask.shape[3], number, replace=False)
                ys = np.random.choice(self.mask.shape[2], number, replace=False)
                sizes_x = np.random.randint(self.mask.shape[3]//16, self.mask.shape[3]//4, number)*np.random.randint(0,2)
                sizes_y = np.random.randint(self.mask.shape[2]//16, self.mask.shape[2]//4, number)*np.random.randint(0,2)
                ones = np.ones_like(self.mask.cpu().numpy()[0][0])
                
                for x, y, s_x, s_y in zip(xs,ys,sizes_x, sizes_y):
                    ones[y:y+s_y, x:x+s_x]=0
                    
                ones = np.where((self.mask[i][0].cpu().numpy()>0.05) & (ones<0.05), 0, 1)
                out.append(np.expand_dims(ones, 0))
                
            self.gt_mask = torch.from_numpy(np.array(out)).to(self.real_depth.device)
#             print(self.gt_mask.shape)
    #             self.gt_mask = torch.from_numpy(np.where((self.mask.cpu().numpy()>0.05) & (ones<0.05), 0, 1)).float().to(self.real_depth.device)
    
        if self.opt.use_masked:
            self.depth_masked = torch.where(self.gt_mask<0.05, torch.tensor(-1).float().to(real_depth.device), real_depth)
            self.real_depth_out =  self.depth_masked
            depth_r_inp =  torch.cat([self.depth_masked, self.real_depth_by_image], dim=1) 
        else:    
            self.real_depth_out =  real_depth
            depth_r_inp =  torch.cat([real_depth, self.real_depth_by_image], dim=1) 
            
        
        
        depth_s_inp =  torch.cat([syn_depth, self.syn_depth_by_image], dim=1)
        self.syn_depth_out =  syn_depth
        syn_task_input = torch.cat([self.netDepth_f_syn(depth_s_inp), image_features_syn, depth_s_inp, self.syn_image], dim=1)
        real_task_input = torch.cat([self.netDepth_f_real(depth_r_inp), image_features_real, depth_r_inp, self.real_image], dim=1)

        
#         syn_task_input = torch.cat([self.netDepth_f_syn(syn_depth), image_features_syn, self.syn_depth_by_image], dim=1)
#         real_task_input = torch.cat([self.netDepth_f_real(real_depth), image_features_real, self.real_depth_by_image], dim=1)

#         syn_task_input = torch.cat([syn_depth, image_features_syn], dim=1)
#         real_task_input = torch.cat([real_depth, image_features_real], dim=1)
        self.pred_syn_depth = self.netTask(syn_task_input)
        self.pred_real_depth = self.netTask(real_task_input)
        
#         print(torch.min(self.pred_real_depth))
        
        
        
        syn_mean = torch.mean(self.syn_depth) 
        syn_pred_mean = torch.mean(self.pred_syn_depth) 
        
        self.loss_syn_mean_diff = (syn_mean-syn_pred_mean).cpu().detach().numpy()
        
        self.loss_mean_of_abs_diff_syn =  np.mean(np.absolute((self.syn_depth-self.pred_syn_depth).cpu().detach().numpy()))
        
        

        real_mean = torch.mean(self.real_depth*mask) 
        real_pred_mean = torch.mean(self.pred_real_depth*mask) 
        
        self.loss_real_mean_diff  = (real_mean-real_pred_mean).cpu().detach().numpy() 
        self.loss_mean_of_abs_diff_real =  np.mean(np.absolute((self.real_depth*mask-self.pred_real_depth*mask).cpu().detach().numpy()))
        
    def backward_D_basic(self, netD, real, fake, back=True):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        if back:
            loss_D.backward()
        return loss_D
    
    def backward_D_depth(self, back=True):
        """Calculate GAN loss for discriminator D_B"""
#         fake_depth_1 = self.fake_depth_pool.query(self.norm_syn_pred)
        fake_depth = self.fake_depth_pool.query(self.norm_real_pred)
#         fake_depth = random.choice([fake_depth_1, fake_depth_2])
        self.loss_D_depth = self.backward_D_basic(self.netD_depth, self.norm_syn, fake_depth, back=back)   

    def backward_G(self, back=True):
        """Calculate the loss for generators G_A and G_B"""

        
        # Forward cycle loss || G_B(G_A(A)) - A||
        mask_syn = torch.where(self.syn2real_depth<self.border, torch.tensor(1).float().to(self.syn2real_depth.device), torch.tensor(0).float().to(self.syn2real_depth.device))
        self.loss_task_syn = self.criterion_task(self.syn_depth, self.pred_syn_depth) 
        self.loss_holes_syn = self.criterion_task(self.syn_depth*mask_syn, self.pred_syn_depth*mask_syn) 
        
        
#         mask_real_d = torch.where(self.real_depth<=self.border, torch.tensor(0).float().to(self.real_depth.device), torch.tensor(1).float().to(self.real_depth.device))
#         mask_real_i = torch.where(self.real_depth<self.border, torch.tensor(1).float().to(self.real_depth.device), torch.tensor(0).float().to(self.real_depth.device))
        
        mask_real_d = self.mask
        mask_real_i = torch.where(self.mask<1, torch.tensor(1).float().to(self.real_depth.device), torch.tensor(0).float().to(self.real_depth.device))
        
        
#         norm_syn = get_normal(self.syn_depth)
#         norm_syn_pred = get_normal(self.pred_syn_depth)
        
        if self.opt.norm_loss:
            calc_norm = SurfaceNormals()
            self.norm_syn = calc_norm(self.syn_depth)
            self.norm_syn2real = calc_norm(self.syn2real_depth)
            self.norm_syn_pred = calc_norm(self.pred_syn_depth)
            self.norm_real = calc_norm(self.real_depth)
            self.norm_real_pred = calc_norm(self.pred_real_depth)
            self.loss_syn_norms = self.criterion_task(self.norm_syn, self.norm_syn_pred) 
        
        self.loss_task_real_by_depth = self.criterion_task(self.real_depth*mask_real_d, self.pred_real_depth*mask_real_d) 
        self.loss_task_real_by_image = self.criterion_task(self.real_depth_by_image*mask_real_i, self.pred_real_depth*mask_real_i) 
        
        if self.opt.use_edge:
            canny = CannyFilter(use_cuda=True)
            self.edges_real = canny(self.real_image)[3]
            self.edges_syn = canny(self.syn_image)[3]
            self.edges_real_pred = canny(self.pred_real_depth)[3]
            self.edges_syn_pred = canny(self.pred_syn_depth)[3]
            
            
        
        if self.opt.use_D:
            self.loss_G_pred =  self.criterionGAN(self.netD_depth(self.norm_real_pred), True)
#             self.loss_G_pred = self.criterionGAN(self.netD_depth(self.norm_real_pred), True)  + self.criterionGAN(self.netD_depth(self.norm_syn_pred), True) 

        
        # combined loss and calculate gradients
        self.loss_G = self.loss_task_syn*self.opt.w_syn_l1 + self.loss_holes_syn*self.opt.w_syn_holes + self.loss_task_real_by_depth*self.opt.w_real_l1_d + self.loss_task_real_by_image*self.opt.w_real_l1_i
        
        if self.opt.use_tv:
            self.loss_tv = tv_loss(self.norm_syn_pred, 1)
            self.loss_G+= self.loss_tv * self.opt.w_tv
            
        if self.opt.use_D:
            self.loss_G+=self.loss_G_pred*self.opt.w_syn_adv
        
        if self.opt.norm_loss:
            self.loss_G+=self.loss_syn_norms*self.opt.w_syn_norm
            
        if self.opt.use_masked:
            mask_real_gt = torch.where(self.gt_mask>0.1, torch.tensor(0).float().to(self.pred_real_depth.device), torch.tensor(1).float().to(self.pred_real_depth.device))
            self.loss_holes_real = self.criterion_task(self.real_depth*mask_real_gt, self.pred_syn_depth*mask_real_gt) 
            self.loss_G+=self.loss_holes_real*self.opt.w_real_holes  

        if self.opt.use_smooth_loss:
            self.loss_smooth = get_smooth_weight(self.pred_real_depth, self.real_image, 3) 
            self.loss_G+=self.loss_smooth*self.opt.w_smooth 
            
        if self.opt.use_edge:
            self.loss_edge_real = self.criterion_task(self.edges_real, self.edges_real_pred) 
            self.loss_edge_syn = self.criterion_task(self.edges_syn, self.edges_syn_pred) 
            print(self.loss_edge_real, self.loss_edge_syn)
            self.loss_G+=self.loss_edge_syn*self.opt.w_edge_s + self.loss_edge_real*self.opt.w_edge_r
            
#         print(self.loss_task_syn*self.opt.w_syn_l1, self.loss_holes_syn*self.opt.w_syn_holes,  self.loss_task_real_by_depth*self.opt.w_real_l1_d, self.loss_task_real_by_image*self.opt.w_real_l1_i, self.loss_syn_norms*self.opt.w_syn_norm, self.loss_holes_real*self.opt.w_real_holes, self.loss_smooth*self.opt.w_smooth, self.opt.use_masked)    
            
        self.loss_G *= self.opt.scale_G
#         print(self.loss_G)
        if back:
            self.loss_G.backward()

    def optimize_parameters(self, iters, fr=1):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
#         self.set_requires_grad([self.netG_A , self.netG_A_d, self.netD_depth], False)  # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netI2D_features_syn, self.netI2D_features_real, self.netImage2Depth ], False)  # Ds require no gradients when optimizing Gs
    
        if self.opt.use_D:
            self.set_requires_grad([self.netI2D_features_syn, self.netI2D_features_real, self.netImage2Depth, self.netD_depth ], False)
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        
        if self.opt.use_D:
            if iters%(fr*self.opt.batch_size) or iters<2000 == 0:
                self.set_requires_grad([self.netD_depth], True)  # Ds require no gradients when optimizing Gs
                self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
                self.backward_D_depth()      # calculate gradients for D_A
                self.optimizer_D.step()  # update D_A and D_B's weights
#         if iters%fr == 0:
#             self.set_requires_grad([self.netD_depth], True)  # Ds require no gradients when optimizing Gs
#             self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
#             self.backward_D_depth()      # calculate gradients for D_A
#             self.optimizer_D.step()  # update D_A and D_B's weights
            
            
    def calculate(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        self.backward_G(back=False)             # calculate gradients for G_A and G_B
        if self.opt.use_D:
            self.backward_D_depth(back=False)      # calculate gradients for D_A
