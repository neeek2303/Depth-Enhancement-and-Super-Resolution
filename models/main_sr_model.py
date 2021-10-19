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
import imageio
from .norms import SurfaceNormals_new, get_imp_matrx, SurfaceNormals


def tv_loss(img):
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
    loss = (h_variance + w_variance)
    return loss


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
    
  
    
class MainSRModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['task_syn','holes_syn','holes_syn_l2', 'task_real_by_depth', 'task_real_by_image','syn_mean_diff', 'real_mean_diff', 'tv_syn_norm','tv_real_norm', 'syn_norms_holes', 'tv_syn_norm_old','tv_real_norm_old', 'syn_norms_old']
        if opt.norm_loss:
            self.loss_names+=['syn_norms']
        
        if opt.use_D:
            self.loss_names+=['G_pred', 'D_depth']
        
        if opt.use_smooth_loss:
            self.loss_names+=['smooth']
     
        if opt.print_mean:
            self.loss_names = ['syn_mean_diff', 'real_mean_diff', 'mean_of_abs_diff_syn', 'mean_of_abs_diff_real', 'L1_syn', 'L1_real']
            
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['syn_image', 'syn_depth', 'syn2real_depth',  'syn_mask', 'pred_syn_depth', 'mask_syn_add_holes', 'syn_depth_by_image']
        visual_names_B = ['real_image', 'real_depth','real_depth_by_image', 'pred_real_depth', 'real_mask', 'mask_real_add_holes']
        
        if opt.norm_loss:
            visual_names_A += ['norm_syn','norm_syn_pred', 'norm_syn2real']
            visual_names_B += [ 'norm_real','norm_real_pred']
        
        if self.opt.use_masked:
            visual_names_B+=['depth_masked']
            visual_names_A += ['syn2real_depth_masked']
            
            self.loss_names+=['holes_real', 'holes_real_l2']

        if opt.use_edge:
            self.loss_names+=['edge_real', 'edge_syn']
            visual_names_B+=['edges_real', 'edges_real_pred']
            visual_names_A+=['edges_syn', 'edges_syn_pred']
            
            
        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        
        
        self.model_names = ['G_A_d',  'I2D_features', 'Image2Depth', 'Task', 'Depth_f']
        
        # Define networks 
        
        if opt.use_D:
            self.model_names +=['D_depth']
        
        self.border = -0.97

        if opt.use_D:
            self.netD_depth = networks.define_D(3, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)         
#         ### Image2Depth 
        self.netI2D_features = networks.define_G(3, opt.ImageDepthf_outf, opt.ImageDepthf_basef, opt.ImageDepthf_type, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.replace_transpose)

        
        
        I2D_input_features = opt.ImageDepthf_outf
        self.netImage2Depth = networks.define_G(I2D_input_features, 1, opt.I2D_base, opt.I2D_type, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.replace_transpose)  
        
        ### Translation 
        translation_input_d = 4 if opt.use_image_for_trans else 1 
#         self.netG_A_d = networks.define_G(translation_input_d, 1, opt.ngf, opt.netG, opt.norm,
#                                         not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.replace_transpose)
        from . import translation_network
        from types import SimpleNamespace

        netG_B_opt = SimpleNamespace(ngf_img=32, ngf_depth=32, ngf=64, norm='group', dropout=False, init_type='normal', gpu_ids =[0,1,2,3], input_nc_img=3, n_downsampling=2, use_semantic = False, n_blocks=9, upsampling_type="transpose", output_nc_depth=1, input_nc_depth=1)
        self.netG_A_d = translation_network.define_Gen(netG_B_opt, input_type='img_depth')

        
        if opt.use_rec_as_real_input :
            self.netG_B_d = networks.define_G(translation_input_d, 1, opt.ngf, opt.netG, opt.norm,
                                                not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.replace_transpose)
 

        self.netDepth_f = networks.define_G(2, opt.Depthf_outf, opt.Depthf_basef, opt.Depthf_type, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.replace_transpose, n_down = opt.Depthf_ndown)
                                            
        task_input_features =  5 +  opt.ImageDepthf_outf + opt.Depthf_outf
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
            

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netDepth_f.parameters(), self.netTask.parameters()), lr=opt.lr)
            self.optimizers.append(self.optimizer_G)
            if self.opt.use_D:
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_depth.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
#             self.optimizers.append(self.optimizer_D)

        


    
    def set_input(self, input):

        AtoB = self.opt.direction == 'AtoB'
        self.syn_image = input['A_i' if AtoB else 'B_i'].to(self.device)
        self.real_image = input['B_i' if AtoB else 'A_i'].to(self.device)
        
        self.syn_depth = input['A_d' if AtoB else 'B_d'].to(self.device)
        self.real_depth = input['B_d' if AtoB else 'A_d'].to(self.device)
        
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.A_paths = input['A_paths']
        self.B_paths = input['B_paths']
        
        self.K_A = input['K_A']
        self.K_B = input['K_B']
        self.crop_A = input['crop_A']
        self.crop_B = input['crop_B']
        
    def forward(self, stage='train'):

        holl_mask = torch.where(self.real_depth<=self.border, torch.tensor(1).float().to(self.real_depth.device), torch.tensor(0).float().to(self.real_depth.device))
        right_mask = holl_mask.clone()
        right_mask[:,:,:-1,:]+=right_mask[:,:,1:,:].clone()     
        right_mask[:,:,1:,:]+=right_mask[:,:,:-1,:].clone()
        right_mask[:,:,:,:-1]+=right_mask[:,:,:,1:].clone()
        right_mask[:,:,:,1:]+=right_mask[:,:,:,:-1].clone()
        
        right_mask = torch.where(right_mask<1, torch.tensor(1).float().to(self.real_depth.device), torch.tensor(0).float().to(right_mask.device))
        mask = right_mask 
        self.real_hole_mask = holl_mask
        self.real_mask = right_mask
        
        
        holl_mask = torch.where(self.syn_depth<=self.border, torch.tensor(1).float().to(self.syn_depth.device), torch.tensor(0).float().to(self.syn_depth.device))
        right_mask = holl_mask.clone()
        right_mask[:,:,:-1,:]+=right_mask[:,:,1:,:].clone()
        right_mask[:,:,1:,:]+=right_mask[:,:,:-1,:].clone()
        right_mask[:,:,:,:-1]+=right_mask[:,:,:,1:].clone()
        right_mask[:,:,:,1:]+=right_mask[:,:,:,:-1].clone()
    
        right_mask = torch.where(right_mask<1, torch.tensor(1).float().to(self.real_depth.device), torch.tensor(0).float().to(right_mask.device))
        mask = right_mask 
        self.syn_mask = right_mask        
        
        del right_mask
        del mask
        
#         print(self.syn_depth.shape)
        if self.opt.use_image_for_trans:
            
            self.syn2real_depth = self.netG_A_d(self.syn_depth, self.syn_image)
            syn_depth = self.syn2real_depth
            real_depth = self.real_depth
            if self.opt.use_rec_as_real_input:
                r2s = self.netG_B_d(torch.cat([self.real_depth, self.real_image], dim=1))
                self.real_rec = self.netG_A_d(torch.cat([r2s, self.real_image], dim=1))
                real_depth = self.real_rec
        else:
            self.syn2real_depth = self.netG_A_d(self.syn_depth)
            self.rec_syn = self.netG_B_d(self.syn2real_depth)
            syn_depth = torch.where(self.syn2real_depth <self.border, torch.tensor(-1).float().to(self.syn_depth.device), self.rec_syn) 
            real_depth = self.real_depth
            
            if self.opt.use_rec_as_real_input:
                self.r2s = self.netG_B_d(self.real_depth)
                self.real_rec = self.netG_A_d(self.r2s)
                real_depth = torch.where(self.real_mask<0.05, torch.tensor(-1).float().to(real_depth.device), self.r2s) 
            
#         print('00000000000000000000000000000000000000000000000000000000000')
#         print(self.netI2D_features)
        if stage =='train':
            image_features_real = self.netI2D_features(F.interpolate(self.real_image, size = (self.opt.crop_size_h, self.opt.crop_size_w) , mode='bicubic'))
            self.real_depth_by_image = self.netImage2Depth(image_features_real)
            self.real_depth_by_image = F.interpolate(self.real_depth_by_image, size = (self.opt.crop_size_h*2, self.opt.crop_size_w*2) , mode='bicubic')
            image_features_real = F.interpolate(image_features_real, size = (self.opt.crop_size_h*2, self.opt.crop_size_w*2) , mode='bicubic')
            
            
            image_features_syn = self.netI2D_features(F.interpolate(self.syn_image, size = (self.opt.crop_size_h, self.opt.crop_size_w) , mode='bicubic'))
            self.syn_depth_by_image = self.netImage2Depth(image_features_syn)
            self.syn_depth_by_image = F.interpolate(self.syn_depth_by_image, size = (self.opt.crop_size_h*2, self.opt.crop_size_w*2) , mode='bicubic')
            image_features_syn = F.interpolate(image_features_syn, size = (self.opt.crop_size_h*2 ,self.opt.crop_size_w*2) , mode='bicubic')
        else:    
            image_features_real = self.netI2D_features(F.interpolate(self.real_image, size = (self.opt.crop_size_h,self.opt.crop_size_w) , mode='bicubic'))
            self.real_depth_by_image = self.netImage2Depth(image_features_real)
            self.real_depth_by_image = F.interpolate(self.real_depth_by_image, size = (self.opt.crop_size_h*2,self.opt.crop_size_w*2) , mode='bicubic')
            image_features_real = F.interpolate(image_features_real, size = (self.opt.crop_size_h*2,self.opt.crop_size_w*2) , mode='bicubic')
#             print(self.real_depth_by_image.shape)
        

#         self.syn_depth_by_image = self.syn_image
#         self.real_depth_by_image = self.real_image
               

        if self.opt.use_masked:
            out = []
            n = 60 if stage =='train' else 11
            p = 0.95 if stage == 'train' else 0
            for i in range(self.real_mask.shape[0]):
                number = np.random.randint(10,n)
                xs = np.random.choice(self.real_mask.shape[3], number, replace=False)
                ys = np.random.choice(self.real_mask.shape[2], number, replace=False)
                sizes_x = np.random.randint(self.real_mask.shape[3]//150, self.real_mask.shape[3]//10, number)*np.random.binomial(1, p)
                sizes_y = np.random.randint(self.real_mask.shape[2]//150, self.real_mask.shape[2]//10, number)*np.random.binomial(1, p)
                ones = np.ones_like(self.real_mask.cpu().numpy()[0][0])
                
                for x, y, s_x, s_y in zip(xs,ys,sizes_x, sizes_y):
                    ones[y:y+s_y, x:x+s_x]=0
                    
                ones = np.where((self.real_mask[i][0].cpu().numpy()>0.05) & (ones<0.05), 0, 1)
                out.append(np.expand_dims(ones, 0))
                
            self.gt_mask_real = torch.from_numpy(np.array(out)).to(self.real_depth.device)
            self.depth_masked = torch.where(self.gt_mask_real<0.05, torch.tensor(-1).float().to(real_depth.device), real_depth)
#             depth_r_inp =  torch.cat([self.depth_masked, self.real_depth_by_image], dim=1) 
#         else:    
#             depth_r_inp =  torch.cat([real_depth, self.real_depth_by_image], dim=1) 
            

 
        if self.opt.use_masked:
            out = []
            n = 60 if stage =='train' else 11
            p = 0.90 if stage == 'train' else 0
            for i in range(syn_depth.shape[0]):
                number = np.random.randint(10,n)
                xs = np.random.choice(syn_depth.shape[3], number, replace=False)
                ys = np.random.choice(syn_depth.shape[2], number, replace=False)
                sizes_x = np.random.randint(syn_depth.shape[3]//150, syn_depth.shape[3]//10, number)*np.random.binomial(1, p)
                sizes_y = np.random.randint(syn_depth.shape[2]//150, syn_depth.shape[2]//10, number)*np.random.binomial(1, p)
                ones = np.ones_like(syn_depth.detach().cpu().numpy()[0][0])
                
                for x, y, s_x, s_y in zip(xs,ys,sizes_x, sizes_y):
                    ones[y:y+s_y, x:x+s_x]=0
                    
                ones = np.where((self.syn_mask[i][0].cpu().numpy()>0.05) & (ones<0.05), 0, 1)
                out.append(np.expand_dims(ones, 0))
                
            self.gt_mask_syn = torch.from_numpy(np.array(out)).to(syn_depth.device)

            self.syn2real_depth_masked = torch.where(self.gt_mask_syn<0.05, torch.tensor(-1).float().to(syn_depth.device), syn_depth)
        else:
            self.syn2real_depth_masked =  syn_depth

        
        del ones
        del out
        del syn_depth
        del real_depth


        feat_real_depth = self.netDepth_f(torch.cat([self.depth_masked, self.real_depth_by_image], dim=1))
#         print(image_features_real.shape, self.depth_masked.shape, self.real_depth_by_image.shape, self.real_image.shape)
        self.pred_real_depth_hr = self.netTask(torch.cat([image_features_real, feat_real_depth, torch.cat([self.depth_masked, self.real_depth_by_image], dim=1), self.real_image], dim=1)) 

        

        
        del image_features_real

        
        if stage=='train':
            feat_syn_depth = self.netDepth_f(torch.cat([self.syn2real_depth_masked, self.syn_depth_by_image], dim=1))
            self.pred_syn_depth = self.netTask(torch.cat([image_features_syn,feat_syn_depth, torch.cat([self.syn2real_depth_masked, self.syn_depth_by_image], dim=1), self.syn_image], dim=1)) 
            del image_features_syn
            self.pred_real_depth = F.interpolate(self.pred_real_depth_hr, size = (self.opt.crop_size_h,self.opt.crop_size_w) , mode='bicubic')
            syn_mean = torch.mean(self.syn_depth*self.syn_mask) 
            syn_pred_mean = torch.mean(self.pred_syn_depth*self.syn_mask) 

            self.loss_syn_mean_diff = (syn_mean-syn_pred_mean).cpu().detach().numpy()
            self.loss_mean_of_abs_diff_syn =  np.mean(np.absolute((self.syn_depth*self.syn_mask-self.pred_syn_depth*self.syn_mask).cpu().detach().numpy()))

            real_mean = torch.mean(F.interpolate(self.real_depth, size = (self.opt.crop_size_h,self.opt.crop_size_w) , mode='bicubic')*F.interpolate(self.real_mask, size = (self.opt.crop_size_h,self.opt.crop_size_w) , mode='bicubic')) 
            real_pred_mean = torch.mean(self.pred_real_depth*F.interpolate(self.real_mask, size = (self.opt.crop_size_h,self.opt.crop_size_w) , mode='bicubic')) 

            self.loss_real_mean_diff  = (real_mean-real_pred_mean).cpu().detach().numpy() 
            self.loss_mean_of_abs_diff_real =  np.mean(np.absolute((F.interpolate(self.real_depth, size = (self.opt.crop_size_h,self.opt.crop_size_w) , mode='bicubic')*F.interpolate(self.real_mask, size = (self.opt.crop_size_h, self.opt.crop_size_w) , mode='bicubic')-self.pred_real_depth*F.interpolate(self.real_mask, size = (self.opt.crop_size_h, self.opt.crop_size_w) , mode='bicubic')).cpu().detach().numpy()))  
        
        
        post = lambda img: np.clip((img.permute(1,2,0).numpy()+1)/2,0,1)[:,:,0]
        if self.opt.save_all and stage=='test':
            
            batch_size = len(self.A_paths)

            for i in range(batch_size):
                path = str(self.B_paths[i])
                path = path.split('/')[-1].split('.')[0]
                file = f'{self.opt.save_image_folder}{path}.png'
                out_np = post(self.pred_real_depth_hr[i][:,32:-32,:].cpu().detach())*5100
                
                imageio.imwrite(file, out_np.astype(np.uint16))
                out_f = imageio.imread(file)
        


    def backward_G(self, back=True):
        """Calculate the loss for generators G_A and G_B"""
        
        self.real_mask = F.interpolate(self.real_mask, size = (self.opt.crop_size_h, self.opt.crop_size_w) , mode='nearest')
        self.real_hole_mask = F.interpolate(self.real_hole_mask, size = (self.opt.crop_size_h, self.opt.crop_size_w) , mode='nearest')
        self.real_depth = F.interpolate(self.real_depth, size = (self.opt.crop_size_h, self.opt.crop_size_w) , mode='bicubic')
        self.real_image = F.interpolate(self.real_image, size = (self.opt.crop_size_h, self.opt.crop_size_w) , mode='bicubic')
        self.real_depth_by_image = F.interpolate(self.real_depth_by_image, size = (self.opt.crop_size_h, self.opt.crop_size_w) , mode='bicubic')

        if self.opt.norm_loss:
            calc_norm = SurfaceNormals()
            self.norm_syn = calc_norm(self.syn_depth)*100
            self.norm_syn2real = calc_norm(self.syn2real_depth_masked)*100
            self.norm_syn_pred = calc_norm(self.pred_syn_depth)*100
            self.norm_real = calc_norm(self.real_depth)*100
            self.norm_real_pred = calc_norm(self.pred_real_depth)*100
            self.norm_real_pred_hr = calc_norm(self.pred_real_depth_hr)*100
            self.loss_tv_syn_norm_old = tv_loss(self.norm_syn_pred)*(10**-7)
            self.loss_tv_real_norm_old = tv_loss(self.norm_real_pred_hr)*(10**-7)
            self.loss_syn_norms_old = self.criterion_task(self.norm_syn, self.norm_syn_pred) 

        a = self.syn2real_depth_masked<self.border 
        b = self.gt_mask_syn<0.1
        c = a + b
#         print(tens.shape, tenss.shape)
        self.mask_syn_add_holes = torch.where(c, torch.tensor(1).float().to(self.syn2real_depth_masked.device), torch.tensor(0).float().to(self.syn2real_depth_masked.device))
        del a
        del b
        del c
        
        

        if self.opt.norm_loss:

            calc_norm = SurfaceNormals_new()
            self.norm_syn = calc_norm(self.syn_depth, self.K_A, self.crop_A)
            self.norm_syn2real = calc_norm(self.syn2real_depth_masked, self.K_A, self.crop_A)
            self.norm_syn_pred = calc_norm(self.pred_syn_depth, self.K_A, self.crop_A)
            calc_norm = SurfaceNormals_new()
            self.norm_real = calc_norm(self.real_depth, self.K_B, self.crop_B)
            self.norm_real_pred = calc_norm(self.pred_real_depth, self.K_B, self.crop_B)
            self.norm_real_pred_hr = calc_norm(self.pred_real_depth_hr, self.K_A, self.crop_A)
            self.loss_tv_syn_norm = tv_loss(self.norm_syn_pred)*(10**-7)
            self.loss_tv_real_norm = tv_loss(self.norm_real_pred)*(10**-7)
            self.loss_syn_norms = self.criterion_task_2(self.norm_syn*self.syn_mask, self.norm_real_pred_hr*self.syn_mask)
            self.loss_syn_norms_holes = self.criterion_task(self.norm_syn*self.syn_mask*self.mask_syn_add_holes, self.norm_syn_pred*self.syn_mask*self.mask_syn_add_holes)       
            
        if self.opt.use_edge:
            canny = CannyFilter(use_cuda=True)
            self.edges_real = canny(self.real_image)[3]
            self.edges_syn = canny(self.syn_image)[3]
            self.edges_real_pred = canny(self.pred_real_depth)[3]
            self.edges_syn_pred = canny(self.pred_syn_depth)[3]

 
        self.loss_holes_syn = self.criterion_task(self.syn_depth*self.syn_mask*self.mask_syn_add_holes, self.pred_syn_depth*self.syn_mask*self.mask_syn_add_holes) 
        self.loss_holes_syn_l2 = self.criterion_task_2(self.syn_depth*self.syn_mask*self.mask_syn_add_holes, self.pred_syn_depth*self.syn_mask*self.mask_syn_add_holes)*5 
        
        self.mask_syn_add_holes = self.pred_syn_depth*self.syn_mask*self.mask_syn_add_holes
 
        self.loss_task_syn = self.criterion_task(self.syn_depth*self.syn_mask, self.pred_syn_depth*self.syn_mask) 
        self.loss_task_real_by_depth = self.criterion_task(self.real_depth*self.real_mask, self.pred_real_depth*self.real_mask) 
        self.loss_task_real_by_image = self.criterion_task(F.interpolate(self.syn_depth, size = (self.opt.crop_size_h,self.opt.crop_size_w) , mode='nearest')*self.real_hole_mask, self.pred_real_depth*self.real_hole_mask) 
            
        # combined loss and calculate gradients
        self.loss_G = self.loss_task_syn*self.opt.w_syn_l1 + self.loss_holes_syn*self.opt.w_syn_holes + self.opt.w_syn_holes*self.loss_holes_syn_l2 + self.loss_task_real_by_depth*self.opt.w_real_l1_d + self.loss_task_real_by_image*self.opt.w_real_l1_i + self.loss_tv_syn_norm*1 + self.loss_syn_norms_holes*self.opt.w_syn_norm*5 + self.loss_tv_real_norm*2 + self.loss_syn_norms_old*self.opt.w_syn_norm*5 + self.loss_tv_real_norm_old*2 + self.loss_tv_syn_norm_old*1
       
        if self.opt.use_masked:
            self.mask_real_add_holes = torch.where(self.gt_mask_real>0.1, torch.tensor(0).float().to(self.pred_real_depth.device), torch.tensor(1).float().to(self.pred_real_depth.device))
            self.mask_real_add_holes = F.interpolate(self.mask_real_add_holes, size = (self.opt.crop_size_h,self.opt.crop_size_w) , mode='nearest')
            self.loss_holes_real = self.criterion_task(self.real_depth*self.mask_real_add_holes, self.pred_real_depth*self.mask_real_add_holes) 
            self.loss_holes_real_l2 = self.criterion_task_2(self.real_depth*self.mask_real_add_holes, self.pred_real_depth*self.mask_real_add_holes)*5 
            self.loss_G+=self.loss_holes_real*self.opt.w_real_holes + self.loss_holes_real_l2*self.opt.w_real_holes
            self.mask_real_add_holes = self.pred_real_depth*self.mask_real_add_holes

        if self.opt.use_D:
            self.loss_G+=self.loss_G_pred*self.opt.w_syn_adv
        
        if self.opt.norm_loss:
            self.loss_G+=self.loss_syn_norms*self.opt.w_syn_norm
            
        if self.opt.use_smooth_loss:
#             print(self.pred_real_depth.shape, self.real_image.shape)
            self.loss_smooth = get_smooth_weight(self.pred_real_depth, self.real_image, 3) 
            self.loss_G+=self.loss_smooth*self.opt.w_smooth 
            
        if self.opt.use_edge:
            self.loss_edge_real = self.criterion_task(self.edges_real, self.edges_real_pred) 
            self.loss_edge_syn = self.criterion_task(self.edges_syn, self.edges_syn_pred) 
            print(self.loss_edge_real, self.loss_edge_syn)
            self.loss_G+=self.loss_edge_syn*self.opt.w_edge_s + self.loss_edge_real*self.opt.w_edge_r
            
      
        self.loss_G *= self.opt.scale_G
        if back:
            self.loss_G.backward()
            

    def optimize_parameters(self, iters, fr=1):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.

        self.set_requires_grad([self.netG_A_d, self.netI2D_features, self.netImage2Depth ], False)  # Ds require no gradients when optimizing Gs
    

        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        

            
            
    def calculate(self, stage='test'):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward(stage)      # compute fake images and reconstruction images.
#         self.backward_G(back=False)             # calculate gradients for G_A and G_B

