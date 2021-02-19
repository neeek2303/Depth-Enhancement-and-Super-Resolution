import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn
import numpy as np

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

class I2DModel(BaseModel):
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
        self.loss_names = ['task_syn', 'task_real']
        if opt.norm_loss:
            self.loss_names+=['syn_norms']
        if self.opt.use_D:
            self.loss_names+=['G_pred', 'D_depth', 'G_pred_r']
        
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['syn_image', 'syn_depth', 'pred_syn_depth']
        visual_names_B = ['real_image', 'real_depth', 'pred_real_depth']
    

         # combine visualizations for A and B
        
        
        self.model_names = ['Image_f', 'Task']

        
        if opt.use_D:
            self.model_names +=['D_depth']
            
        if opt.norm_loss:
            visual_names_A += ['norm_syn','norm_syn_pred']
            visual_names_B += [ 'norm_real','norm_real_pred']
#         print(visual_names_A)
        
        
        self.visual_names = visual_names_A + visual_names_B 
        
        ### part_2 
        self.netImage_f = networks.define_G(3, opt.Imagef_outf, opt.Imagef_basef, opt.Imagef_type, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.replace_transpose, n_down = opt.Imagef_ndown)
        
#         self.netImage_f_syn = networks.define_G(3, opt.Imagef_outf, opt.Imagef_basef, opt.Imagef_type, opt.norm,
#                                         not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.replace_transpose, n_down = opt.Imagef_ndown)
        
        task_input_features = opt.Imagef_outf
        
#         if opt.use_D:
#             self.netD_depth = networks.define_D(task_input_features, opt.ndf, opt.netD,
#                                                 opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) 
        
        self.netTask = networks.define_G(task_input_features, 1, opt.Task_basef, opt.Task_type, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.replace_transpose, n_down = opt.Task_ndown)
        

        self.loss_L1_syn=0
        self.loss_L1_real=0
        self.loss_D_depth = 0 
        
        if self.isTrain:
            self.fake_depth_pool = ImagePool(opt.pool_size) 
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
                
            # define loss functions

            self.criterion_task = torch.nn.L1Loss()
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain( self.netTask.parameters()), lr=opt.lr)
            self.optimizers.append(self.optimizer_G)
            if self.opt.use_D:
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_depth.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)



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
        
        

            
        self.features_syn = self.netImage_f(self.syn_image)
        self.features_real =  self.netImage_f(self.real_image)
        self.pred_syn_depth = self.netTask(self.features_syn)
        self.pred_real_depth = self.netTask(self.features_real)
        
    def backward_D_basic(self, netD, real, fake, back=True):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
#         Real
        pred_real = netD(real.detach())
        loss_D_real = self.criterionGAN(pred_real, True)
#         Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        if back:
            loss_D.backward()
        return loss_D
    
    def backward_D_depth(self, back=True):
        """Calculate GAN loss for discriminator D_B"""
        fake = self.fake_depth_pool.query(self.features_real)
        self.loss_D_depth = self.backward_D_basic(self.netD_depth, self.features_syn, fake, back=back)       
            

    def backward_features(self, back=True):
        """Calculate the loss for generators G_A and G_B"""

        self.loss_G_pred = self.criterionGAN(self.netD_depth(self.features_real), True)
        self.loss_G_pred_r = self.criterionGAN(self.netD_depth(self.features_syn), True)
        self.loss_G_p = self.loss_G_pred *self.opt.w_syn_adv 
        if back:
            self.loss_G_pred.backward(retain_graph=True)            

    def backward_G(self, back=True):
        """Calculate the loss for generators G_A and G_B"""

        
        if self.opt.norm_loss:
            calc_norm = SurfaceNormals()
            self.norm_syn = calc_norm(self.syn_depth)
            self.norm_syn_pred = calc_norm(self.pred_syn_depth)
            self.norm_real = calc_norm(self.real_depth)
            self.norm_real_pred = calc_norm(self.pred_real_depth)
            self.loss_syn_norms = self.criterion_task(self.norm_syn, self.norm_syn_pred) 
        
        mask_syn= torch.where(self.syn_depth<-0.97, torch.tensor(0).float().to(self.syn_depth.device), torch.tensor(1).float().to(self.syn_depth.device))
        self.loss_task_syn= self.criterion_task(self.syn_depth*mask_syn, self.pred_syn_depth*mask_syn) 
        
        mask_real = torch.where(self.real_depth<-0.97, torch.tensor(0).float().to(self.real_depth.device), torch.tensor(1).float().to(self.real_depth.device))
        self.loss_task_real = self.criterion_task(self.real_depth*mask_real, self.pred_real_depth*mask_real) 
        
        # combined loss and calculate gradients
        self.loss_G = self.loss_task_syn*self.opt.w_syn_l1  + self.loss_task_real*self.opt.w_real_l1 
        
        if self.opt.norm_loss:
            self.loss_G+=self.loss_syn_norms*self.opt.w_syn_norm
        
#         if self.opt.use_D:
#             self.loss_G_pred = self.criterionGAN(self.netD_depth(self.features_syn), True)  
#             self.loss_G+=self.loss_G_pred*self.opt.w_syn_adv
            
        self.loss_G *= self.opt.scale_G
#         print(self.loss_G)
        if back:
            self.loss_G.backward()

    def optimize_parameters(self, iters, fr=700):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
#         print('aaaaa')
        self.forward() 
        self.optimizer_G.zero_grad()
        if self.opt.use_D:
            self.set_requires_grad([self.netD_depth], False)
            self.backward_features()  
        self.backward_G()             
        self.optimizer_G.step()       
        if self.opt.use_D:
            if (iters%(fr*self.opt.batch_size)==0) or (iters<800):
                print(iters,fr*self.opt.batch_size, iters%(fr*self.opt.batch_size))
                self.set_requires_grad([self.netD_depth], True)  
                self.optimizer_D.zero_grad()   
                self.backward_D_depth()      
                self.optimizer_D.step()  
            
            
    def calculate(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        self.backward_G(back=False)             # calculate gradients for G_A and G_B

