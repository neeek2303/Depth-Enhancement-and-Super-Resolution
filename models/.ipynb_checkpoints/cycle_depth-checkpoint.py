import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn
import numpy as np


class CycleGANModel_depth(BaseModel):
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
        self.loss_names = ['task_syn', 'task_real', 'D_depth', 'syn_mean_diff', 'real_mean_diff', 'holes']
        if opt.print_mean:
            self.loss_names = ['syn_mean_diff', 'real_mean_diff', 'mean_of_abs_diff_syn', 'mean_of_abs_diff_real', 'L1_syn', 'L1_real']
            
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['syn_image', 'syn_depth', 'syn2real_depth',   'pred_syn_depth']
        visual_names_B = ['real_image', 'real_depth', 'pred_real_depth', 'mask']
        


        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        
        
        self.model_names = ['D_depth', 'G_A_d', 'Image_f', 'Depth_f', 'Task']
        
        
        # Define networks 
        
        ### part_1
        self.netD_depth = networks.define_D(1, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) 
        translation_input_d = 4 if opt.use_image_for_trans else 1 
        self.netG_A_d = networks.define_G(translation_input_d, 1, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.replace_transpose)
        if opt.use_rec_as_real_input or opt.use_image_for_trans:
            self.netG_B_d = networks.define_G(translation_input_d, 1, opt.ngf, opt.netG, opt.norm,
                                                not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.replace_transpose)
        
        ### part_2 
        self.netImage_f = networks.define_G(3, opt.Imagef_outf, opt.Imagef_basef, opt.Imagef_type, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.replace_transpose, n_down = opt.Imagef_ndown)
        
        self.netDepth_f = networks.define_G(1, opt.Depthf_outf, opt.Depthf_basef, opt.Depthf_type, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.replace_transpose, n_down = opt.Depthf_ndown)  
        
        task_input_features = opt.Imagef_outf + opt.Depthf_outf
        self.netTask = networks.define_G(task_input_features, 1, opt.Task_basef, opt.Task_type, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.replace_transpose, n_down = opt.Task_ndown)
        

        self.loss_L1_syn=0
        self.loss_L1_real=0

        
        if self.isTrain:
            self.fake_depth_pool = ImagePool(opt.pool_size) 
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
                
            # define loss functions

            self.criterion_task = torch.nn.L1Loss()
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netDepth_f.parameters(), self.netImage_f.parameters(), self.netTask.parameters()), lr=opt.lr)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_depth.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
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
        
        
        if self.opt.use_image_for_trans:
            self.syn2real_depth = self.netG_A_d(torch.cat([self.syn_image, self.syn_depth], dim=1) )
            syn_depth = self.syn2real_depth
            real_depth = self.real_depth
            if self.opt.use_rec_as_real_input:
                self.real_rec = self.netG_A_d(self.netG_B_d(torch.cat([self.real_image, self.real_depth], dim=1) ))
                real_depth = self.real_rec
        else:
            self.syn2real_depth = self.netG_A_d(self.syn_depth)
            syn_depth = self.syn2real_depth
            real_depth = self.real_depth
            
            if opt.use_rec_as_real_input:
                self.real_rec = self.netG_A_d(self.netG_B_d(self.real_depth))
                real_depth = self.real_rec
            

        self.pred_syn_depth = self.netTask(torch.cat([self.netImage_f(self.syn_image), self.netDepth_f(syn_depth)], dim=1))
        self.pred_real_depth = self.netTask(torch.cat([self.netImage_f(self.real_image), self.netDepth_f(real_depth)], dim=1))
        
        
        
        
        
        syn_mean = torch.mean(self.syn_depth) 
        syn_pred_mean = torch.mean(self.pred_syn_depth) 
        
        self.loss_syn_mean_diff = (syn_mean-syn_pred_mean).cpu().detach().numpy()
        
        self.loss_mean_of_abs_diff_syn =  np.mean(np.absolute((self.syn_depth-self.pred_syn_depth).cpu().detach().numpy()))
        
    
        self.mask = torch.where(self.real_depth<-0.97, torch.tensor(0).float().to(self.real_depth.device), torch.tensor(1).float().to(self.real_depth.device))
        mask = self.mask 
        real_mean = torch.mean(self.real_depth*mask) 
        real_pred_mean = torch.mean(self.pred_real_depth*mask) 
        
        self.loss_real_mean_diff  = (real_mean-real_pred_mean).cpu().detach().numpy() 
        self.loss_mean_of_abs_diff_real =  np.mean(np.absolute((self.real_depth*mask-self.pred_real_depth*mask).cpu().detach().numpy()))
        
        
#         self.loss_L1_syn+= self.criterion_task(self.syn_depth, self.pred_syn_depth)*4/5000
# #         print(torch.min(self.syn_depth), torch.min(self.pred_syn_depth), torch.max(self.syn_depth), torch.max(self.pred_syn_depth))
#         mask = torch.where(self.real_depth<-0.97, torch.tensor(0).float().to(self.real_depth.device), torch.tensor(1).float().to(self.real_depth.device))
#         self.loss_L1_real+= self.criterion_task(self.real_depth*mask, self.pred_real_depth*mask)*4/5000
            
            

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
        fake_depth = self.fake_depth_pool.query(self.pred_syn_depth)
        self.loss_D_depth = self.backward_D_basic(self.netD_depth, self.syn_depth , fake_depth, back=back)

    def backward_G(self, back=True):
        """Calculate the loss for generators G_A and G_B"""

        
        # Forward cycle loss || G_B(G_A(A)) - A||
        mask_syn = torch.where(self.syn2real_depth<-0.97, torch.tensor(1).float().to(self.syn2real_depth.device), torch.tensor(0).float().to(self.syn2real_depth.device))
        self.loss_task_syn= self.criterion_task(self.syn_depth, self.pred_syn_depth) 
        self.loss_holes = self.criterion_task(self.syn_depth*mask_syn, self.pred_syn_depth*mask_syn) 
        
        
        self.loss_G_pred = self.criterionGAN(self.netD_depth(self.pred_syn_depth), True)  
        mask_real = torch.where(self.real_depth<-0.97, torch.tensor(0).float().to(self.real_depth.device), torch.tensor(1).float().to(self.real_depth.device))
        self.loss_task_real = self.criterion_task(self.real_depth*mask_real, self.pred_real_depth*mask_real) 
        

        # combined loss and calculate gradients
        self.loss_G = self.loss_task_syn + self.loss_G_pred*self.opt.w_syn_adv + self.loss_task_real*self.opt.w_real_l1 + self.loss_holes*self.opt.w_holles
#         print(self.loss_G)
        if back:
            self.loss_G.backward()

    def optimize_parameters(self, iters, fr=1):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
#         self.set_requires_grad([self.netG_A , self.netG_A_d, self.netD_depth], False)  # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netG_A_d, self.netD_depth], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        if iters%fr == 0:
            self.set_requires_grad([self.netD_depth], True)  # Ds require no gradients when optimizing Gs
            self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
            self.backward_D_depth()      # calculate gradients for D_A
            self.optimizer_D.step()  # update D_A and D_B's weights
            
            
    def calculate(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        self.backward_G(back=False)             # calculate gradients for G_A and G_B
        self.backward_D_depth(back=False)      # calculate gradients for D_A
