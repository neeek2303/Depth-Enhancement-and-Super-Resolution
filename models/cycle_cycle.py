import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import imageio

class CycleGANModel(BaseModel):
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
        self.opt = opt
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        if self.opt.use_rec_iou_error:
            self.loss_names+=['iou_rec']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['syn_image', 'syn_depth', 'fake_B', 'rec_A']
        visual_names_B = ['real_image', 'real_depth', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']
        
        
        input_channels = 1
        if self.opt.cat:
            input_channels = 4
        
        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(input_channels, 1, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.replace_transpose)
        self.netG_B = networks.define_G(input_channels, 1, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.replace_transpose)
        

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(1, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(1, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
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
        
        self.A_paths = input['A_paths']
        self.B_paths = input['B_paths']
    
        
        self.input_syn = self.syn_depth
        self.input_real = self.real_depth


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        
        
        
        if self.opt.cat:
            self.fake_B = self.netG_A(torch.cat([self.syn_depth, self.syn_image], dim=1))  # G_A(A)
            self.fake_A = self.netG_B(torch.cat([self.real_depth, self.real_image], dim=1))  # G_B(B)
        else:
            self.fake_B = self.netG_A(self.syn_depth)  # G_A(A)
            self.fake_A = self.netG_B(self.real_depth)  # G_B(B)
        

        if self.opt.cat:
            self.idt_A = self.netG_A(torch.cat([self.real_depth, self.real_image], dim=1))
            self.idt_B = self.netG_B(torch.cat([self.syn_depth, self.syn_image], dim=1))
        else:
            self.idt_A = self.netG_A(self.real_depth)
            self.idt_B = self.netG_B(self.syn_depth)     
                     
        
        if self.opt.cat:
            self.rec_A = self.netG_B(torch.cat([self.fake_B, self.syn_image], dim=1))   # G_B(G_A(A))
            self.rec_B = self.netG_A(torch.cat([self.fake_A, self.real_image], dim=1))   # G_A(G_B(B))
        else:
            self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
            self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))        
        
        
        
        
        


            batch_size = len(self.A_paths)

            for i in range(batch_size):
                path = str(self.A_paths[i])
                path = path.split('/')[-1].split('.')[0]
                file = f'/root/gans_depth/pytorch-CycleGAN-and-pix2pix/fakeB_cycle/{path}.png'
                imageio.imwrite(file, (post(self.fake_B.cpu().detach())*8000).astype(np.uint16))
                
                
            for i in range(batch_size):
                path = str(self.B_paths[i])
                path = path.split('/')[-1].split('.')[0]
                file = f'/root/gans_depth/pytorch-CycleGAN-and-pix2pix/BAB_cycle/{path}.png'
                imageio.imwrite(file, (post(self.rec_A.cpu().detach())*8000).astype(np.uint16))    
                
            for i in range(batch_size):
                path = str(self.A_paths[i])
                path = path.split('/')[-1].split('.')[0]
                file = f'/root/gans_depth/pytorch-CycleGAN-and-pix2pix/ABA_cycle/{path}.png'
                imageio.imwrite(file, (post(self.rec_B.cpu().detach())*8000).astype(np.uint16))        
     #             np.save(file, post(self.fake_B.cpu().detach()))  




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

    def backward_D_A(self, back=True):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_depth, fake_B, back)

    def backward_D_B(self, back=True):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.syn_depth, fake_A, back)

    def backward_G(self, back=True):
        """Calculate the loss for generators G_A and G_B"""
        use_rec_masks = self.opt.use_rec_masks
        use_idt_masks = self.opt.use_idt_masks
        
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
                
            if use_idt_masks:
                mask_real = torch.where(self.real_depth<-0.97, torch.tensor(0).float().to(self.real_depth.device),  torch.tensor(1).float().to(self.real_depth.device))
                mask_idn = torch.where(self.idt_A<-0.97, torch.tensor(0).float().to(self.idt_A.device), torch.tensor(1).float().to(self.idt_A.device))

                self.loss_idt_A = self.criterionIdt(self.idt_A*mask_real*mask_idn, self.real_depth*mask_real*mask_idn) * lambda_B * lambda_idt
            
            else:
                self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_depth) * lambda_B * lambda_idt
                
            # G_B should be identity if real_A is fed: ||G_B(A) - A||

            
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.syn_depth) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.syn_depth) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        if use_rec_masks:
            mask_real = torch.where(self.real_depth<-0.97, torch.tensor(0).float().to(self.real_depth.device), torch.tensor(1).float().to(self.real_depth.device))
            mask_rec = torch.where(self.rec_B<-0.97, torch.tensor(0).float().to(self.rec_B.device), torch.tensor(1).float().to(self.rec_B.device))
            self.loss_cycle_B = self.criterionCycle(self.rec_B*mask_real*mask_rec, self.real_depth*mask_real*mask_rec) * lambda_B
        else:
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_depth) * lambda_B
        
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B 
        
        
        if self.opt.use_rec_iou_error:
            real_holes = torch.where(self.real_depth<-0.99, torch.tensor(1).float().to(self.real_depth.device), torch.tensor(0).float().to(self.real_depth.device))
            rec_holes = torch.where(self.rec_B<-0.99, torch.tensor(1).float().to(self.rec_B.device), torch.tensor(0).float().to(self.rec_B.device))
            mask_and = torch.where(((real_holes==1) & (rec_holes==1)), torch.tensor(1).float().to(self.real_depth.device),   torch.tensor(0).float().to(self.real_depth.device))
            SMOOTH = 1e-6
#             print(torch.sum(rec_holes),  torch.sum(real_holes)  ,  torch.sum(mask_and), torch.min(real_holes)     )
            intersection =  torch.sum(-self.rec_B*mask_and, dim=(2,3))
            union =  torch.sum(-self.rec_B*rec_holes, dim=(2,3) ) + torch.sum(-self.real_depth*real_holes, dim=(2,3)) - intersection
            iou = (intersection + SMOOTH) / (union + SMOOTH)
#             print( intersection, torch.sum(-self.rec_B*rec_holes, dim=(2,3) ),  union)
            
            mean_iou = torch.mean(iou)
#             print(iou.shape)
#             print(mean_iou)
            self.loss_iou_rec = mean_iou
            if self.opt.back_rec_iou_error:
                self.loss_G += self.loss_iou_rec*self.opt.iou_error_weight
        
        
        # combined loss and calculate gradients
        
        if back:
            self.loss_G.backward()

    def optimize_parameters(self, iters, fr=1):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        if iters%fr == 0:
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
            self.backward_D_A()      # calculate gradients for D_A
            self.backward_D_B()      # calculate graidents for D_B
            self.optimizer_D.step()  # update D_A and D_B's weights
    
    
    def calculate(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        self.backward_G(back=False)             # calculate gradients for G_A and G_B
        self.backward_D_A(back=False)      # calculate gradients for D_A
        self.backward_D_B(back=False)      # calculate graidents for D_B