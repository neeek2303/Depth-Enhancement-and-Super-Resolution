import itertools
from .base_model import BaseModel
from . import translation_network
from util import util
import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace


class TranslationModel(BaseModel, nn.Module):
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--l_cycle_A_begin', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--l_cycle_A_end', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--l_cycle_B_begin', type=float, default=5.0, help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--l_cycle_B_end', type=float, default=5.0, help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--l_identity', type=float, default=1, help='identical loss')
        parser.add_argument('--l_normal', type=float, default=1., help='weight for normals cycle loss')
        parser.add_argument('--l_reconstruction_semantic', type=float, default=0.0, help='weight for reconstruction loss')
        parser.add_argument('--l_depth_A_begin', type=float, default=5.0, help='start of depth range loss')
        parser.add_argument('--l_depth_A_end', type=float, default=0.0, help='finish of depth range loss')
        parser.add_argument('--l_depth_B_begin', type=float, default=5.0, help='start of depth range loss')
        parser.add_argument('--l_depth_B_end', type=float, default=0.0, help='finish of depth range loss')
        parser.add_argument('--l_mean_A', type=float, default=0.0, help='weight for mean_dif for A')
        parser.add_argument('--l_mean_B', type=float, default=0.0, help='weight for mean_dif for B')
        parser.add_argument('--l_tv_A', type=float, default=0.0, help='weight for mean_dif for B')
        parser.add_argument('--l_max_iter', type=int, default=5000, help='max iter with big depth rec. loss')
        parser.add_argument('--l_num_iter', type=int, default=5000, help='max iter with big depth rec. loss')
        parser.add_argument('--num_iter_gen', type=int, default=3, help='iteration of gen per 1 iter of dis')
        parser.add_argument('--num_iter_dis', type=int, default=1, help='iteration of dis per 1 iter of gen')
        parser.add_argument('--no_idt_A', action="store_true", default=True, help='whether to not use identical loss on A') 
        parser.add_argument('--use_cycle_A', action='store_true', default=False, help='use cycle loss A2B2A')
        parser.add_argument('--use_cycle_B', action='store_true', default=True, help='use cycle loss B2A2B')
        parser.add_argument('--disc_for_normals', action='store_true', default=True, help='use disc on normals')
        parser.add_argument('--disc_for_depth', action='store_true', default=True, help='use disc on depth')
        parser.add_argument('--inp_B', type=str, default='img_depth', help='which input use for G_B. [depth | img_depth ]')
        parser.add_argument('--norm_d', type=str, default='none', help='instance normalization or batch normalization [instance | batch | none | group]')
        parser.add_argument('--w_decay_G', type=float, default=0.0001, help='weight decat L2 reguralization for Gen')
        return parser
    
    def __init__(self, opt):
        super(TranslationModel, self).__init__(opt)
        nn.Module.__init__(self)
        
        if self.isTrain:
            self.loss_names = ['G_A', 'G_B', 'depth_dif_A', 'depth_dif_B']
            if opt.l_mean_A > 0:
                self.loss_names.extend(['mean_dif_A'])
            if opt.l_mean_B > 0:
                self.loss_names.extend(['mean_dif_B'])
            if opt.use_cycle_A:
                self.loss_names.extend(['cycle_A', 'cycle_n_A'])
            if opt.use_cycle_B:
                self.loss_names.extend(['cycle_B', 'cycle_n_B'])
            if opt.disc_for_depth:
                self.loss_names.extend(['D_A_depth', 'D_B_depth'])
            if opt.disc_for_normals:
                self.loss_names.extend(['D_A_normal', 'D_B_normal'])
            if opt.l_identity > 0 :
                self.loss_names.extend(['idt_A', 'idt_B'])
            if opt.l_depth_A_begin > 0:
                self.loss_names.extend(['depth_range_A'])
            if opt.l_depth_B_begin > 0:
                self.loss_names.extend(['depth_range_B'])
            if opt.l_tv_A > 0:
                self.loss_names.extend(['tv_norm_A'])
        self.loss_names_test = ['depth_dif_A', 'depth_dif_B'] 
                
        self.visual_names = ['real_img_A', 'real_depth_A',
                              'real_img_B', 'real_depth_B',
                              'fake_depth_B',
                              'fake_depth_A',
                              'name_A', 'name_B']
        if opt.use_cycle_A:
            self.visual_names.extend(['rec_depth_A'])
        if opt.use_cycle_B:
            self.visual_names.extend(['rec_depth_B'])

        if self.isTrain:
            if opt.l_identity > 0:
                self.visual_names.extend(['idt_A', 'idt_B'])
        
        self.model_names = ['G_A', 'G_B']
        netG_A_opt = SimpleNamespace(ngf_img=32, ngf_depth=32, ngf=64, norm='group', dropout=False, init_type=opt.init_type, gpu_ids =opt.gpu_ids, input_nc_img=3, n_downsampling=2, use_semantic = False, n_blocks=9, upsampling_type="transpose", output_nc_depth=1, input_nc_depth=1)
        self.netG_A =  translation_network.define_Gen(netG_A_opt, input_type='img_depth')
            
        netG_B_opt = SimpleNamespace(ngf_img=32, ngf_depth=32, ngf=64, norm='group', dropout=False, init_type=opt.init_type, gpu_ids =opt.gpu_ids, input_nc_img=3, n_downsampling=2, use_semantic=False, n_blocks=9, upsampling_type="transpose", output_nc_depth=1, input_nc_depth=1)
        self.netG_B =  translation_network.define_Gen(netG_B_opt, input_type=opt.inp_B)
        
        if self.isTrain:
            if opt.disc_for_depth:
                self.model_names.extend(['D_A_depth', 'D_B_depth'])
            if opt.disc_for_normals:
                self.model_names.extend(['D_A_normal', 'D_B_normal'])
            self.disc = []
            if opt.disc_for_depth:
                self.netD_A_depth =  translation_network.define_D(opt, input_type = 'depth')
                self.netD_B_depth =  translation_network.define_D(opt, input_type = 'depth')
                self.disc.extend([self.netD_A_depth, self.netD_B_depth])
            if opt.disc_for_normals:
                self.netD_A_normal =  translation_network.define_D(opt, input_type = 'normal')
                self.netD_B_normal =  translation_network.define_D(opt, input_type = 'normal')
                self.disc.extend([self.netD_A_normal, self.netD_B_normal])
        
        self.criterionMaskedL1 =  translation_network.MaskedL1Loss() 
        self.criterionL1 = nn.L1Loss()
        if self.isTrain:

            self.criterionGAN =  translation_network.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionMeanDif =  translation_network.MaskedMeanDif()
            self.criterionCosSim =  translation_network.CosSimLoss()
            self.criterionMaskedCosSim =  translation_network.MaskedCosSimLoss()
            self.TVnorm =  translation_network.TV_norm(surf_normal=True)

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.w_decay_G)
            self.optimizer_D = torch.optim.Adam(itertools.chain(*[m.parameters() for m in self.disc]), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.extend([self.optimizer_G, self.optimizer_D])
            self.opt_names = ['optimizer_G', 'optimizer_D']
            
            self.l_depth_A = opt.l_depth_A_begin
            self.l_depth_B = opt.l_depth_B_begin
            self.l_cycle_A = opt.l_cycle_A_begin
            self.l_cycle_B = opt.l_cycle_B_begin
            self.calc_l_step()
            self.surf_normals = translation_network.SurfaceNormals()

    
    def set_input(self, input):
        self.name_A = input['A_name']
        self.name_B = input['B_name']
        
        self.real_img_A = input['A_img'].to(self.device, non_blocking=torch.cuda.is_available())
        self.real_depth_A = input['A_depth'].to(self.device, non_blocking=torch.cuda.is_available())
        
        self.real_img_B = input['B_img'].to(self.device, non_blocking=torch.cuda.is_available())
        self.real_depth_B = input['B_depth'].to(self.device, non_blocking=torch.cuda.is_available())
    
    def forward(self):             
        ###Masks
        self.hole_mask_A = self.get_mask(self.real_depth_A)
        self.real_depth_A_inp = self.real_depth_A
                    
        inp_A = [self.real_depth_A_inp, self.real_img_A]
        if self.opt.inp_B == 'depth':
            inp_B = [self.real_depth_B]
        else:
            inp_B = [self.real_depth_B, self.real_img_B]

        self.fake_depth_B = self.netG_A(*inp_A)
        self.fake_depth_A = self.netG_B(*inp_B)
        
        ###Normals
        if self.isTrain:
            self.real_norm_A = self.surf_normals(self.real_depth_A)
            self.real_norm_B = self.surf_normals(self.real_depth_B)
            self.fake_norm_A = self.surf_normals(self.fake_depth_A)
            self.fake_norm_B = self.surf_normals(self.fake_depth_B)
        
        ###Masks
        self.hole_mask_B = self.get_mask(self.fake_depth_A)
        self.fake_depth_A_inp = self.fake_depth_A
        
        ###Cycle
        if self.opt.use_cycle_A:
            if self.opt.inp_B == 'depth':
                inp_B_c = [self.fake_depth_B]
            else:
                inp_B_c = [self.fake_depth_B, self.real_img_A]
            self.rec_depth_A = self.netG_B(*inp_B_c)
            if self.isTrain:
                self.rec_norm_A = self.surf_normals(self.rec_depth_A)
        
        if self.opt.use_cycle_B:
            inp_A_c = [self.fake_depth_A_inp, self.real_img_B]
            self.rec_depth_B = self.netG_A(*[i.detach() for i in inp_A_c])
            self.rec_depth_B = self.netG_A(*inp_A_c)
            if self.isTrain:
                self.rec_norm_B = self.surf_normals(self.rec_depth_B)
        
        ### Identical    
        if self.isTrain and self.opt.l_identity > 0:
                inp_A_i = [self.real_depth_B, self.real_img_B]
                if self.opt.inp_B == 'depth':
                    inp_B_i = [self.real_depth_A]
                else:
                    inp_B_i = [self.real_depth_A, self.real_img_A]
                self.idt_A = self.netG_A(*inp_A_i)
                self.idt_B = self.netG_B(*inp_B_i)
                
    def backward_D_base(self, netD, real, fake):
        pred_real = netD(real.detach())
        pred_fake = netD(fake.detach())
        loss_D = 0.5 * (self.criterionGAN(pred_real, True) + self.criterionGAN(pred_fake, False))
        loss_D.backward()
        return loss_D
    
    def backward_D_A(self):           
        if self.opt.disc_for_depth:
            self.loss_D_A_depth = self.backward_D_base(self.netD_A_depth, self.rec_depth_B, self.fake_depth_B) #could be real_depth but we followed Maeda
        if self.opt.disc_for_normals:
            self.loss_D_A_normal = self.backward_D_base(self.netD_A_normal, self.rec_norm_B, self.fake_norm_B) #could be real_norm but we followed Maeda
    
    def backward_D_B(self):
        if self.opt.disc_for_depth:
            self.loss_D_B_depth = self.backward_D_base(self.netD_B_depth, self.real_depth_A, self.fake_depth_A)
        if self.opt.disc_for_normals:
            self.loss_D_B_normal = self.backward_D_base(self.netD_B_normal, self.real_norm_A, self.fake_norm_A)
    
    def backward_G(self):
        loss_A = 0.0
        loss_B = 0.0
        self.loss_G_A = 0.0
        self.loss_G_B = 0.0
        if self.opt.disc_for_depth:
            self.loss_G_A = self.loss_G_A + 0.5*self.criterionGAN(self.netD_A_depth(self.fake_depth_B), True)
            self.loss_G_B = self.loss_G_B + 0.5*self.criterionGAN(self.netD_B_depth(self.fake_depth_A), True)
        if self.opt.disc_for_normals:
            self.loss_G_A = self.loss_G_A + 0.5*self.criterionGAN(self.netD_A_normal(self.fake_norm_B), True)
            self.loss_G_B = self.loss_G_B + 0.5*self.criterionGAN(self.netD_B_normal(self.fake_norm_A), True)
        loss_A = loss_A + self.loss_G_A
        loss_B = loss_B + self.loss_G_B
        
        if self.opt.use_cycle_A:
            self.loss_cycle_A = self.criterionMaskedL1(self.rec_depth_A, self.real_depth_A, ~self.hole_mask_A) * self.l_cycle_A
            self.loss_cycle_n_A = self.criterionMaskedCosSim(self.rec_norm_A, self.real_norm_A, ~self.hole_mask_A.repeat(1,3,1,1)) * self.opt.l_normal * self.l_cycle_A
            loss_A = loss_A + self.loss_cycle_A + self.loss_cycle_n_A
        if self.opt.use_cycle_B:
            self.loss_cycle_B = self.criterionL1(self.rec_depth_B, self.real_depth_B) * self.l_cycle_B
            self.loss_cycle_n_B = self.criterionCosSim(self.rec_norm_B, self.real_norm_B) * self.opt.l_normal * self.l_cycle_B
            loss_B = loss_B + self.loss_cycle_B + self.loss_cycle_n_B
        
        if self.opt.l_identity > 0:
            if not self.opt.no_idt_A:
                self.loss_idt_A = self.criterionL1(self.idt_A, self.real_depth_B) * self.opt.l_identity
                loss_A = loss_A + self.loss_idt_A
            else:
                self.loss_idt_A = 0
            self.loss_idt_B = self.criterionL1(self.idt_B, self.real_depth_A) * self.opt.l_identity
            loss_B = loss_B + self.loss_idt_B    

        if self.opt.l_mean_A > 0:
            self.loss_mean_dif_A = self.criterionMeanDif(self.fake_depth_B, self.real_depth_A, ~self.hole_mask_A) * self.opt.l_mean_A
            loss_A = loss_A + self.loss_mean_dif_A
        if self.opt.l_mean_B > 0:
            self.loss_mean_dif_B = self.criterionMeanDif(self.fake_depth_A, self.real_depth_B, ~self.hole_mask_B) * self.opt.l_mean_B
            loss_B = loss_B + self.loss_mean_dif_B
            
        if self.opt.l_tv_A > 0:
            self.loss_tv_norm_A = self.TVnorm(self.fake_norm_B) * self.opt.l_tv_A
            loss_A = loss_A + self.loss_tv_norm_A
            
        if self.l_depth_A > 0:
            self.loss_depth_range_A = self.criterionMaskedL1(self.fake_depth_B, self.real_depth_A, ~self.hole_mask_A)* self.l_depth_A 
            loss_A = loss_A + self.loss_depth_range_A
        if self.l_depth_B > 0:
            self.loss_depth_range_B = self.criterionMaskedL1(self.fake_depth_A, self.real_depth_B, ~self.hole_mask_B) * self.l_depth_B
            loss_B = loss_B + self.loss_depth_range_B

        self.loss_G = loss_A + loss_B
        self.loss_G.backward()
        
        # so to visualize holes
        self.real_depth_A = self.real_depth_A_inp
        
        # visualization
        with torch.no_grad():
            self.loss_depth_dif_A = self.criterionMaskedL1(util.data_to_meters(self.real_depth_A, self.opt.max_distance), util.data_to_meters(self.fake_depth_B, self.opt.max_distance) ,
                                                           ~self.hole_mask_A).item()
            self.loss_depth_dif_B = self.criterionMaskedL1(util.data_to_meters(self.real_depth_B, self.opt.max_distance), util.data_to_meters(self.fake_depth_A, self.opt.max_distance) ,
                                                           ~self.hole_mask_B).item()
        
    def optimize_parameters(self, iters, fr=1):
        self.set_requires_grad(self.disc, False)
        for _ in range(self.opt.num_iter_gen):
            self.forward()
            self.zero_grad([self.netG_A, self.netG_B])
            self.backward_G()
            self.optimizer_G.step()
        self.set_requires_grad(self.disc, True)
        
        self.set_requires_grad([self.netG_A, self.netG_B], False)
        for j in range(self.opt.num_iter_dis):
            if j > 0:
                self.forward()
            self.zero_grad(self.disc)
            self.backward_D_A()
            self.backward_D_B()
            self.optimizer_D.step()
        self.set_requires_grad([self.netG_A, self.netG_B], True)
    
    def calc_l_step(self):
        with torch.no_grad():
            self.l_depth_A_step = abs(self.opt.l_depth_A_begin - self.opt.l_depth_A_end) / self.opt.l_num_iter
            self.l_depth_B_step = abs(self.opt.l_depth_B_begin - self.opt.l_depth_B_end) / self.opt.l_num_iter
            self.l_cycle_A_step = abs(self.opt.l_cycle_A_begin - self.opt.l_cycle_A_end) / self.opt.l_num_iter
            self.l_cycle_B_step = abs(self.opt.l_cycle_B_begin - self.opt.l_cycle_B_end) / self.opt.l_num_iter
    
    def update_loss_weight(self, global_iter):
        if global_iter > self.opt.l_max_iter:
            self.l_depth_A -= self.l_depth_A_step
            self.l_depth_B -= self.l_depth_B_step
            self.l_cycle_A += self.l_cycle_A_step
            self.l_cycle_B += self.l_cycle_B_step

    def calc_test_loss(self):
        self.test_depth_dif_A = self.criterionMaskedL1(util.data_to_meters(self.real_depth_A, self.opt.max_distance), util.data_to_meters(self.fake_depth_B, self.opt.max_distance), ~self.hole_mask_A)
        self.test_depth_dif_B = self.criterionMaskedL1(util.data_to_meters(self.real_depth_B, self.opt.max_distance), util.data_to_meters(self.fake_depth_A, self.opt.max_distance), ~self.hole_mask_B)

    def get_L1_loss(self):
        return self.criterionMaskedL1(util.data_to_meters(self.real_depth_A, self.opt.max_distance), util.data_to_meters(self.fake_depth_B, self.opt.max_distance), ~self.hole_mask_A).item()
    def get_L1_loss_syn(self):
        return self.criterionMaskedL1(util.data_to_meters(self.real_depth_B, self.opt.max_distance), util.data_to_meters(self.fake_depth_A, self.opt.max_distance), ~self.hole_mask_B).item()
    def get_L1_loss_cycle(self):
        return self.criterionMaskedL1(util.data_to_meters(self.real_depth_A, self.opt.max_distance), util.data_to_meters(self.rec_depth_A, self.opt.max_distance), ~self.hole_mask_A).item()
    def get_L1_loss_cycle_syn(self):
        return nn.L1Loss()(util.data_to_meters(self.real_depth_B, self.opt.max_distance), util.data_to_meters(self.rec_depth_B, self.opt.max_distance)).item()

    def get_dif(self): #x,y; y-x
        return  translation_network.MaskedLoss()(util.data_to_meters(self.real_depth_A, self.opt.max_distance), util.data_to_meters(self.fake_depth_B, self.opt.max_distance) , ~self.hole_mask_A).item()
    def get_dif_syn(self):
        return  translation_network.MaskedLoss()(util.data_to_meters(self.real_depth_B, self.opt.max_distance), util.data_to_meters(self.fake_depth_A, self.opt.max_distance) , ~self.hole_mask_B).item()
    def get_mask(self, input):
        hole_mask = input <= -0.98
        return hole_mask
    