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
from torch import nn
import kornia
from .pytorch_ssim import SSIM
from . import networks_for_SR

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

    
    
    
def edge_loss(out, target, cuda=True):
    x_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    y_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    convx = torch.nn.functional.conv2d
    convy = torch.nn.functional.conv2d
    
    weights_x = torch.from_numpy(x_filter).float().unsqueeze(0).unsqueeze(0)
    weights_y = torch.from_numpy(y_filter).float().unsqueeze(0).unsqueeze(0)


    weights_x = weights_x.to(out.device)
    weights_y = weights_y.to(out.device)


    g1_x = convx(out, weights_x, stride=1, padding=1, bias=None)
    g2_x = convx(target, weights_x, stride=1, padding=1, bias=None)
    g1_y = convy(out, weights_y, stride=1, padding=1, bias=None)
    g2_y = convy(target, weights_y, stride=1, padding=1, bias=None)

    g_1 = torch.sqrt(torch.pow(g1_x, 2) + torch.pow(g1_y, 2))
    g_2 = torch.sqrt(torch.pow(g2_x, 2) + torch.pow(g2_y, 2))

    return torch.mean((g_1 - g_2).pow(2)), g_1, g_2    


    
  
    
class MLPFLModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['L1', 'edge', 'L1_bibc', 'edge_bibc', 'ssim_bib', 'ssim']
            
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['image', 'hr_bibcubic_depth', 'lr_depth', 'hr_gt_depth', 'prediction', 'edges_hr', 'edges_pred', 'edges_bibc']
        self.model_names = ['EncoderImage', 'EncoderDepth',  'Decoder']
        
        
        # Define networks 
        self.netEncoderImage = networks_for_SR.define_net('EncoderImage', gpu_ids = self.gpu_ids)

        self.netEncoderDepth = networks_for_SR.define_net('EncoderDepth', gpu_ids = self.gpu_ids)
#         print( next( self.netEncoderDepth.module.parameters()).device)

        self.netDecoder = networks_for_SR.define_net('Decoder', gpu_ids = self.gpu_ids)

    
        
        if self.isTrain:
            self.l1_loss = torch.nn.L1Loss()
            self.ssim_loss = SSIM(window_size = 11)
            self.edge_loss = edge_loss

            
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(itertools.chain(self.netEncoderImage.parameters(), self.netEncoderDepth.parameters(), self.netDecoder.parameters()), lr=opt.lr)




    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        
        self.image = input['image'].to(self.device)
        self.lr_depth = input['lr_depth'].to(self.device)
        self.hr_bibcubic_depth = input['hr_bibcubic_depth'].to(self.device)
        self.hr_gt_depth = input['hr_depth'].to(self.device)
        
#         print(self.image.shape, self.lr_depth.shape, self.hr_bibcubic_depth.shape, self.hr_gt_depth.shape)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        
        
        self.image_features = self.netEncoderImage(self.image)
        
#         for i in self.image_features:
#             print('im feature: ', i.shape)
        
        self.depth_features = self.netEncoderDepth(self.hr_bibcubic_depth)
        
        
#         for i in self.depth_features:
#             print('im feature: ', i.shape)
        
        self.prediction = self.netDecoder(self.image_features, self.depth_features)
        

    def backward(self, back=True):
        """Calculate the loss for generators G_A and G_B"""

        if self.opt.norm_loss:
            calc_norm = SurfaceNormals()

        
#         print(self.prediction)
#         print(torch.mean(self.prediction))
        self.loss_L1 = self.l1_loss(self.hr_gt_depth, self.prediction)
        self.edges_hr = kornia.filters.sobel(self.hr_gt_depth)
        self.edges_pred = kornia.filters.sobel(self.prediction)
        self.edges_bibc = kornia.filters.sobel(self.hr_bibcubic_depth)
        
#         print(self.edges_hr)
        
        self.loss_L1_bibc = self.l1_loss(self.hr_gt_depth, self.hr_bibcubic_depth)
        
        self.loss_edge = self.l1_loss(self.edges_hr, self.edges_pred)
        
        self.loss_edge_bibc = self.l1_loss(self.edges_bibc, self.edges_pred)
        
#         self.loss_edge_l1 = self.edge_loss(self.hr_gt_depth, self.prediction)
        self.loss_ssim = 1-self.ssim_loss((self.hr_gt_depth+1)/2, (self.prediction+1)/2) 
        self.loss_ssim_bib = 1-self.ssim_loss((self.hr_gt_depth+1)/2, (self.edges_bibc+1)/2) 

        # combined loss and calculate gradients
        self.loss_G = self.loss_L1*self.opt.w_loss_l1 + self.loss_edge*self.opt.w_edge_l1 + self.loss_ssim*self.opt.w_ssim
        
        if back:
            self.loss_G.backward()

    def optimize_parameters(self, iters, fr=1):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()    
        self.optimizer.zero_grad() 
        self.backward()            
        self.optimizer.step()     
        
            
            
    def calculate(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        self.backward(back=False)             # calculate gradients for G_A and G_B

