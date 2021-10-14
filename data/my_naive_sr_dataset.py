import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageFile
import random
ImageFile.LOAD_TRUNCATED_IMAGES=True
import numpy as np
import torch
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import glob
import albumentations as A
import numpy as np

class MyUnalignedDataset(BaseDataset):

    def apply_transformer(self, transformations, img, depth, semantic=None):
        if semantic is not None:
            target = {
                'image':'image',
                'depth':'image',
                'mask': 'mask',}
            res = A.Compose(transformations, p=1, additional_targets=target)(image=img, depth=depth, mask=semantic)
        else:
            target = {
                'image':'image',
                'depth':'image',}
            res = A.Compose(transformations, p=1, additional_targets=target)(image=img, depth=depth)
        return res
    
    def trasform(self, depth, img, full=True, train=True):
        

        img = img.astype(np.float32)
        img = (img - 127.5) / 127.5  
        meters = 5100 
        if depth.dtype == np.int32:
            m_in_mm = meters
            depth = np.where(depth>m_in_mm, m_in_mm, depth)/m_in_mm
            depth = depth*2 - 1

        else:     
            depth = np.where(depth<0.33, 0, depth)
            depth = np.where(depth>meters, meters, depth)/meters
            depth = depth*2 - 1
            

        depth = depth.astype(np.float32)

        transform_list  = []
        



        height =  960
        width = 1280
        
        
        height_c = self.opt.crop_size_h*2
        width_c = self.opt.crop_size_w*2
        

        
        transform_list = []

        if train:
            if full:
                transform_list.append(A.Resize(height=height, width=width, interpolation=3, p=1))
                transform_list.append(A.PadIfNeeded(1024, 1280, p=1))

                transform_list_d = []
                transform_list_d.append(A.Resize(height=height_c//2, width=width_c//2, interpolation=3, p=1))
                
                transformed_or = self.apply_transformer(transform_list, img, depth)
                h, w = np.random.randint(0,1024-height_c+1), np.random.randint(0,1280-width_c+1)
                depth_origin = transformed_or['depth'][h:h+height_c, w:w+width_c]
                img_origin = transformed_or['image'][h:h+height_c, w:w+width_c]
        else:
            transform_list.append(A.Resize(height=height, width=width, interpolation=3, p=1))
            transform_list.append(A.PadIfNeeded(1024, 1280, p=1))
            
            transform_list_d = []
            transform_list_d.append(A.Resize(height=height_c, width=width_c, interpolation=3, p=1))
            h, w = 0, 0

            transformed_or = self.apply_transformer(transform_list, img, depth)
            depth_origin = transformed_or['depth']
            img_origin = transformed_or['image']
        
        
        
        

        transformed = self.apply_transformer(transform_list_d, img_origin, depth_origin)
        depth = np.clip(transformed['depth'], -1, 1)
        img = np.clip(transformed['image'], -1, 1)
        img_origin = np.clip(img_origin, -1, 1)
        depth_origin = np.clip(depth_origin, -1, 1)
        
        depth = torch.from_numpy(depth).unsqueeze(0)
        img = torch.from_numpy(img).permute(2, 0, 1)
        img_origin = torch.from_numpy(img_origin).permute(2, 0, 1)
        depth_origin = torch.from_numpy(depth_origin).unsqueeze(0)
        
#         print(depth_origin.shape, depth.shape, img.shape)
        return depth_origin, img_origin, depth, img, h, w
        
    
    
    def __init__(self, opt, stage='train'):
        
        BaseDataset.__init__(self, opt)
        
        self.opt = opt    
        self.dir_A = os.path.join(opt.path_A)  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.path_B)  # create a path '/path/to/data/trainB'

        if opt.image_and_depth:
            self.dir_A_add = os.path.join(opt.A_add_paths)  # create a path '/path/to/data/trainA'
            self.dir_B_add = os.path.join(opt.B_add_paths)  # create a path '/path/to/data/trainB'
        self.train = True
        if stage=='test':
            self.train = False
            self.dir_A = os.path.join(opt.path_A_test)  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(opt.path_B_test)  # create a path '/path/to/data/trainB'

            if opt.image_and_depth:
                self.dir_A_add = os.path.join(opt.A_add_paths_test)  # create a path '/path/to/data/trainA'
                self.dir_B_add = os.path.join(opt.B_add_paths_test)  # create a path '/path/to/data/trainB'
        
        if opt.take>0:
            np.random.seed(23)
            A = sorted(glob.glob(self.dir_A +'/*'))   
            B = sorted(glob.glob(self.dir_B +'/*'))  
            indexes_A = np.random.randint(0, len(A)-1, opt.take)
            indexes_B = np.random.randint(0, len(B)-1, opt.take)
            
            self.A_paths = np.array(A)[indexes_A]
            self.B_paths = np.array(B)[indexes_B]
            
            if opt.image_and_depth:
                A_add_paths = sorted(glob.glob(self.dir_A_add +'/*'))   
                B_add_paths = sorted(glob.glob(self.dir_B_add +'/*'))  
                
                self.A_add_paths = np.array(A_add_paths)[indexes_A]
                self.B_add_paths = np.array(B_add_paths)[indexes_B]
            
        else:
            self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
            self.B_paths = sorted(glob.glob(f'{self.dir_B}/*'))  # load images from '/path/to/data/trainB'

            if opt.image_and_depth:
                self.A_add_paths = sorted(make_dataset(self.dir_A_add, opt.max_dataset_size))   
                self.B_add_paths = sorted(glob.glob(f'{self.dir_B_add}/*'))  # load images from '/path/to/data/trainB'
            
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image

    def get_imp_matrx(self, f_name):
        f_name = f_name.split('/')[-1].split('.')[0]
        K = np.loadtxt(os.path.join(self.opt.path_to_intr, f_name[:12], 'intrinsic', 'intrinsic_depth.txt'))[:3,:3]
        return K
    
    def __getitem__(self, index):


        index_B = index % self.B_size
        B_path = self.B_paths[index_B]
        index_A = random.randint(0, self.A_size - 1)
        A_path = self.A_paths[index_A]
        

        if self.opt.image_and_depth:
            
            A_depth = np.array(Image.open(A_path))
            B_depth = np.array(Image.open(B_path)) if self.opt.use_scannet else np.array(np.load(B_path)).astype(np.float32) 


            A_img = np.array(Image.open(self.A_add_paths[index_A])).astype(np.float32)
            B_img = np.array(Image.open(self.B_add_paths[index_B])).astype(np.float32)
            
            A_depth, A_img, _, _, h_a, w_a  = self.trasform(A_depth, A_img, train =self.train, full=True)
            B_depth, B_img, _, _, h_b, w_b = self.trasform(B_depth, B_img, train =self.train, full=True)
#             print(A_path)

            K_B = self.get_imp_matrx(B_path)
            
            if self.opt.interiornet:
                K_A = np.asarray([[600, 0, 320],
                        [0, 600, 240],
                        [0, 0, 1]])
            else:
                K_A = self.get_imp_matrx(A_path)
                
            scale_K = np.array([[2., 1., 2.],[1., 2., 2.],[1., 1., 1.]])
            K_A = K_A*scale_K
            if self.train:
                crop_A = np.array([h_a, self.opt.crop_size_h*2+h_a, w_a, self.opt.crop_size_w*2+w_a])
                crop_B = np.array([h_b, self.opt.crop_size_h+h_b, w_b, self.opt.crop_size_w+w_b])
                
            else:
                crop_A = np.array([0, 512*2, 0, 640*2])
                crop_B = np.array([0, 512, 0, 640])                
         
        return {'A_i': A_img, 'B_i': B_img, 'A_d': A_depth, 'B_d': B_depth, 'A_paths': A_path, 'B_paths': B_path, 'K_A': K_A, 'K_B': K_B, 'crop_A': crop_A, 'crop_B': crop_B}



    def __len__(self):

        print(self.A_size, self.B_size)
        return min(self.A_size, self.B_size)  #if self.opt.direction=='AtoB' else self.B_size

    