import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageFile
import random
ImageFile.LOAD_TRUNCATED_IMAGES=True
import numpy as np
import torch
import random
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import glob
import albumentations as A
import numpy as np

class SRDataset(data.Dataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """


    def apply_transformer(self, transformations,  depth):

#         target = {
#             'depth':'depth',}
        res = A.Compose(transformations, p=1)(image=depth)
        return res
    
    def trasform(self, depth, img, semantic=None):
        

        img = img.astype(np.float32)
        img = (img - 127.5) / 127.5   #HYPERPRAM
#         print(depth.dtype, np.min(depth), np.mean(depth), np.max(depth))
#         meters = 10 if self.opt.notscannet else 8
        if depth.dtype == np.int32:
            m_in_mm = 5100
            depth = np.where(depth>m_in_mm, m_in_mm, depth)/m_in_mm
            depth = depth*2 - 1

        else:     
            depth = np.where(depth<0.33, 0, depth)
            depth = np.where(depth>meters, meters, depth)/meters
            depth = depth*2 - 1
            
#         depth = (depth -np.max(depth)/2) / (np.max(depth)/2)    
#         print(depth.dtype, np.min(depth), np.mean(depth), np.max(depth))
#         print('=========================================================================')
        depth = depth.astype(np.float32)
        
        
        transform_list  = []
        transform_list_d = []
#         print(depth.shape, img.shape)

        height =  depth.shape[0]
        width = depth.shape[1]
        
#         height_c = 384//2
#         width_c = 512//2
        
        
        height_c = height//4
        width_c = width//4
        
        transform_list.append(A.Resize(height=height_c, width=width_c, interpolation=2, p=1))
        
        
        transform_list_d.append(A.Resize(height=height, width=width, interpolation=2, p=1))
        transformed_d = self.apply_transformer(transform_list_d, img)
        img = transformed_d['image']
#             transform_list.append(A.Rotate(limit = [-30,30], p=0.8))
#             transform_list.append(A.RandomCrop(height=height_c, width=width_c, p=1))
#         transform_list.append(A.HorizontalFlip(p=0.5))
        

#         transform_list.append(A.Resize(height=480, width=640, interpolation=4, p=1))


        transformed = self.apply_transformer(transform_list, depth)
        
#         img = np.clip(transformed['image'], -1, 1)
        depth_lr = np.clip(transformed['image'], -1, 1)
    
        transform_list_bic = [A.Resize(height=height, width=width, interpolation=2, p=1)]
        transformed = self.apply_transformer(transform_list_bic, depth_lr)
        depth_hr_bib = np.clip(transformed['image'], -1, 1)
        
        
        img = torch.from_numpy(img).permute(2, 0, 1)
        depth_lr = torch.from_numpy(depth_lr).unsqueeze(0)
        depth_hr_bib = torch.from_numpy(depth_hr_bib).unsqueeze(0)
        depth = torch.from_numpy(depth).unsqueeze(0)
#         print( depth.shape, img.shape)
        return img, depth, depth_lr, depth_hr_bib
    
    
    def __init__(self, opt, stage='train'):
        
        self.opt = opt    
    
        
        self.dir_A = os.path.join(opt.path_A)  # create a path '/path/to/data/trainA'      
        self.dir_A_add = os.path.join(opt.A_add_paths)  # create a path '/path/to/data/trainA'

        
        if stage=='test':
            self.dir_A = os.path.join(opt.path_A_test)  # create a path '/path/to/data/trainA
            self.dir_A_add = os.path.join(opt.A_add_paths_test)  # create a path '/path/to/data/trainA'

        
        if opt.take>0:
            np.random.seed(23)
            A = sorted(glob.glob(self.dir_A +'/*'))   
            indexes_A = np.random.randint(0, len(A)-1, opt.take)
            self.A_paths = np.array(A)[indexes_A]
            if opt.image_and_depth:
                A_add_paths = sorted(glob.glob(self.dir_A_add +'/*'))   
                self.A_add_paths = np.array(A_add_paths)[indexes_A]
            
        else:
            self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
            self.A_add_paths = sorted(make_dataset(self.dir_A_add, opt.max_dataset_size))   

        
        print(len(self.A_paths), len(self.A_add_paths))        
        self.A_size = len(self.A_paths)  # get the size of dataset A



    def __getitem__(self, index):

        index_A = index % self.A_size
        A_path = self.A_paths[index_A]  # make sure index is within then range
        A_depth = np.array(Image.open(A_path))

#         jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        A_img = np.array(Image.open(self.A_add_paths[index_A])).astype(np.float32)

            
        img, depth, depth_lr, depth_hr_bib = self.trasform(A_depth, A_img)


        return {'image': img, 'hr_depth': depth, 'lr_depth': depth_lr, 'hr_bibcubic_depth': depth_hr_bib, 'A_paths': A_path}

    def __len__(self):

        return self.A_size 

    