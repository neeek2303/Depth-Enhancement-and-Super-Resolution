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
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """


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
    
    def trasform(self, depth, img, semantic=None):
        

        img = img.astype(np.float32)
        img = (img - 127.5) / 127.5   #HYPERPRAM
#         print(depth.dtype, np.min(depth), np.mean(depth), np.max(depth))
        
        if depth.dtype == np.int32:
            depth = np.where(depth>10000, 10000, depth)/10000
            depth = depth*2 - 1

        else:     
            depth = np.where(depth<0.33, 0, depth)
            depth = np.where(depth>10, 10, depth)/10
            depth = depth*2 - 1
            
#         depth = (depth -np.max(depth)/2) / (np.max(depth)/2)    
#         print(depth.dtype, np.min(depth), np.mean(depth), np.max(depth))
#         print('=========================================================================')
        depth = depth.astype(np.float32)
            
        transform_list  = []
        transform_list.append(A.Resize(height=self.opt.load_size, width=self.opt.load_size, interpolation=4, p=1))
        if self.opt.isTrain:
            transform_list.append(A.Rotate(limit = [-30,30], p=0.8))
            transform_list.append(A.RandomCrop(height=self.opt.crop_size, width=self.opt.crop_size, p=1))
        transform_list.append(A.HorizontalFlip(p=0.5))

        transformed = self.apply_transformer(transform_list, img, depth)          
        img = np.clip(transformed['image'], -1, 1)
        depth = np.clip(transformed['depth'], -1, 1)
        img = torch.from_numpy(img).permute(2, 0, 1)
        depth = torch.from_numpy(depth).unsqueeze(0)

        return depth, img
    
    
    def __init__(self, opt, stage='train'):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        
        self.opt = opt    
    
        
        
        self.dir_A = os.path.join(opt.path_A)  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.path_B)  # create a path '/path/to/data/trainB'

        if opt.image_and_depth:
            self.dir_A_add = os.path.join(opt.A_add_paths)  # create a path '/path/to/data/trainA'
            self.dir_B_add = os.path.join(opt.B_add_paths)  # create a path '/path/to/data/trainB'
        
        if stage=='test':
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
        
        print(len(self.A_paths),len(self.B_paths),len(self.A_add_paths),len(self.B_add_paths))        
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=False)
        self.transform_B = get_transform(self.opt, grayscale=False)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
#         B_path = self.B_paths[index % self.B_size]        
        if self.opt.image_and_depth:
            
            A_depth = np.array(Image.open(A_path))
            B_depth = np.array(np.load(B_path)).astype(np.float32)
#             B_depth = np.array(Image.open(B_path))
            
            A_img = np.array(Image.open(self.A_add_paths[index % self.A_size]).convert('RGB')).astype(np.float32)
            B_img = np.array(Image.open(self.B_add_paths[index_B]).convert('RGB')).astype(np.float32)
 
            A_depth, A_img, = self.trasform(A_depth, A_img)
            B_depth, B_img, = self.trasform(B_depth, B_img)
            
        return {'A_i': A_img, 'B_i': B_img, 'A_d': A_depth, 'B_d': B_depth, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return min( self.A_size, self.B_size)  #if self.opt.direction=='AtoB' else self.B_size

    