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

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        
        self.opt = opt    
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        
        if opt.custom_pathes:
            self.dir_A = os.path.join(opt.path_A)  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(opt.path_B)  # create a path '/path/to/data/trainB'
            
        if opt.image_and_depth:
            self.dir_A_add = os.path.join(opt.A_add_paths)  # create a path '/path/to/data/trainA'
            self.dir_B_add = os.path.join(opt.B_add_paths)  # create a path '/path/to/data/trainB'
            
        
        if opt.take>0:
            np.random.seed(23)
            A = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   
            B = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  

            indexes_A = np.random.randint(0, len(A)-1, opt.take)
            indexes_B = np.random.randint(0, len(B)-1, opt.take)
            
            self.A_paths = np.array(A)[indexes_A]
            self.B_paths = np.array(B)[indexes_B]
            
        else:
            self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
            self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
            
            if opt.image_and_depth:
                self.A_add_paths = sorted(make_dataset(self.dir_A_add, opt.max_dataset_size))   
                self.B_add_paths = sorted(make_dataset(self.dir_B_add, opt.max_dataset_size))  
                
                
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
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        
        
        
        
        if not self.opt.uint16:
        
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')

            A = self.transform_A(A_img)
            B = self.transform_B(B_img)
        
        else:
            A_img = Image.open(A_path)
            B_img = Image.open(B_path)
            A = self.transform_A(A_img)
            B = self.transform_B(B_img)
            A = np.where(np.array(A, np.float32)>8000, 8000, np.array(A, np.float32))/8000
            B = np.where(np.array(B, np.float32)>8000, 8000, np.array(B, np.float32))/8000
            convert = transforms.Lambda(lambda image: torch.from_numpy(image).unsqueeze(0))
            norm = transforms.Normalize((0.5,), (0.5,))
            
            A = norm(convert(A))
            B = norm(convert(B))
        
        if self.opt.image_and_depth:
            
            A_depth = Image.open(A_path)
            B_depth = Image.open(B_path)
            
            A_img = Image.open(self.A_add_paths[index % self.A_size]).convert('RGB')
            B_img = Image.open(self.B_add_paths[index_B]).convert('RGB')
            
            
            A_i, A_d = paired_transform(self.opt, A_img, A_depth)
            B_i, B_d = paired_transform(self.opt, B_img, B_depth)
            resize = transforms.Resize([256, 256], Image.BICUBIC)
            
            
            
            A_d = resize(A_d)
            A_i = resize(A_i)
            B_d = resize(B_d)
            B_i = resize(B_i)
            
            
                
            A_d = np.where(np.array(A_d, np.float32)>8000, 8000, np.array(A_d, np.float32))/8000
            B_d = np.where(np.array(B_d, np.float32)>8000, 8000, np.array(B_d, np.float32))/8000
            
            
            convert = transforms.Lambda(lambda image: torch.from_numpy(image).unsqueeze(0))
            norm = transforms.Normalize((0.5,), (0.5,))
            
            convert_i = transforms.ToTensor()
            norm_i = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            
            A_d = norm(convert(A_d))
            B_d = norm(convert(B_d))
            
            A_i = norm_i(convert_i(A_i))
            B_i = norm_i(convert_i(B_i)) 
            
            A = torch.cat([A_i, A_d], dim=0)
            B = torch.cat([B_i, B_d], dim=0)
            

            
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    
def paired_transform(opt, image, depth):
    scale_rate = 1.0

    if True:
        n_flip = random.random()
        if n_flip > 0.5:
            image = F.hflip(image)
            depth = F.hflip(depth)

    if False:
        n_rotation = random.random()
        if n_rotation > 0.5:
            degree = random.randrange(-500, 500)/100
            image = F.rotate(image, degree, Image.BICUBIC)
            depth = F.rotate(depth, degree, Image.BILINEAR)

    return image, depth