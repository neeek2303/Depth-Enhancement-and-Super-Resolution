import os
import numpy as np
import albumentations as A
import imageio
import torch
import queue
import torch.utils.data as data
from abc import ABC, abstractmethod
import glob

class BaseDataset(data.Dataset, ABC):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.scale = self.opt.max_distance / 2
        self.IMG_EXTENSIONS = []
        self.transforms = [A.Resize(height=self.opt.load_size_h, width=self.opt.load_size_w, interpolation=4, p=4)]
        self.dir_A = os.path.join(self.root, self.opt.phase + 'A')
        self.dir_B = os.path.join(self.root, self.opt.phase + 'B')
    
    def add_extensions(self, ext_list):
        self.IMG_EXTENSIONS.extend(ext_list)    
    
    def is_image_files(self, files):
        for f in files:
            assert any(f.endswith(extension) for extension in self.IMG_EXTENSIONS), 'not implemented file extention type {}'.format(f.split('.')[1])
        
    def get_paths(self, dir, reverse=False):
        files = []
        assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)
        files = sorted(glob.glob(os.path.join(dir, '**/*.*'), recursive=True), reverse=reverse)
        return files[:min(self.opt.max_dataset_size, len(files))]
    
    def get_name(self, file_path):
        img_n = os.path.basename(file_path).split('.')[0]
        return img_n
    
    def normalize_img(self, img):
        if isinstance(img, np.ndarray):
            if img.dtype == np.uint8:
                if img.shape[2] > 3:
                    img = img[:,:,:3]
                img = img.astype(np.float32)
                img = img / 127.5 - 1.0
                return img

            else:
                print(img.dtype)
                raise AssertionError('Img datatype')
        else:
            raise AssertionError('Img filetype')
    
    def normalize_depth(self, depth):
        if isinstance(depth, np.ndarray):
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32)
                depth = np.where(depth>self.opt.max_distance, self.opt.max_distance, depth)
                depth = depth / self.scale - 1
                return depth
            else:
                print(depth.dtype)
                raise AssertionError('Depth datatype')
        else:
            raise AssertionError('Depth filetype')
    
    def read_data(self, path):
        return imageio.imread(path)
    
    @abstractmethod
    def __len__(self):
        return 0
    @abstractmethod
    def __getitem__(self, index):
        pass

class MyUnalignedDataset(BaseDataset):
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--max_distance', type=float, default=5100.0, help='all depth bigger will seted to this value')
        if is_train:
            pass
        return parser

    def __init__(self, opt, stage='train'):
        super().__init__(opt)
        self.add_extensions(['.png', '.jpg'])
        self.transforms_A = []
        self.transforms_B = []
        self.add_base_transform()
        self.dir_A_img = os.path.join(self.dir_A, 'img') 
        self.dir_A_depth = os.path.join(self.dir_A, 'depth') 
        self.dir_B_img = os.path.join(self.dir_B, 'img') 
        self.dir_B_depth = os.path.join(self.dir_B, 'depth') 
#         self.intrinsic_mtrx_path = opt.int_mtrx_scan

        self.A_imgs = self.get_paths(self.dir_A_img)
        self.A_depths = self.get_paths(self.dir_A_depth)
        
        assert (len(self.A_imgs) == len(self.A_depths)), 'not pair img depth' 
        self.is_image_files(self.A_imgs + self.A_depths)
        self.B_imgs = self.get_paths(self.dir_B_img)
        self.B_depths = self.get_paths(self.dir_B_depth)
        assert (len(self.B_imgs) == len(self.B_depths)), 'not pair img depth'
        self.is_image_files(self.B_imgs + self.B_depths)
        
        self.A_size = len(self.A_imgs)
        self.B_size = len(self.B_imgs)
        self.queue_A_index = queue.Queue()
        
    def update_A_idx(self):
        index = torch.randperm(self.A_size)
        
        for i in range(len(index)):
            self.queue_A_index.put(index[i].item())
        
    def __getitem__(self, index):
        A_depth, A_img, A_semantic, B_depth, B_img, A_img_n, B_img_n = self.load_data(index)
        return {'A_depth': A_depth, 'A_img': A_img, 'A_name': A_img_n, 'B_depth': B_depth, 'B_img': B_img, 'B_name':B_img_n}
    
    def load_data(self, index):
        
        if  self.A_size != self.B_size:
            if self.queue_A_index.empty():
                self.update_A_idx()
            index_A = self.queue_A_index.get()
        else:
            index_A = index
        index_B = index
        
        A_img_path = self.A_imgs[index_A]
        A_depth_path = self.A_depths[index_A]
        
        B_img_path = self.B_imgs[index_B]
        B_depth_path = self.B_depths[index_B]

        A_img_n = self.get_name(A_img_path)
        A_depth_n = self.get_name(A_depth_path)
        assert (A_img_n == A_depth_n), 'not pair img depth '
        B_img_n = self.get_name(B_img_path)
        B_depth_n = self.get_name(B_depth_path)
        assert (B_img_n == B_depth_n), 'not pair img depth'
        
        A_depth = self.read_data(A_depth_path)
        B_depth = self.read_data(B_depth_path)
        A_semantic = None
            
        A_img = self.read_data(A_img_path)
        B_img = self.read_data(B_img_path)
        
        A_depth, A_img, A_semantic = self.transform('A', A_depth, A_img, A_semantic)
        B_depth, B_img, _ = self.transform('B', B_depth, B_img)
        if self.opt.isTrain:
            if self.bad_img(A_depth, A_img, B_depth, B_img):
                print('Try new img')
                A_depth, A_img, A_semantic, B_depth, B_img, A_img_n, B_img_n = self.load_data(torch.randint(low=0, high=self.B_size, size=(1,)).item())
        return A_depth, A_img, A_semantic, B_depth, B_img, A_img_n, B_img_n
        
    
    def bad_img(self, *imgs):
        for i in imgs:
            if not torch.isfinite(i).all():
                print('NaN in img')
                return True
            elif torch.unique(i).shape[0] < 2:
                print('All values are same')
                return True
        return False
        
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
    
    def add_base_transform(self):
        self.transforms_A.append(A.Resize(height=320, width=320, interpolation=4, p=1))
        self.transforms_B.append(A.Resize(height=320, width=320, interpolation=4, p=1))
#         self.transforms_B.append(A.Resize(height=self.opt.load_size_h_B, width=self.opt.load_size_w_B, interpolation=4, p=1))
        if self.opt.isTrain:
            self.transforms_A.append(A.RandomCrop(height=self.opt.crop_size_h, width=self.opt.crop_size_w, p=1))
            self.transforms_B.append(A.RandomCrop(height=self.opt.crop_size_h, width=self.opt.crop_size_w, p=1))
            self.transforms_A.append(A.HorizontalFlip(p=0.5))
            self.transforms_B.append(A.HorizontalFlip(p=0.5))

    def transform(self, domain, depth, img, semantic=None):
        img = self.normalize_img(img)
        depth = self.normalize_depth(depth)
        if domain == 'A':
            transformed = self.apply_transformer(self.transforms_A, img, depth, semantic)
        elif domain == 'B':
            transformed = self.apply_transformer(self.transforms_B, img, depth, semantic)
        img = transformed['image']
        depth = transformed['depth']
        img = torch.from_numpy(img).permute(2, 0, 1)
        depth = torch.from_numpy(depth).unsqueeze(0)
        if semantic is not None:
            semantic = transformed['mask']
            semantic = torch.from_numpy(semantic).long()
        return depth, img, semantic
    
    def __len__(self):
        return self.B_size