import glob
import torch
import random

import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader

class PairedImgDataset(Dataset):
    def __init__(self, data_source, mode, crop=256, random_resize=None):
        if not mode in ['train', 'val', 'test']:
            raise Exception('The mode should be "train", "val" or "test".')
        
        self.random_resize = random_resize
        self.crop = crop
        self.mode = mode
        
        self.img_paths = sorted(glob.glob(data_source + '/' + mode + '/input' + '/*.*'))
        self.gt_paths = sorted(glob.glob(data_source + '/'  + mode + '/GT' + '/*.*'))

    def __getitem__(self, index):
        img = np.load(self.img_paths[index % len(self.img_paths)])
        gt = np.load(self.gt_paths[index % len(self.gt_paths)])
        
        img, gt = self.tone_map(img, gt)
        
        img, gt = self.to_tensor(img, gt)
        
        if self.mode == 'train':
            if self.random_resize is not None:
                # random resize
                scale_factor = random.uniform(self.crop/self.random_resize, 1.)
                img = F.interpolate(img.unsqueeze(0), scale_factor=scale_factor, align_corners=False, mode='bilinear', recompute_scale_factor=False).squeeze(0)
                gt = F.interpolate(gt.unsqueeze(0), scale_factor=scale_factor, align_corners=False, mode='bilinear', recompute_scale_factor=False).squeeze(0)
            
            # crop
            h, w = img.size(1), img.size(2)
            offset_h = random.randint(0, max(0, h - self.crop - 1))
            offset_w = random.randint(0, max(0, w - self.crop - 1))

            img = img[:, offset_h:offset_h + self.crop, offset_w:offset_w + self.crop]
            gt = gt[:, offset_h:offset_h + self.crop, offset_w:offset_w + self.crop]
        
            # flip
            # vertical flip
            if random.random() < 0.5:
                idx = [i for i in range(img.size(1) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                img = img.index_select(1, idx)
                gt = gt.index_select(1, idx)
            # horizontal flip
            if random.random() < 0.5:
                idx = [i for i in range(img.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                img = img.index_select(2, idx)
                gt = gt.index_select(2, idx)
        
        return img, gt

    def __len__(self):
        return max(len(self.img_paths), len(self.gt_paths))
    
    def tone_map(self, x, y):
        return x / (x + 0.25), y / (y + 0.25)
    
    def to_tensor(self, x, y):
        """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
        x = np.transpose(x, (2, 0, 1))
        y = np.transpose(y, (2, 0, 1))
        x  = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        return x, y
        
class SingleImgDataset(Dataset):
    def __init__(self, data_source):
        
        self.img_paths = sorted(glob.glob(data_source + '/' + 'test' + '/input' + '/*.*'))

    def __getitem__(self, index):
        
        path = self.img_paths[index % len(self.img_paths)]
        
        img = np.load(path)
        
        img = self.tone_map(img)
        
        img = self.to_tensor(img)
        
        return img, path

    def __len__(self):
        return len(self.img_paths)
    
    def tone_map(self, x):
        return x / (x + 0.25)
    
    def to_tensor(self, x):
        """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
        x = np.transpose(x, (2, 0, 1))
        x  = torch.from_numpy(x).float()
        return x