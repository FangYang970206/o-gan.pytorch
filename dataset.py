import os
import torch
import torch.utils.data as data
import numpy as np
import scipy.misc as misc


class CelebA_Dataset(data.Dataset):
    def __init__(self, data_files, resize_shape):
        self.resize_shape = resize_shape
        self.data_files = data_files
        
    def __getitem__(self, ind):
        path = self.data_files[ind]
        img = misc.imread(path, mode='RGB')
        img = misc.imresize(img, self.resize_shape)
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1) / 255 * 2 - 1
        return torch.from_numpy(img)

    def __len__(self):
        return len(self.data_files)