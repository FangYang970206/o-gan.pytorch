import os
import torch
import torch.utils.data as data
import numpy as np
import scipy.misc as misc


class CelebA_Dataset(data.Dataset):
    def __init__(self, data_files, img_dim=128):
        self.data_files = data_files
        self.img_dim = img_dim
        
    def __getitem__(self, ind):
        path = self.data_files[ind]
        img = misc.imread(path, mode='RGB')
        img = misc.imresize(img, (self.img_dim, self.img_dim))
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1) / 255 * 2 - 1
        return torch.from_numpy(img)

    def __len__(self):
        return len(self.data_files)


if __name__ == '__main__':
    from glob import glob
    from os.path import join
    from torch.utils import data
    data_files = glob(join('F:\\img_align_celeba', '*.jpg'))
    dataset = CelebA_Dataset(data_files)
    dataload = data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)
    for i, inp in enumerate(dataload):
        print(i, inp.size())