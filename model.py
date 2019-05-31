import torch.nn as nn
import numpy as np


class E_Model(nn.Module):
    def __init__(self, img_dim=128, z_dim=128):
        super(E_Model, self).__init__()
        self.module_list = []
        self.img_dim = img_dim
        self.z_dim = z_dim
        self.max_num_channels = img_dim * 8
        self.num_layers = int(np.log2(img_dim)) - 3
        self.f_size = img_dim // 2**(self.num_layers + 1)
        self.in_ch = 3
        self.out_ch = 64
        self.module_list.append(nn.Conv2d(self.in_ch, self.out_ch, 5, 2, 2))
        self.module_list.append(nn.LeakyReLU(0.2, inplace=True))
        
        for i in range(1, self.num_layers + 1):
            self.in_ch = self.out_ch
            self.out_ch = self.max_num_channels // 2**(self.num_layers - i)
            self.module_list.append(self._convBlock(self.in_ch, self.out_ch))

        self.main = nn.ModuleList(self.module_list)
        self.linear = nn.Linear(self.f_size**2 * self.max_num_channels, 
                                self.z_dim)

    def _convBlock(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 5, 2, 2),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size[0], -1)
        x = self.linear(x)
        return x


class G_Model(nn.Module):
    def __init__(self, img_dim=128, z_dim=128):
        super(G_Model, self).__init__()
        self.img_dim = img_dim
        self.z_dim = z_dim
        self.max_num_channels = img_dim * 8
        self.num_layers = int(np.log2(img_dim)) - 3
        self.f_size = img_dim // 2**(self.num_layers + 1)
        self.linear = nn.Linear(z_dim, self.f_size**2 * self.max_num_channels)
        self.module_list = []
        self.module_list.append(nn.BatchNorm2d(self.max_num_channels))
        self.module_list.append(nn.ReLU(inplace=True))
        self.in_ch = self.max_num_channels
        for i in range(self.num_layers):
            self.out_ch = self.max_num_channels // 2**(i + 1)
            self.module_list.append(self._convTransposeBlock(self.in_ch, self.out_ch))
            self.in_ch = self.out_ch
        self.module_list.append(nn.Conv2d(self.in_ch, 3, 5, 2, 2))
        self.module_list.append(nn.Tanh)
        self.conv_transpose = nn.ModuleList(self.module_list)
        
    def _convTransposeBlock(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 5, 2, 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size()[0], self.max_num_channels, self.f_size, self.f_size)
        x = self.conv_transpose(x)
        return x