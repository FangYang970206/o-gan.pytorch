import torch
import torch.nn.init as init
import torch.nn as nn
import numpy as np


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

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
        self.module_list.append(nn.Conv2d(self.in_ch, self.out_ch, 4, 2, 1))
        self.module_list.append(nn.LeakyReLU(0.2, inplace=True))
        
        for i in range(1, self.num_layers + 1):
            self.in_ch = self.out_ch
            self.out_ch = self.max_num_channels // 2**(self.num_layers - i)
            self.module_list.append(self._convBlock(self.in_ch, self.out_ch))

        self.convs = nn.ModuleList(self.module_list)
        self.linear = nn.Linear(self.f_size**2 * self.max_num_channels, 
                                self.z_dim)

    def _convBlock(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        x = x.view(x.size()[0], -1)
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
        self.module_list.append(nn.ConvTranspose2d(self.in_ch, 3, 4, 2, 1))
        self.module_list.append(nn.Tanh())
        self.conv_transpose = nn.ModuleList(self.module_list)
        
    def _convTransposeBlock(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size()[0], self.max_num_channels, self.f_size, self.f_size)
        for layer in self.conv_transpose:
            x = layer(x)
        return x


if __name__ == '__main__':
    z = torch.randn(4, 128).to('cuda')
    img = torch.randn(4, 3, 128, 128).to('cuda')
    e_model = E_Model().to('cuda')
    g_model = G_Model().to('cuda')
    print(g_model(z).size())
    print(e_model(img).size())
    print(e_model.__class__.__name__)