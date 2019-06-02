import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
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
    net.apply(init_func)


def ScaleShift(inputs):
    z, beta, gamma = inputs
    for _ in range(z.dim() - 2):
        beta = beta.unsqueeze(2)
        gamma = gamma.unsqueeze(2)
    return z * (gamma + 1) + beta

class SelfModulatedBatchNormalization(nn.Module):
    def __init__(self, out_ch, c, z_dim=128):
        super(SelfModulatedBatchNormalization, self).__init__()
        self.num_hidden = z_dim
        self.dim = out_ch
        self.c = c
        self.bn = nn.BatchNorm2d(self.dim, momentum=0.0, affine=False, track_running_stats=False)
        self.linear_beta1 = nn.Linear(self.c.size()[1], self.num_hidden)
        self.linear_beta2 = nn.Linear(self.num_hidden, self.dim)
        self.linear_gamma1 = nn.Linear(self.c.size()[1], self.num_hidden)
        self.linear_gamma2 = nn.Linear(self.num_hidden, self.dim)
        init.normal_(self.linear_beta1.weight.data, 0.0, 0.02)
        init.normal_(self.linear_beta2.weight.data, 0.0, 0.02)
        init.normal_(self.linear_gamma1.weight.data, 0.0, 0.02)
        init.normal_(self.linear_gamma2.weight.data, 0.0, 0.02)

    def forward(self, h):
        h = self.bn(h)
        beta = self.linear_beta1(self.c)
        beta = F.relu(beta, inplace=True)
        beta = self.linear_beta2(beta)
        gamma = self.linear_gamma1(self.c)
        gamma = F.relu(gamma, inplace=True)
        gamma = self.linear_gamma2(gamma)
        return ScaleShift([h, beta, gamma])


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
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x


class MyConvTranspose(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MyConvTranspose, self).__init__()
        self.convtranspose = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)
        init.normal_(self.convtranspose.weight.data, 0.0, 0.02)
    
    def forward(self, x):
        return self.convtranspose(x)


class G_Model(nn.Module):
    def __init__(self, use_sm_bn=False, device='cuda', img_dim=128, z_dim=128):
        super(G_Model, self).__init__()
        self.img_dim = img_dim
        self.z_dim = z_dim
        self.use_sm_bn = use_sm_bn
        self.device = device
        self.max_num_channels = img_dim * 8
        self.num_layers = int(np.log2(img_dim)) - 3
        self.f_size = img_dim // 2**(self.num_layers + 1)
        self.linear = nn.Linear(z_dim, self.f_size**2 * self.max_num_channels)
        # self.module_list = []
        # if use_sm_bn:
        #     self.module_list.append(SelfModulatedBatchNormalization(self.max_num_channels))
        # else:
        #     self.module_list.append(nn.BatchNorm2d(self.max_num_channels))
        # self.module_list.append(nn.ReLU(inplace=True))
        # self.in_ch = self.max_num_channels
        # for i in range(self.num_layers):
        #     self.out_ch = self.max_num_channels // 2**(i + 1)
        #     self.module_list.append(self._convTransposeBlock(self.in_ch, self.out_ch))
        #     self.in_ch = self.out_ch
        # self.module_list.append(nn.ConvTranspose2d(self.in_ch, 3, 4, 2, 1))
        # self.module_list.append(nn.Tanh())
        # self.conv_transpose = nn.ModuleList(self.module_list)
        
    def forward(self, z_in):
        z = self.linear(z_in)
        z = z.view(z.size()[0], self.max_num_channels, self.f_size, self.f_size)
        if self.use_sm_bn:
            z = SelfModulatedBatchNormalization(self.max_num_channels, z_in).to(self.device)(z)
        else:
            z = nn.BatchNorm2d(self.max_num_channels).to(self.device)(z)
        z = nn.ReLU(inplace=True).to(self.device)(z)
        self.in_ch = self.max_num_channels
        for i in range(self.num_layers):
            self.out_ch = self.max_num_channels // 2**(i + 1)
            z = MyConvTranspose(self.in_ch, self.out_ch).to(self.device)(z)
            if self.use_sm_bn:
                z = SelfModulatedBatchNormalization(self.out_ch, z_in).to(self.device)(z)
            else:
                z = nn.BatchNorm2d(self.out_ch).to(self.device)(z)
            z = nn.ReLU(inplace=True).to(self.device)(z)
            self.in_ch = self.out_ch
        z = MyConvTranspose(self.in_ch, 3).to(self.device)(z)
        z = nn.Tanh().to(self.device)(z)
        return z


if __name__ == '__main__':
    z = torch.randn(4, 128).to('cuda')
    img = torch.randn(4, 3, 128, 128).to('cuda')
    e_model = E_Model().to('cuda')
    g_model = G_Model(use_sm_bn=True).to('cuda')
    print(g_model(z).size())
    print(e_model(img).size())