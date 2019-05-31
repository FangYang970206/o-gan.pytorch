import os
import argparse
import warnings
from time import gmtime, strftime
from os.path import join, exists
from glob import glob
warnings.filterwarnings("ignore")

import torch
import imageio
import numpy as np
from model import *
from tensorboardX import SummaryWriter
from dataset import CelebA_Dataset
from trainer import Trainer
from torch.utils.data import DataLoader


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset_path', type=str, default='F:\\img_align_celeba')
    parse.add_argument('--batch_size', type=int, default=64)
    parse.add_argument('--lr', type=float, default=1e-4)
    parse.add_argument('--num_workers', type=int, default=8)
    parse.add_argument('--epochs', type=int, default=1000)
    parse.add_argument('--save_model', type=bool, default=True)
    parse.add_argument('--save_path', type=str, default='logs/')
    parse.add_argument('--sample_n', type=int, default=9)
    parse.add_argument('--z_dim', type=int, default=128)
    parse.add_argument('--img_dim', type=int, default=128)

    args = vars(parse.parse_args())

    if not exists(args['save_path']):
        os.mkdir(args['save_path'])
    
    # pylint: disable=E1101
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # pylint: disable=E1101

    e_model = E_Model().to(device)
    g_model = G_Model().to(device)
    init_weights(e_model)
    init_weights(g_model)

    data_files = glob(join(args['dataset_path'], '*.jpg'))
    dataset = CelebA_Dataset(data_files, args['img_dim'])

    dataload = DataLoader(dataset,
                          batch_size=args['batch_size'],
                          shuffle=True,
                          num_workers=args['num_workers'])

    time_str = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    # writer_path = join(args['save_path'], time_str)
    os.mkdir(join(args['save_path'], time_str))
    writer = SummaryWriter(log_dir=join(args['save_path'], time_str))

    # fixed_noise = np.random.randn(args['sample_n']**2, args['z_dim'])
    # fixed_noise = torch.from_numpy(fixed_noise).to(device)
    fixed_noise = torch.randn(args['sample_n']**2, args['z_dim'])
    figure = np.zeros((args['img_dim'] * args['sample_n'],
                       args['img_dim'] * args['sample_n'], 3))

    for epoch in range(args['epochs']):
        print(f'<Main> epoch{epoch}')
        trainer = Trainer(e_model, g_model, dataload, epoch, 
                          args['lr'], device, writer)
        trainer.train()
        if (epoch+1) % 1 == 0:
            if args['save_model']:
                e_state = e_model.state_dict()
                torch.save(e_state, f'logs/e_state_{epoch}.pth')
                g_state = g_model.state_dict()
                torch.save(g_state, f'logs/g_state_{epoch}.pth')
            samples = g_model(fixed_noise.to(device)).detach().cpu().numpy()
            # print(samples.shape)
            for i in range(args['sample_n']):
                for j in range(args['sample_n']):
                    # print(samples[j+i*args['sample_n']].shape)
                    figure[i * args['img_dim']:(i+1) * args['img_dim'],
                           j * args['img_dim']:(j+1) * args['img_dim'], :] = \
                           samples[j+i*args['sample_n']].transpose(1, 2, 0)
            figure = (figure + 1) / 2 * 255
            figure = np.round(figure, 0).astype('uint8')
            imageio.imwrite(join(args['save_path'], f'{epoch}.jpg'), figure)


if __name__ == '__main__':
    main()