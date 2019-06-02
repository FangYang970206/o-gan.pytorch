import itertools
import numpy as np
import torch
from torch.optim import RMSprop
from tqdm import tqdm


class Trainer:
    def __init__(self, e_model, g_model, dataload, epoch, lr, device, writer):
        self.e_model = e_model
        self.g_model = g_model
        self.dataload = dataload
        self.epoch = epoch
        self.lr = lr
        self.device = device
        self.writer = writer
        self.step = 0
        self.optimizer = RMSprop(itertools.chain(self.e_model.parameters(), 
                                                 self.g_model.parameters()), lr=self.lr)

    # def _set_requires_grad(self, net, requires_grad=False):
    #     for param in net.parameters():
    #         param.requires_grad = requires_grad

    def _l2_normalize(self, x, axis=-1):
        y = torch.max(torch.sum(x**2, axis, keepdim=True), axis, keepdim=True)[0]
        return x / torch.sqrt(y)

    def _correlation(self, x, y):
        x = x - torch.mean(x, dim=1, keepdim=True)
        y = y - torch.mean(y, dim=1, keepdim=True)
        x = self._l2_normalize(x, 1)
        y = self._l2_normalize(y, 1)
        return torch.sum(x * y, 1, keepdim=True)

    # def _forward(self):
    #     self.x_fake = self.g_model(self.z)
    #     self.z_fake = self.e_model(self.x_fake)
    #     self.x_fake_ng = self.x_fake.detach()
    #     self.z_real = self.e_model(self.x_real)
    #     self.z_fake_ng = self.e_model(self.x_fake_ng)

    # def backward(self):
    #     self.z_real_mean = torch.mean(self.z_real, 1, keepdim=True)
    #     self.z_fake_ng_mean = torch.mean(self.z_fake_ng, 1, keepdim=True)
    #     self.z_fake_mean = torch.mean(self.z_fake, 1, keepdim=True)
    #     self.t1_loss = self.z_real_mean - self.z_fake_ng_mean
    #     self.t2_loss = self.z_fake_mean - self.z_fake_ng_mean
    #     self.z_corr = self._correlation(self.z, self.z_fake)
    #     self.qp_loss = 0.25 * self.t1_loss[:, 0] ** 2 / \
    #                    torch.mean((self.x_real - self.x_fake_ng)**2, dim=[1, 2, 3])
    #     self.z_corr = self._correlation(self.z, self.z_fake_ng)
    #     self.loss = torch.mean(self.t1_loss + self.t2_loss - 0.5 * self.z_corr) + \
    #                   torch.mean(self.qp_loss)
    #     self.loss.backward()

    def _epoch(self):
        progress = tqdm(total=len(self.dataload.dataset))
        for _, x in enumerate(self.dataload):
            z = torch.randn(x.size()[0], 128).to(self.device)
            x_real = x.to(self.device)

            self.optimizer.zero_grad()
            x_fake = self.g_model(z)
            x_fake_ng = x_fake.detach()
            z_fake = self.e_model(x_fake)
            z_real = self.e_model(x_real)
            z_fake_ng = self.e_model(x_fake_ng)

            z_real_mean = torch.mean(z_real, 1, keepdim=True)
            z_fake_ng_mean = torch.mean(z_fake_ng, 1, keepdim=True)
            z_fake_mean = torch.mean(z_fake, 1, keepdim=True)

            t1_loss = z_real_mean - z_fake_ng_mean
            t2_loss = z_fake_mean - z_fake_ng_mean
            z_corr = self._correlation(z, z_fake)
            qp_loss = 0.25 * t1_loss[:, 0] ** 2 / \
                      torch.mean((x_real - x_fake_ng)**2, dim=[1, 2, 3])
            loss = torch.mean(t1_loss + t2_loss - 0.5 * z_corr) + \
                              torch.mean(qp_loss)
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar('t1_loss', torch.mean(t1_loss), self.step)
            self.writer.add_scalar('z_corr', torch.mean(z_corr), self.step)

            self.step += 1
            progress.update(self.dataload.batch_size)
            progress.set_description(f't1_loss: {torch.mean(t1_loss).item()}, \
                                       z_corr: {torch.mean(z_corr).item()}')

    def train(self):
        self._epoch()

