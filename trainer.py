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
        self.optimizer_E = RMSprop(self.e_model.parameters(), lr=self.lr)
        self.optimizer_G = RMSprop(self.g_model.parameters(), lr=self.lr)

    def _set_requires_grad(self, net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad

    def _l2_normalize(self, x, axis=-1):
        y = torch.max(torch.sum(x**2, axis, keepdim=True), axis, keepdim=True)[0]
        return x / torch.sqrt(y)

    def _correlation(self, x, y):
        x = x - torch.mean(x, dim=1, keepdim=True)
        y = y - torch.mean(y, dim=1, keepdim=True)
        x = self._l2_normalize(x, 1)
        y = self._l2_normalize(y, 1)
        return torch.sum(x * y, 1, keepdim=True)

    def _forward(self):
        self.x_fake = self.g_model(self.z)
        self.z_fake = self.e_model(self.x_fake)
        self.x_fake_ng = self.x_fake.detach()
        self.z_real = self.e_model(self.x_real)
        self.z_fake_ng = self.e_model(self.x_fake_ng)
        self.z_real_mean = torch.mean(self.z_real, 1, keepdim=True)
        self.z_fake_ng_mean = torch.mean(self.z_fake_ng, 1, keepdim=True)
        self.z_fake_mean = torch.mean(self.z_fake, 1, keepdim=True)

    def backward_G(self):
        # self.t1_loss = self.z_real_mean - self.z_fake_ng_mean
        self.t1_loss = self.z_fake_mean
        self.z_corr_1 = self._correlation(self.z, self.z_fake)
        self.g_loss = torch.mean(self.t1_loss - self.z_corr_1)
        self.g_loss.backward()

    def backward_E(self):
        self.t2_loss = self.z_real_mean - self.z_fake_ng_mean
        # self.t2_loss = self.z_real_mean
        self.qp_loss = 0.25 * self.t2_loss[:, 0] ** 2 / \
                       torch.mean((self.x_real - self.x_fake_ng)**2, dim=[1, 2, 3])
        # self.e_loss = torch.mean(self.t1_loss + self.t2_loss - 0.5 * self.z_corr) + \
        #               torch.mean(self.qp_loss)
        self.z_corr_2 = self._correlation(self.z, self.z_fake_ng)
        self.e_loss = torch.mean(self.t2_loss - 0.5 * self.z_corr_2) + \
                      torch.mean(self.qp_loss)
        self.e_loss.backward()

    def _epoch(self):
        progress = tqdm(total=len(self.dataload.dataset))
        for _, x in enumerate(self.dataload):
            self.z = torch.randn(x.size()[0], 128).to(self.device)
            self.x_real = x.to(self.device)

            # self._set_requires_grad(self.g_model, True)
            # self._set_requires_grad(self.e_model, False)

            self._forward()

            self._set_requires_grad(self.e_model, False)

            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()

            # self._set_requires_grad(self.g_model, False)
            self._set_requires_grad(self.e_model, True)

            self.optimizer_E.zero_grad()
            self.backward_E()
            self.optimizer_E.step()

            self.writer.add_scalar('g_loss', self.g_loss.item(), self.step)
            self.writer.add_scalar('e_loss', self.e_loss.item(), self.step)

            self.step += 1
            progress.update(self.dataload.batch_size)
            progress.set_description(f'g_loss: {self.g_loss.item()}, \
                                       e_loss: {self.e_loss.item()}')

    def train(self):
        self._epoch()

