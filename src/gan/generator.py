from __future__ import print_function
import torch.nn as nn


class DCGAN_generator(nn.Module):
    def __init__(self):
        super(DCGAN_generator, self).__init__()
        self.noise_vector_size = 100
        self.fm_depth = 64
        self.img_nc = 3

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.noise_vector_size, self.fm_depth * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.fm_depth * 8),
            nn.ReLU(True),
            # state size. (fm_depth * 8) x 4 x 4
            # output depth -> 64 *  8 = 512
            nn.ConvTranspose2d(self.fm_depth * 8, self.fm_depth * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 4),
            nn.ReLU(True),
            # state size. (fm_depth*4) x 8 x 8
            # output depth -> 64 *  4 = 256
            nn.ConvTranspose2d(self.fm_depth * 4, self.fm_depth * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 2),
            nn.ReLU(True),
            # state size. (fm_depth*2) x 16 x 16
            # output depth -> 64 *  2 = 128
            nn.ConvTranspose2d(self.fm_depth * 2, self.fm_depth, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth),
            nn.ReLU(True),
            # state size. (fm_depth) x 32 x 32
            # output depth -> 64 *  8 = 64
            nn.ConvTranspose2d(self.fm_depth, self.img_nc, 4, 2, 1, bias=False),
            nn.Tanh())
            # state size. (img_nc) x 64 x 64
            # output depth -> img_nc = 3

    def forward(self, input_data):
        return self.main(input_data)


class RasGAN_generator(nn.Module):
    def __init__(self):
        super(RasGAN_generator, self).__init__()
        self.noise_vector_size = 100
        self.fm_depth = 64
        self.img_nc = 3

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.noise_vector_size, self.fm_depth * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.fm_depth * 8),
            nn.ReLU(True),
            # state size. (fm_depth * 8) x 4 x 4
            # output depth -> 64 *  8 = 512
            nn.ConvTranspose2d(self.fm_depth * 8, self.fm_depth * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 4),
            nn.ReLU(True),
            # state size. (fm_depth*4) x 8 x 8
            # output depth -> 64 *  4 = 256
            nn.ConvTranspose2d(self.fm_depth * 4, self.fm_depth * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 2),
            nn.ReLU(True),
            # state size. (fm_depth*2) x 16 x 16
            # output depth -> 64 *  2 = 128
            nn.ConvTranspose2d(self.fm_depth * 2, self.fm_depth, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth),
            nn.ReLU(True),
            # state size. (fm_depth) x 32 x 32
            # output depth -> 64 *  8 = 64
            nn.ConvTranspose2d(self.fm_depth, self.img_nc, 4, 2, 1, bias=False),
            nn.Tanh())
            # state size. (img_nc) x 64 x 64
            # output depth -> img_nc = 3

    def forward(self, input_data):
        return self.main(input_data)
