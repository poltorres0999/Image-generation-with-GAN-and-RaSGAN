from __future__ import print_function

import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from IPython.display import HTML
from datetime import datetime
from src.gan.generator import *
from src.gan.discriminator import *

from utils import save_model

from src.fid_score.fid import calculate_fid


class GAN:

    def __init__(self, params):
        self.params = params
        self.gen_optimizer = None
        self.disc_optimizer = None
        self.generator_net = None
        self.discriminator_net = None
        self.gan_type = self.params['gan_type']
        self.loss_fn = None
        self.device = None
        self.gen_loss = []
        self.disc_loss = []
        self.generated_images = []
        self.fid_score = []

        self.init_gan(params)

    def init_gan(self, params):
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and int(params["n_gpu"]) > 0) else "cpu")
        self.set_loss_fn()
        self.create_models(params["gen_model"], params["disc_model"], int(params["n_gpu"]))
        self.set_optimizer(params["optimizer"], params['learning_rate'], int(params['beta']))

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def create_models(self, gen_model, disc_model, n_gpu):
        generator = globals()[gen_model]
        discriminator = globals()[disc_model]
        self.generator_net = generator().to(self.device)
        self.discriminator_net = discriminator().to(self.device)

        if (self.device.type == 'cuda') and (n_gpu > 1):
            self.generator_net = nn.DataParallel(self.generator_net, list(range(n_gpu)))
            self.discriminator_net = nn.DataParallel(self.discriminator_net, list(range(n_gpu)))

    def set_loss_fn(self):
        if self.gan_type == "DCGAN":
            self.loss_fn = nn.BCELoss()
        if self.gan_type == "RasGAN":
            self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def set_optimizer(self, optimizer, lr, beta):

        if optimizer == "ADAM":
            self.gen_optimizer = optim.Adam(self.generator_net.parameters(), lr=lr, betas=(beta, 0.999))
            self.disc_optimizer = optim.Adam(self.discriminator_net.parameters(), lr=lr, betas=(beta, 0.999))

    def optimize_discriminator_net(self, real_data, fake_data, real_labels, fake_labels):
        ############################
        # Update Discriminator network
        ###########################
        # -------- Train discriminator with all-real batch ------------
        # Set discriminator gradient values to 0
        self.discriminator_net.zero_grad()
        # Discriminator error calculation using BCELoss
        if self.gan_type == "DCGAN":
            # Calculate D's loss on the all-fake batch batch and D gradients in backward pass
            disc_err_real = self.loss_fn(self.discriminator_net(real_data).view(-1), real_labels)
            disc_err_real.backward()
            # Calculate D's loss on the all-real batch and D gradients in backward pass
            disc_err_fake = self.loss_fn(self.discriminator_net(fake_data.detach()).view(-1), fake_labels)
            disc_err_fake.backward()
            # The disc error is calculated as the sum of the error of the real and fake data
            disc_error = float(disc_err_real) + float(disc_err_fake)
        # Discriminator error calculation using BCEWithLogitsLoss
        if self.gan_type == "RasGAN":
            disc_error = self.loss_fn(self.discriminator_net(real_data).view(-1) -
                                      self.discriminator_net(fake_data).view(-1), real_labels)
            # Calculate gradients for D in backward pass
            disc_error.backward(retain_graph=True)
        # Update discriminator with the calculated gradients
        self.disc_optimizer.step()

        return float(disc_error)

    def optimize_generator_net(self, real_data, fake_data, real_labels):
        ############################
        # Update Generator network
        ###########################
        # Set generator gradient values to 0
        self.generator_net.zero_grad()
        if self.gan_type == "DCGAN":
            # Generator error calculation using BCELoss
            gen_error = self.loss_fn(self.discriminator_net(fake_data).view(-1), real_labels)
            # Calculate gradients for Generator
            gen_error.backward()
        if self.gan_type == "RasGAN":
            # Generator error calculation using BCEWithLogitsLoss
            gen_error = self.loss_fn(self.discriminator_net(fake_data).view(-1) -
                                     self.discriminator_net(real_data).view(-1), real_labels)
            # Calculate gradients for Generator
            gen_error.backward()

        # Update G
        self.gen_optimizer.step()

        return float(gen_error)

    def train(self, data):
        best_fid = 100000
        best_fid_images = ""
        real_label = 1
        fake_label = 0
        # Create batch of latent vectors that we will use to visualize the progression of the generator
        fixed_noise = torch.randn(128, self.params["vector_size"], 1, 1, device=self.device)
        print("Starting Training Loop...")
        # Load models
        # todo: load models
        # For each epoch
        for epoch in range(self.params["num_epochs"]):
            print(f"Started epoch [{epoch}] / [{self.params['num_epochs']}]")
            # For each batch in the data_loader
            # Note -> enumerate outputs (int index, item from iterable object)
            current_epoch = epoch
            for i, train_data in enumerate(data):
                data_length = len(data)
                print(f"Iteration [{i}] / [{data_length}]")
                # ---Prepare data for the current training step---
                real_data = train_data[0].to(self.device)
                # Get batch size
                batch_size = real_data.size(0)
                # Generate batch of latent vectors
                noise = torch.randn(batch_size, self.params["vector_size"], 1, 1, device=self.device)
                # Generate fake image batch with Generator
                fake_data = self.generator_net(noise)
                # Create labels for real (1) and fake (0) data
                real_labels = torch.full((batch_size,), real_label, device=self.device)
                fake_labels = torch.full((batch_size,), fake_label, device=self.device)
                # FID Calculation and FID optimization
                if self.params["fid"]:
                    if i % self.params["fid_fr"] == 0:
                        current_fid = calculate_fid(fake_data, real_data, False, self.params['fid_bs'], False)
                        self.fid_score.append(current_fid)
                        print(f"FID Value: {current_fid}")
                        if self.params['fid_opt']:
                            if current_fid < best_fid:
                                best_fid = current_fid
                                best_fid_images = fake_data
                            else:
                                fake_data = best_fid_images

                disc_error = self.optimize_discriminator_net(real_data, fake_data, real_labels,
                                                             fake_labels)
                # fake labels are real for generator cost (maximize log(D(G(z))))
                gen_error = self.optimize_generator_net(real_data, fake_data, real_labels)
                # Save generator and discriminator loss
                self.disc_loss.append(float(disc_error))
                self.gen_loss.append(float(gen_error))
                # If enabled, print generator and discriminator loss
                self.__print_stats(i, current_epoch, disc_error, gen_error)
                # Check how the generator is doing by saving G's output on fixed_noise
                if self.params['track_path'] is not None:
                    self.__track_training(fixed_noise, self.params["track_fr"], epoch, self.params['num_epochs'],
                                          data_length, i, self.params["track_folder"])

    # todo: add debug
    def __print_stats(self, iteration, epoch, disc_error, gen_error):
        if iteration % 50 == 0:
            # Output training stats
            print(f'[{epoch}/{self.params["num_epochs"]}]]\n'
                  f'Loss_D: {disc_error:.4f}\n'
                  f'Loss_G: {gen_error:.4f}\n')

    def __track_training(self, fixed_noise, store_fr, current_epoch, num_epochs, data_len, iteration, temp_path):
        if (iteration % store_fr == 0) or ((current_epoch == num_epochs - 1) and (iteration == data_len - 1)):
            with torch.no_grad():
                fake_images = self.generator_net(fixed_noise).detach().cpu()
                fake_images_path = f"{temp_path}/generated_images/{datetime.now().strftime('%d-%m-%Y_%I-%M-%S_%p')}_epoch_{current_epoch}_iter_{iteration}.png"
                vutils.save_image(fake_images[0:self.params["track_num_images"]], fake_images_path)
                save_model(self.generator_net, self.gen_optimizer, current_epoch, f"{temp_path}/models/generator")
                save_model(self.discriminator_net, self.disc_optimizer, current_epoch,
                           f"{temp_path}/models/discriminator")
