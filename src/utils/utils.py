from __future__ import print_function

import random
import os
from datetime import datetime
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dataset_utils
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torchvision import utils as vutils

from discriminator import *
from generator import *


def load_model(model, model_path):
    model = model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def execute_model(model_path, model_instance, batch_size, noise_v_size, images_save_path, n_gpu=0):
    generator = load_model(model_instance, model_path)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and int(n_gpu) > 0) else "cpu")
    generate_fake_images(generator, batch_size, noise_v_size, device, images_save_path)


def save_model(model, optimizer, epoch, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"{save_path}/{model.__class__.__name__}.pth")


def plot_loss_results(gen_loss, disc_loss):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(gen_loss, label="G")
    plt.plot(disc_loss, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def save_loss_plot(gen_loss, disc_loss, path):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(gen_loss, label="G")
    plt.plot(disc_loss, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{path}/gen_disc_loss.png")


def generate_fake_images(generator, batch_size, noise_v_size, device, images_save_path):
    noise = torch.randn(batch_size, noise_v_size, 1, 1, device=device)
    fake_images = generator(noise).detach().cpu()
    count = 0
    for i in fake_images:
        img_path = f"{images_save_path}/{count}.png"
        vutils.save_image(i, img_path)
        count += 1


def generate_images_from_single_noise(generator, noise_v_size, device, images_save_path, n_samples=100, n_rows=8,
                                      padding=2, norm=True, alpha=0.1):
    noise = torch.randn(1, noise_v_size, 1, 1, device=device)
    count = 0
    for i in range(n_samples):
        fake_image = generator(noise).detach().cpu()
        noise += alpha
        img_path = f"{images_save_path}/{count}.png"
        vutils.save_image(fake_image, img_path)
        count += 1


def generate_images_from_zero_noise(generator, noise_v_size, device, images_save_path, n_samples=100, n_rows=8,
                                    padding=2, norm=True, alpha=0.1):
    noise = torch.randn(1, noise_v_size, 1, 1, device=device).zero_()
    count = 0
    for i in range(n_samples):
        fake_image = generator(noise).detach().cpu()
        noise += alpha
        img_path = f"{images_save_path}/{count}.png"
        vutils.save_image(fake_image, img_path)
        count += 1


def create_report(r_path, s_date, gen_net, disc_net, optimizer, loss_fn, dataset_name, batch_size, image_size, lr,
                  epochs, disc_loss, gen_loss):
    current_date = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
    filename = f"{r_path}/{current_date}.txt"

    with open(filename, 'w') as file:
        file.write(f"Start date:\t{s_date.strftime('%d-%m-%Y_%I-%M-%S_%p')}\n")
        file.write(f"End date:\t{current_date}\n")
        file.write("------------NET CONFIGURATION------------\n")
        file.write("Generator configuration\n")
        file.write(f"{gen_net}\n")
        file.write("Discriminator configuration\n")
        file.write(f"{disc_net}\n")
        file.write("------------LOSS FN / OPTIMIZER------------\n")
        file.write(f"Optimizer\t{optimizer}\n")
        file.write(f"Loss Function:\t{loss_fn}\n")
        file.write("------------Dataset------------\n")
        file.write(f"Data set name:\t{dataset_name}\n")
        file.write(f"Batch Size:\t{batch_size}\n")
        file.write(f"Image Size:\t{image_size}\n")
        file.write("------------Training------------\n")
        file.write(f"Learning rate:\t{lr}\n")
        file.write(f"Epochs:\t{epochs}\n")
        file.write("------------Results------------\n")
        file.write(f"Generator loss:\t{gen_loss}\n")
        file.write(f"Discriminator loss:\t{disc_loss}\n")


def create_result_directories(results_path):
    # Check if results folder path exists, if not create new directory
    __check_path(results_path)
    # Create experiment results folder
    experiment_folder = f"{results_path}/{datetime.now().strftime('%d-%m-%Y_%I-%M-%S_%p')}"
    os.mkdir(experiment_folder)
    report_path = f"{experiment_folder}/report"
    models_path = f"{experiment_folder}/models"
    generated_images_path = f"{experiment_folder}/generated_images"
    plot_path = f"{experiment_folder}/plot"
    # Create nested folder (model, report, generated_images, plot)
    os.mkdir(report_path)
    os.mkdir(models_path)
    os.mkdir(generated_images_path)
    os.mkdir(plot_path)

    return experiment_folder


def create_tracking_directories(track_path):
    # Check if results folder path exists, if not create new directory
    __check_path(track_path)
    track_folder = f"{track_path}/{datetime.now().strftime('%d-%m-%Y_%I-%M-%S_%p')}"
    models = f"{track_folder}/models"
    gen_path = f"{track_folder}/models/generator"
    disc_path = f"{track_folder}/models/discriminator"
    generated_images_path = f"{track_folder}/generated_images"
    # Create nested folder ()
    os.mkdir(track_folder)
    os.mkdir(models)
    os.mkdir(gen_path)
    os.mkdir(disc_path)
    os.mkdir(generated_images_path)

    return track_folder


def __check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
