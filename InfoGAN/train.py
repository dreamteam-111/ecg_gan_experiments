import argparse
import os
import numpy as np
import math
import itertools
from typing import NamedTuple

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from InfoGAN.dataset_creator import ECGDataset
from InfoGAN.generator import Generator, save_pics
from InfoGAN.discriminator import Discriminator
from InfoGAN.saver import save_models, save_training_curves


class Params(NamedTuple):
    experiment_folder: str
    # params of data generator
    step_size: int = 25      # num discrets in one step
    max_steps_left: int = 2  # num steps from patch center, allowed for the moving complex
    n_classes: int = 5       # number of classes for dataset
    patch_len: int = 256     # size of ecg patch, need to be degree of 2
    num_channels: int = 1    # "number of channels in ecg, no more than 12"

    # params of model
    code_dim: int = 2
    latent_dim: int = 36

    # params of training
    lr: float = 0.0002    #adam: learning rate
    b1: float = 0.5
    b2: float = 0.999     #adam: decay of first order momentum of gradient
    batch_size: int = 12
    n_epochs: int = 2501

    # params of logger
    save_pic_interval: int = 30 # in epoches
    save_model_interval: int = 30 # in epoches

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def train(opt):
    best_loss = None
    cuda = True if torch.cuda.is_available() else False

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


    # Loss functions
    adversarial_loss = torch.nn.MSELoss()
    categorical_loss = torch.nn.CrossEntropyLoss()
    continuous_loss = torch.nn.MSELoss()

    # Loss weights
    lambda_cat = 1
    lambda_con = 0.1


    # Initialize generator and discriminator
    generator = Generator(latent_dim=opt.latent_dim,
                          n_classes=opt.n_classes, code_dim=opt.code_dim,
                          patch_len=opt.patch_len,
                          num_channels=opt.num_channels)
    discriminator = Discriminator(n_classes=opt.n_classes,
                                  code_dim=opt.code_dim,
                                  patch_len=opt.patch_len,
                                  num_channels=opt.num_channels)


    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        categorical_loss.cuda()
        continuous_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)


    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_info = torch.optim.Adam(
        itertools.chain(generator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )

    dataset_object = ECGDataset(opt.patch_len,
                                    max_steps_left=opt.max_steps_left,
                                    step_size=opt.step_size,
                                    num_leads=opt.num_channels)
    dataloader = torch.utils.data.DataLoader(dataset_object,
            batch_size=opt.batch_size,
            shuffle=True
        )
    # ----------
    #  Training
    # ----------
    for epoch in range(opt.n_epochs):
        for i, (ecgs, labels) in enumerate(dataloader):

            batch_size = ecgs.shape[0]
            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_ecgs = Variable(ecgs.type(FloatTensor))
            #labels = to_categorical(labels.numpy(), num_columns=opt.n_classes)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z, _, label_input_one_hot, code_input = generator.sample_input_numpy(batch_size)

            label_input_one_hot = Variable(FloatTensor(label_input_one_hot))
            code_input = Variable(FloatTensor(code_input))
            z = Variable(FloatTensor(z))

            # Generate a batch of images
            gen_ecgs = generator(z, label_input_one_hot, code_input)
            # Loss measures generator's ability to fool the discriminator
            validity, _, _ = discriminator(gen_ecgs)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()
            # Loss for real images
            real_pred, _, _ = discriminator(real_ecgs)
            d_real_loss = adversarial_loss(real_pred, valid)

            # Loss for fake images
            fake_pred, _, _ = discriminator(gen_ecgs.detach())
            d_fake_loss = adversarial_loss(fake_pred, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # ------------------
            # Information Loss
            # ------------------

            optimizer_info.zero_grad()
            # Sample labels

            z, label_input_int, label_input_one_hot, code_input = generator.sample_input_numpy(batch_size)
            z = Variable(FloatTensor(z))
            code_input = Variable(FloatTensor(code_input))
            gt_labels = Variable(LongTensor(label_input_int), requires_grad=False)
            label_input_one_hot = Variable(FloatTensor(label_input_one_hot))


            gen_ecgs = generator(z, label_input_one_hot, code_input)
            _, pred_label, pred_code = discriminator(gen_ecgs)

            info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(
                pred_code, code_input
            )

            info_loss.backward()
            optimizer_info.step()

            # report current result etc..
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), info_loss.item())
            )

            if epoch % opt.save_model_interval == 0:
                filename = "epoch_" + str(epoch)
                folder = opt.experiment_folder + "/checkpoints"
                if best_loss is None:
                    best_loss = info_loss.item()
                else:
                    if best_loss <= info_loss.item():
                        best_loss = info_loss.item()
                        save_models(filename, folder, generator, discriminator)

            if epoch % opt.save_pic_interval == 0:
                filename = str(epoch) + "_epoch"
                folder = opt.experiment_folder + "/img"
                save_pics(filename, folder, generator)


    # at the end of training
    filename = "LAST"
    folder = opt.experiment_folder + "/checkpoints"
    save_models(filename, folder, generator, discriminator)


if __name__ == "__main__":
    parameters = Params("experiment")
    print(parameters._asdict())
    train(parameters)
