import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from InfoGAN.dataset_creator import ECGDataset

parser = argparse.ArgumentParser()

# Describe movements of the center of QRS (or  other) complex from center of the patch
parser.add_argument("--step_size", type=int, default=20, help="num discrets in one step")
parser.add_argument("--max_steps_left", type=int, default=2, help="num steps from patch center, allowed for the moving complex")

# состав входа для генератора
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--code_dim", type=int, default=2, help="latent code")
parser.add_argument("--n_classes", type=int, default=5, help="number of classes for dataset")

# настройки обучения
parser.add_argument("--n_epochs", type=int, default=2501, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=12, help="size of the batches")
parser.add_argument("--patch_len", type=int, default=256, help="size of ecg patch, need to be degree of 2")
parser.add_argument("--channels", type=int, default=3, help="number of channels in ecg, no more than 12")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

# насттройки логгера
parser.add_argument("--report_interval", type=int, default=200, help="interval (num of batches done) between reports")
parser.add_argument("--model_interval", type=int, default=1000, help="interval (num of batches done) between saving the model")
opt = parser.parse_args()

print(opt)

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return Variable(FloatTensor(y_cat))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        input_dim = opt.latent_dim + opt.n_classes + opt.code_dim

        self.init_len = opt.patch_len // 4
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_len))


        self.conv_block = nn.Sequential(
            # nn.BatchNorm1d(128),

            nn.Upsample(scale_factor=2, mode='linear'),
            nn.Conv1d(128, 128, 3, stride=1, padding=1),
            # nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode='linear'),
            nn.Conv1d(128, 64, 3, stride=1, padding=1),
            # nn.BatchNorm1d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, opt.channels, 3, stride=1, padding=1),
        )

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_len)

        out = self.conv_block(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def downscale_block(in_filters, out_filters, bn=False):
            block = [nn.Conv1d(in_filters, out_filters, 9, 2, 4), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.25)]
            if bn:
                block.append(nn.BatchNorm1d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *downscale_block(opt.channels, 16, bn=False),
            *downscale_block(16, 32),
            *downscale_block(32, 64),
            *downscale_block(64, 128),
        )

        # The lenght of downsampled ecg patch
        ds_len = opt.patch_len // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_len, 1))
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_len, opt.n_classes), nn.Softmax())
        self.latent_layer = nn.Sequential(nn.Linear(128 * ds_len, opt.code_dim))

    def forward(self, ecg):
        out = self.model(ecg)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code


# Loss functions
adversarial_loss = torch.nn.MSELoss()
categorical_loss = torch.nn.CrossEntropyLoss()
continuous_loss = torch.nn.MSELoss()

# Loss weights
lambda_cat = 1
lambda_con = 0.1


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()


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
                                selected_leads=['i', 'ii', 'iii'])
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
        labels = to_categorical(labels.numpy(), num_columns=opt.n_classes)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        label_input = to_categorical(np.random.randint(0, opt.n_classes, batch_size), num_columns=opt.n_classes)
        code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

        # Generate a batch of images
        gen_ecgs = generator(z, label_input, code_input)
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
        sampled_labels = np.random.randint(0, opt.n_classes, batch_size)
        gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)

        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        label_input = to_categorical(sampled_labels, num_columns=opt.n_classes)

        # Sample code
        code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

        gen_ecgs = generator(z, label_input, code_input)
        _, pred_label, pred_code = discriminator(gen_ecgs)

        info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(
            pred_code, code_input
        )

        info_loss.backward()
        optimizer_info.step()
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), info_loss.item())
        )



