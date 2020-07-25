__author__ = "Sereda"
import argparse
import os
import numpy as np

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch

from DC_GAN.create_dataset import ECGDataset
from DC_GAN.saver import save_batch_to_images, save_models, save_training_curves

Log = False

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=2501, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=30, help="size of the batches")
parser.add_argument("--patch_len", type=int, default=512, help="size of ecg patch, need to be degree of 2")
parser.add_argument("--latent_dim", type=int, default=60, help="dimensionality of the latent space")
parser.add_argument("--channels", type=int, default=3, help="number of channels in ecg, no more than 12")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--report_interval", type=int, default=200, help="interval (num of batches done) between reports")
parser.add_argument("--model_interval", type=int, default=1000, help="interval (num of batches done) between saving the model")
opt = parser.parse_args()


class Generator(nn.Module):
    """
    Model with 2 upsemplings (x2). I.e. upscale in conv block = 2x2=4 times
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.init_len = opt.patch_len // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_len))

        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(128),

            nn.Upsample(scale_factor=2, mode='linear'),
            nn.Conv1d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode='linear'),
            nn.Conv1d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, opt.channels, 3, stride=1, padding=1),

        )


    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_len)
        if Log: print("after first in generator, size" + str(out.size()))
        out = self.conv_block(out)
        if Log: print("last   in generator, size" + str(out.size()))
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def downscale_block(in_filters, out_filters, bn=True):
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
        print("lenght of downsampled ecg patch in discriminator =" + str(ds_len))
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_len, 1), nn.Sigmoid())

    def forward(self, ecg):
        out = self.model(ecg)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

def train():
    print(opt)

    part_id = 0
    PATH = "C:\\!mywork\\datasets\\BWR_data_schiller\\"
    json_file = PATH + "data_part_" + str(part_id) + ".json"

    dataset_object = ECGDataset(json_file, opt.patch_len)
    dataloader = torch.utils.data.DataLoader(dataset_object,
        batch_size=opt.batch_size,
        shuffle=True
    )
    os.makedirs("images", exist_ok=True)
    cuda = True if torch.cuda.is_available() else False

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        if Log: print("Cuda is avaliable.")

    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
        if Log: print("Weights are reinitialized.")

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    # ----------
    #  Training
    # ----------

    loss_values_G = []
    loss_values_D = []

    for epoch in range(opt.n_epochs):
        epoch_D_loss = 0.0
        epoch_G_loss = 0.0
        for i, batch in enumerate(dataloader):
            print ("batch size:" + str(batch.size()))

            # fake/real labels
            valid = Variable(Tensor(batch.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(batch.shape[0], 1).fill_(0.0), requires_grad=False)

            real_ecgs = Variable(batch.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of fake ecgs
            z = Variable(Tensor(np.random.normal(0, 1, (batch.shape[0], opt.latent_dim))))
            if Log: print("generated noise of shape:" + str(z.size()))
            fake_ecgs = generator(z)


            # Generator wants discriminator to say "valid" on fake ecgs
            g_loss = adversarial_loss(discriminator(fake_ecgs), valid)

            # Logging
            epoch_G_loss += g_loss.item()

            g_loss.backward()

            # We change only Generator here, while Discriminator is fixed
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Discriminator wants to say "valid" for real ecgs and to say "fake" for fake ecgs
            real_loss = adversarial_loss(discriminator(real_ecgs), valid)
            fake_loss = adversarial_loss(discriminator(fake_ecgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            # Logging
            epoch_D_loss += d_loss.item()

            d_loss.backward()

            # We do not change generator to minimise the above loss, only change discriminator
            optimizer_D.step()

            # ------------------------
            #  Report current results
            # -----------------------
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.report_interval == 0:
                save_batch_to_images(batches_done, fake_ecgs[:5].detach().cpu().numpy())

            if batches_done % opt.model_interval == 0:
                save_models(batches_done, generator, discriminator)

        # Logging
        loss_values_G.append(epoch_G_loss / len(dataloader))
        loss_values_D.append(epoch_D_loss / len(dataloader))

    # -----------------
    # End of training
    # -----------------
    save_models("LAST", generator, discriminator)
    save_training_curves(G_losses=loss_values_G, D_losses=loss_values_D)

if __name__ == "__main__":
    train()







