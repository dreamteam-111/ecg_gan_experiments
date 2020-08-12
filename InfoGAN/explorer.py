__author__="Sereda"
import os
import random
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from DC_GAN_near_center.train import Generator, Discriminator, opt
from DC_GAN_near_center.saver import save_ecg_to_explore, save_ecgs_varied_one_plot, save_ecgs_varied_several_plots
from Representation.latent_interpolate import get_interpolation


exp_parser = argparse.ArgumentParser()
exp_parser.add_argument("--left_bound", type=float, default=-1, help="left bound of variation of latent var")
exp_parser.add_argument("--right_bound", type=float, default=1, help="right bound of variation of latent var")
exp_parser.add_argument("--num_steps", type=int, default=15, help="num steps of variation of latent var")
exp_parser.add_argument("--normal", type=bool, default=False, help="z is noise from normal, or z is zeros")
exp_parser.add_argument("--eval", type=bool, default=True, help="eval or train regime of generator|discriminator")
eopt = exp_parser.parse_args()

def get_ci_varied_over_zeros(i):
    """ Return numpy matrix with inserted column, inserted at i-th position"""
    zeros = np.zeros((eopt.num_steps, opt.latent_dim-1))
    values = np.linspace(eopt.left_bound, eopt.right_bound, eopt.num_steps)
    b = np.insert(zeros, i, values, axis=1)
    return b

def get_ci_varied_over_normal(i):
    """ Return numpy matrix with inserted column, inserted at i-th position"""
    zeros = np.random.normal(0,1,(eopt.num_steps, opt.latent_dim-1))
    values = np.linspace(eopt.left_bound, eopt.right_bound, eopt.num_steps)
    b = np.insert(zeros, i, values, axis=1)
    return b


def get_trained_models(id="LAST"):
    """
    Explore (visualise) representation of the model from the folder "images" named "id".tar
    :param id: name of .tar file from images.
    :return:
    """

    # Open saved model
    path = "images/" + str(id) + '.tar'
    checkpoint = torch.load(path)

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    # Load trained weights
    generator.load_state_dict(checkpoint['G_state_dict'])
    discriminator.load_state_dict(checkpoint['D_state_dict'])

    # Dropout, Batchnorm, etc - switch to evaluation|training mode
    if eopt.eval is True:
        generator.eval()
        discriminator.eval()
    else:
        generator.train()
        discriminator.train()
    return generator, discriminator


def draw_fake_at_zero(generator):
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator.cuda()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # --------------------------
    # draw fake ECG  with z= zeros
    # --------------------------
    z_zero = Variable(Tensor(np.zeros((1, opt.latent_dim))))
    fake_ecg_zero = generator(z_zero)[0]
    save_ecg_to_explore(fake_ecg_zero.detach().cpu().numpy(), "zero.png")

def draw_interpolation(generator, num_steps):
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator.cuda()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    p1 = np.random.normal(0, 1, opt.latent_dim)
    p2 = np.random.normal(0, 1, opt.latent_dim)
    batch = get_interpolation(num_steps, p1, p2)

    batch = Variable(Tensor(batch).transpose_(0,1))
    ecgs_fake_varied = generator(batch).detach().cpu().numpy()
    some_num = random.randint(0, 1111)
    save_ecgs_varied_one_plot(ecgs_fake_varied, "INTERPOL" +str(some_num))
    save_ecgs_varied_several_plots(ecgs_fake_varied, "INTERPOL"+str(some_num))

def draw_variation_at_i_one_latent(generator, index_in_latent):
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator.cuda()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # ---------------------------
    # Show interpolation on c1
    # ---------------------------

    if eopt.normal is True:
        z_varied = get_ci_varied_over_normal(index_in_latent)
    else:
        z_varied = get_ci_varied_over_zeros(index_in_latent)
    z_varied = Variable(Tensor(z_varied))
    ecgs_fake_varied = generator(z_varied).detach().cpu().numpy()
    save_ecgs_varied_one_plot(ecgs_fake_varied, "c" + str(index_in_latent))
    save_ecgs_varied_several_plots(ecgs_fake_varied, "c" + str(index_in_latent))

if __name__ == "__main__":
    os.makedirs("explore", exist_ok=True)
    print("experiment  infogan visualisation")
    generator, discriminator = get_trained_models("17000")

    draw_fake_at_zero(generator)
    for i in range(25):
        draw_interpolation(generator, num_steps=6)

    for i in range(0, opt.latent_dim,15):
        draw_variation_at_i_one_latent(generator, i)
