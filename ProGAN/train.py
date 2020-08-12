from tqdm import tqdm
import torch
from torch import nn, optim
from torch.autograd import Variable, grad

from ProGAN.model import Discriminator, Generator


generator = Generator().cuda()
discriminator = Discriminator().cuda()

g_optimizer = optim.Adam(generator.parameters(), lr=0.001, betas=(0.0, 0.99))
d_optimizer = optim.Adam(
    discriminator.parameters(), lr=0.001, betas=(0.0, 0.99))

def train(generator, discriminator, loader):
    step = 0
    iterations = tqdm(range(600000))

    iteration = 0
    for iteration in iterato:
        alpha = min(1, 0.00002 * iteration)
        dataset = sample_data(loader, 4 * 2 ** step)
        real_image = next(dataset)
