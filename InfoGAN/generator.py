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

import torch


# Здесь описываем генератор
# Все настраиваемые вещи ему в конструктор
# доп методы - сохранение чекпоинтов и виуализаций

def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0
    return y_cat

class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, code_dim, patch_len, num_channels):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.code_dim = code_dim
        self.num_channels = num_channels

        input_dim = latent_dim + n_classes + code_dim

        self.init_len = patch_len // 4
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

            nn.Conv1d(64, num_channels, 3, stride=1, padding=1),
        )


    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_len)

        out = self.conv_block(out)
        return out


    def sample_input_numpy(self, batch_size):
        z = np.random.normal(0, 1, (batch_size, self.latent_dim))

        label_input_int = np.random.randint(0, self.n_classes, batch_size)
        label_input_one_hot = to_categorical(label_input_int, num_columns=self.n_classes)

        code_input = np.random.uniform(-1, 1, (batch_size, self.code_dim))
        return z, label_input_int, label_input_one_hot, code_input
