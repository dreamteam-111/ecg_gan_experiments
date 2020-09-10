import argparse
import os
import numpy as np
import math
import itertools
#gkf

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn

import torch
from Representation.latent_interpolate import get_interpolation
from InfoGAN.saver import savefig_first_lead_several_axs

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

    def get_same_code_all_classes(self):
        # батч длиной n_classes
        # latent и code выбираются рандомно
        # а классы перебируются все до единого  - т.е внутри батча различие между ними лишь в классе
        z = np.random.normal(0, 1, (self.n_classes, self.latent_dim))
        code_input = np.random.uniform(-1, 1, (self.n_classes, self.code_dim))
        all_labels = np.array([num for num in range(self.n_classes)])
        all_label_one_hot = to_categorical(all_labels, self.n_classes)
        return z, code_input, all_label_one_hot


    def get_same_class_codei_vary(self, class_id, code_i, steps):
        # батч длиной steps
        # класс фиксирован
        # переменная ci (i-тая по счету среди непрерывных) пробегает диапазон
        # остальные c равны нулю
        z = np.random.normal(0, 1, (steps, self.latent_dim))
        label = to_categorical(np.array([class_id]), self.n_classes)
        labels_one_hot = np.repeat(label, steps, axis=0)


        p1 = np.array([0 for _ in range(self.code_dim)])
        p1[code_i] = -1
        p2 = np.array([0 for _ in range(self.code_dim)])
        p2[code_i] = 1
        code_input = get_interpolation(steps, p1, p2)
        code_input = code_input.transpose((1,0))
        return z, code_input, labels_one_hot

def save_pics(filename, folder, generator):
    os.makedirs(folder, exist_ok=True)
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # варьируем дискретную переменную при постоянных континуальных
    z, code_input, all_label_one_hot = generator.get_same_code_all_classes()

    all_label_one_hot = Variable(FloatTensor(all_label_one_hot))
    code_input = Variable(FloatTensor(code_input))
    z = Variable(FloatTensor(z))

    fake_ecgs = generator(noise=z, labels=all_label_one_hot, code=code_input).detach().cpu().numpy()
    savefig_first_lead_several_axs(fake_ecgs, folder, filename, title="same code, all classes")


    # варьируем континуальую переменную при фиксированном классе
    code_i = 0
    steps = 10
    for class_id in range(generator.n_classes):
        folder_path =  folder + "/" + str(class_id) + "_fixed"
        z, code_input, labels_one_hot = generator.get_same_class_codei_vary(class_id, code_i, steps)
        labels_one_hot = Variable(FloatTensor(labels_one_hot))
        code_input = Variable(FloatTensor(code_input))
        z = Variable(FloatTensor(z))
        fake_ecgs = generator(noise=z, labels=labels_one_hot, code=code_input).detach().cpu().numpy()
        savefig_first_lead_several_axs(fake_ecgs, folder_path, filename, title="vary code " + str(code_i))






if __name__ == "__main__":
    generator = Generator(latent_dim=7,
                          n_classes=5, code_dim=2,
                          patch_len=512,
                          num_channels=3)
    z, code_input, labels_one_hot = generator.get_same_class_codei_vary(class_id=3, code_i=1, steps=10)
    print (labels_one_hot)
    print(code_input)