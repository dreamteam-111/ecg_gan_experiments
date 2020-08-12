import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable

from math import sqrt

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel,
                 kernel_size1, padding1,
                 kernel_size2, padding2):
        super().__init__()



        self.conv = nn.Sequential(nn.Conv1d(in_channel, out_channel,
                                                kernel_size1, padding=padding1),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv1d(out_channel, out_channel,
                                                    kernel_size2, padding=padding2),
                                        nn.LeakyReLU(0.2))

    def forward(self, input):
        out = self.conv(input)
        return out

class Generator(nn.Module):
    def __init__(self, n_leads=3):
        super().__init__()

        self.label_embed.weight.data.normal_()
        self.progression = nn.ModuleList([ConvBlock(512, 512, 4, 3, 3, 1),
                                          ConvBlock(512, 512, 3, 1),
                                          ConvBlock(512, 512, 3, 1),
                                          ConvBlock(512, 512, 3, 1),
                                          ConvBlock(512, 256, 3, 1),
                                          ConvBlock(256, 128, 3, 1)])

        self.to_ecg = nn.ModuleList([nn.Conv1d(512, n_leads, 1),
                                     nn.Conv1d(512, n_leads, 1),
                                     nn.Conv1d(512, n_leads, 1),
                                     nn.Conv1d(512, n_leads, 1),
                                     nn.Conv1d(256, n_leads, 1),
                                     nn.Conv1d(128, n_leads, 1)])

    def forward(self, input, step=0, alpha=-1):
        for i, (conv, to_ecg) in enumerate(zip(self.progression, self.to_ecg)):
            if i > 0 and step > 0:
                upsample = F.upsample(input, scale_factor=2)
                out = conv(upsample)

            else:
                out = conv(out)

            if i == step:
                current_ecg = to_ecg(out)

                if i > 0 and 0 <= alpha < 1:
                    upsampled_old_ecg = self.to_ecg[i - 1](upsample)
                    out = (1 - alpha) * upsampled_old_ecg + alpha * current_ecg

                break

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.progression = nn.ModuleList([ConvBlock(128, 256, 3, 1),
                                          ConvBlock(256, 512, 3, 1),
                                          ConvBlock(512, 512, 3, 1),
                                          ConvBlock(512, 512, 3, 1),
                                          ConvBlock(512, 512, 3, 1),
                                          ConvBlock(513, 512, 3, 1, 4, 0)])

        self.from_ecg = nn.ModuleList([nn.Conv1d(3, 128, 1),
                                       nn.Conv1d(3, 256, 1),
                                       nn.Conv1d(3, 512, 1),
                                       nn.Conv1d(3, 512, 1),
                                       nn.Conv1d(3, 512, 1),
                                       nn.Conv1d(3, 512, 1)])

        self.n_layer = len(self.progression)
        self.linear = nn.Linear(512, 1)

    def forward(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1): # ...,5,4,...,1,0.
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_ecg[index](input)

            out = self.progression[index](out)

            if i > 0:
                out = F.avg_pool1d(out, 2) # уменьшаем в 2 раза на каждом шаге кроме нулевого

                if i == step and 0 <= alpha < 1:
                    skip_ecg = F.avg_pool1d(input, 2)
                    skip_ecg = self.from_ecg[index + 1](skip_ecg)
                    out = (1 - alpha) * skip_ecg + alpha * out
        # выходной слой дискриминатора
        out = out.squeeze()
        out = self.linear(out)

        return out