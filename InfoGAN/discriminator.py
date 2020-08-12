import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_classes, code_dim, patch_len, num_channels):
        super(Discriminator, self).__init__()

        def downscale_block(in_filters, out_filters, bn=False):
            block = [nn.Conv1d(in_filters, out_filters, 9, 2, 4), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.25)]
            if bn:
                block.append(nn.BatchNorm1d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *downscale_block(num_channels, 16, bn=False),
            *downscale_block(16, 32),
            *downscale_block(32, 64),
            *downscale_block(64, 128),
        )

        # The lenght of downsampled ecg patch
        ds_len = patch_len // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_len, 1))
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_len, n_classes), nn.Softmax())
        self.latent_layer = nn.Sequential(nn.Linear(128 * ds_len, code_dim))

    def forward(self, ecg):
        out = self.model(ecg)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code