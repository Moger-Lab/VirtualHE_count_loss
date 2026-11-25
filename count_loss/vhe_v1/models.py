"""
Shared model architectures for the morphologically-constrained CycleGAN
project. Imported by every training-stage script -- same self-containment
reasoning as common.py: no dependency on another script's kernel state.
"""

import random

import torch
import torch.nn as nn


# =============================================================================
# WEIGHT INITIALISATION
# =============================================================================

def init_weights(module: nn.Module, init_gain: float = 0.02):
    """Normal(0, init_gain) init for conv/linear weights, matching the
    CycleGAN/pix2pix paper's convention. InstanceNorm affine params (if
    present) get weight~Normal(1, init_gain), bias=0. Apply via
    `model.apply(init_weights)`.
    """
    classname = module.__class__.__name__
    if hasattr(module, "weight") and ("Conv" in classname or "Linear" in classname):
        nn.init.normal_(module.weight.data, 0.0, init_gain)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "InstanceNorm2d" in classname and getattr(module, "affine", False):
        nn.init.normal_(module.weight.data, 1.0, init_gain)
        nn.init.constant_(module.bias.data, 0.0)


# =============================================================================
# GENERATOR (Johnson/Zhu-style ResNet generator, as used in CycleGAN)
# =============================================================================

class ResidualBlock(nn.Module):
    """Two 3x3 reflection-padded convs + InstanceNorm + ReLU, with a skip
    connection. No dropout -- standard for CycleGAN's ResNet generator
    (unlike the U-Net/Pix2Pix decoder, which does use dropout).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)


class UpBlock(nn.Module):
    """Upsamples by 2x. 'nearest_conv' (default) -- nearest-neighbour
    upsample followed by a stride-1 conv -- avoids the checkerboard
    artifacts that stride-2 ConvTranspose2d with a small kernel can
    produce. 'conv_transpose' matches the original CycleGAN paper's
    implementation exactly, kept available for comparison/ablation.
    """
    def __init__(self, in_channels: int, out_channels: int, mode: str = "nearest_conv"):
        super().__init__()
        if mode == "nearest_conv":
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
            )
        elif mode == "conv_transpose":
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2,
                                    padding=1, output_padding=1),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unknown upsample mode: {mode!r}. Use 'nearest_conv' or 'conv_transpose'.")

    def forward(self, x):
        return self.block(x)


class ResnetGenerator(nn.Module):
    """c7s1-ngf, d(ngf*2), d(ngf*4), N x residual blocks, u(ngf*2), u(ngf),
    c7s1-out_channels, tanh. in_channels/out_channels are independent --
    supports asymmetric domains (e.g. 1-channel SRS -> 2-channel H,E).
    """
    def __init__(self, in_channels: int, out_channels: int, ngf: int = 64,
                 n_residual_blocks: int = 9, upsample_mode: str = "nearest_conv"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4, affine=True),
            nn.ReLU(inplace=True),
        ]
        for _ in range(n_residual_blocks):
            layers.append(ResidualBlock(ngf * 4))

        layers += [
            UpBlock(ngf * 4, ngf * 2, mode=upsample_mode),
            UpBlock(ngf * 2, ngf, mode=upsample_mode),

            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# =============================================================================
# DISCRIMINATOR (70x70-receptive-field PatchGAN, Isola/Zhu convention)
# =============================================================================

class PatchGANDiscriminator(nn.Module):
    """Outputs a map of raw logits (no sigmoid) -- paired with an LSGAN
    (MSE) adversarial loss, not binary cross-entropy. Fully convolutional,
    so it works at any input resolution; the receptive field (~70x70) is
    fixed by the architecture, not the input size.
    """
    def __init__(self, in_channels: int, ndf: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)


# =============================================================================
# IMAGE REPLAY BUFFER (Shrivastava et al.; used by CycleGAN to stabilise
# discriminator training)
# =============================================================================

class ImagePool:
    """With some probability, the discriminator sees an OLDER generated
    sample from a buffer instead of only the very latest one -- prevents
    the discriminator from chasing a rapidly-moving target and stabilises
    adversarial training. Standard CycleGAN technique, pool_size=50 in the
    original paper.
    """
    def __init__(self, pool_size: int = 50):
        self.pool_size = pool_size
        self.images = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.pool_size == 0:
            return images
        output = []
        for image in images:
            image = image.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(image.clone())
                output.append(image)
            elif random.random() > 0.5:
                idx = random.randint(0, self.pool_size - 1)
                stored = self.images[idx].clone()
                self.images[idx] = image.clone()
                output.append(stored)
            else:
                output.append(image)
        return torch.cat(output, dim=0)
