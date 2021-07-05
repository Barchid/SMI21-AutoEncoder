"""Contains several basic backbones"""
from typing import List, OrderedDict
from utils import ConvBNAct, get_backbone_from_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from vonenet import VOneNet


class BaseBackbone(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        channels: List[int] = [32, 64, 128, 256, 512],
        kernel_sizes: List[int] = [7, 3, 3, 3, 3]
    ):
        super(BaseBackbone, self).__init__()
        # check params
        assert len(channels) == 5 and len(kernel_sizes) == 5

        # insert values to iterate
        channels.insert(0, in_channels)
        kernel_sizes.insert(0, 0)

        for i in range(len(channels) - 1):
            self.add_module(
                f'conv_{i}',
                ConvBNAct(
                    channels[i],
                    channels_out=channels[i+1],
                    kernel_size=kernel_sizes[i+1]
                )
            )
            self.add_module(
                f'maxpool_{i}',
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )


class VOneNetBackbone(nn.Module):
    """Some Information about VOneNetBackbone"""

    def __init__(self,
                 sf_corr=0.75, sf_max=9, sf_min=0, rand_param=False, gabor_seed=0,
                 simple_channels=256, complex_channels=256,
                 noise_mode='neuronal', noise_scale=0.35, noise_level=0.07, k_exc=25,
                 model_arch='resnet50', image_size=224, visual_degrees=8, ksize=25, stride=4, n=2):
        super(VOneNetBackbone, self).__init__()
        # Get the original VOneNet classifier
        vonenet = VOneNet(
            sf_corr, sf_max, sf_min, rand_param, gabor_seed,
            simple_channels, complex_channels,
            noise_mode, noise_scale, noise_level, k_exc, model_arch, image_size, visual_degrees,
            ksize, stride
        )

        # retrieve only the backbone (without FC layer at the end)
        self.vonenet_backbone = nn.Sequential(OrderedDict([
            ('vone_block', vonenet.vone_block),
            ('bottleneck', vonenet.bottleneck),
            ('model', get_backbone_from_model(vonenet.model, n=n)),
        ]))
        

    def forward(self, x):
        x = self.vonenet_backbone(x)
        return x
