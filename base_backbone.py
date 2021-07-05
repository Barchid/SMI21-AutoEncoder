"""Contains several basic backbones"""
from typing import List
from utils import ConvBNAct
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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