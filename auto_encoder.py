from typing import List, Tuple, Union
from utils import ConvBNAct
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        backbone: nn.Module,
        backbone_out: int,
        latent_dim: Union[int, None] = None,
        decoder_channels: List = [512, 256, 128, 64, 3],
        decoder_strategy: str = 'upsampling',
        is_cornets: bool = False,
        **kwargs
    ):
        """Constructor

        Args:
            in_channels (int): Number of channels of the input image.
            latent_dim (Union[int, None], optional): Dimension of the latent vector. If the type is 'int', then the encoder will build a 1D latent vector. If the type is None, then the encoder willd build a 2D latent tensor.
            backbone (nn.Module): pytorch module representing the backbone network used in the encoder
            backbone_out (int): output channels of the backbone network.
            decoder_channels (List, optional): channel configuration of the decoder. Defaults to [512, 256, 128, 64, 3]
            decoder_strategy (str, optional): strategy employed to upsample the decoder through the 5 blocks. Can be either 'upsampling' or 'transpose_conv'.
        """
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(
            backbone,
            backbone_out=backbone_out,
            latent_dim=latent_dim
        )
        self.decoder = Decoder(
            in_channels=backbone_out if latent_dim is None else latent_dim,
            channels=decoder_channels,
            strategy=decoder_strategy,
            is_cornets=is_cornets
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Encoder(nn.Module):
    """
    Module that encapsulates the encoder part of an AutoEncoder model.
    It uses a backbone network to extract a 2D conv feature map and applies a Dense layer
    to create a latent vector. This latent vector is the final output of this module.
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_out: int = None,
        latent_dim: int = None
    ):
        """Constructor

        Args:
            backbone (nn.Module): backbone network to use in the Encoder. The backbone network MUST end with a 2D conv layer.
            latent_dim (int, optional): size of the output latent vector. If None, keeps the 2D feature maps of the backbone network as a latent tensor.
            backbone_out (int, optional): dimension of the flattened backbone output feature map. If None, keeps the 2D feature maps of the backbone network as a latent tensor.
        """
        super(Encoder, self).__init__()
        self.backbone = backbone

        if latent_dim is None or backbone_out is None:
            self.latent_module = None
        else:
            self.latent_module = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=backbone_out, out_features=latent_dim)
            )

    def forward(self, x):
        x = self.backbone(x)

        if self.latent_module is not None:
            x = self.latent_module(x)

        return x


class Decoder(nn.Module):
    """Module that represents the Decoder module"""

    def __init__(
        self,
        in_channels: int,
        channels: List = [512, 256, 128, 64, 3],
        strategy: str = 'upsampling',
        is_cornets: bool = False
    ):
        """Constructor

        Args:
            in_channels (int): input channels
            channels (List, optional): List of int values representing the channels for each of the 5 decoder's blocks. Defaults to [512, 256, 128, 64, 3].
            strategy (str, optional): Strategy to upsample the feature maps through the decoder's blocks. Values available: 'upsampling' or 'transpose_conv'. Defaults to 'upsampling'.
            is_cornets (bool, optional): flag that indicates whether the encoder's backbone is a CoRNet-S architecture. This is basically a dirty trick
        """
        super(Decoder, self).__init__()

        # check params
        if len(channels) != 5:
            raise ValueError(
                f'channels argument must have a length of 5. Got {channels} instead.')

        if strategy not in ['upsampling', 'transpose_conv']:
            raise ValueError(
                f'strategy argument must be \'upsampling\' or \'transpose_conv\'. Got {strategy} instead.')

        blocks = [None, None, None, None, None]
        if strategy == 'upsampling':
            blocks[0] = nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBNAct(channels_in=in_channels,
                          channels_out=channels[0], kernel_size=3)
            )
            blocks[1] = nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBNAct(
                    channels_in=channels[0], channels_out=channels[1], kernel_size=3)
            )
            blocks[2] = nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBNAct(
                    channels_in=channels[1], channels_out=channels[2], kernel_size=3)
            )
            blocks[3] = nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBNAct(
                    channels_in=channels[2], channels_out=channels[3], kernel_size=3)
            )
            blocks[4] = nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBNAct(
                    channels_in=channels[3], channels_out=channels[4], kernel_size=3)
            )
        else:
            # TRANSPOSE CONVOLUTION STRATEGY
            blocks[0] = ConvBNAct(
                channels_in=in_channels, channels_out=channels[0], kernel_size=2, is_transposed=True, stride=2)
            blocks[1] = ConvBNAct(
                channels_in=channels[0], channels_out=channels[1], kernel_size=2, is_transposed=True, stride=2)
            blocks[2] = ConvBNAct(
                channels_in=channels[1], channels_out=channels[2], kernel_size=2, is_transposed=True, stride=2)
            blocks[3] = ConvBNAct(
                channels_in=channels[2], channels_out=channels[3], kernel_size=2, is_transposed=True, stride=2)
            blocks[4] = ConvBNAct(
                channels_in=channels[3], channels_out=channels[4], kernel_size=2, is_transposed=True, stride=2)

        self.is_cornets = True

        self.decoder = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.decoder(x)
        if self.is_cornets:
            x = F.interpolate(x, scale_factor=0.5)
        return x
