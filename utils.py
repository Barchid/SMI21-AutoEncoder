import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ConvBNAct(nn.Sequential):
    """Sequential module to implement the Conv2D+BatchNorm+Activation block
    """

    def __init__(self, channels_in, channels_out, kernel_size, activation=nn.ReLU(inplace=True), dilation=1, stride=1, use_bn=True, is_transposed=False):
        """Constructor

        Args:
            channels_in (int): input channel number
            channels_out (int): output channel number
            kernel_size (int|tuple): Size the conv kernel
            activation (any, optional): activation function. Defaults to nn.ReLU(inplace=True).
            dilation (int, optional): dilation rate parameter of the conv layer. Defaults to 1.
            stride (int, optional): stride parameter of the conv layer. Defaults to 1.
            use_bn (bool, optional): flag that indicates whether a batch norm is used
        """
        super(ConvBNAct, self).__init__()
        padding = kernel_size // 2 + dilation - 1
        if is_transposed:
            self.add_module('conv', nn.ConvTranspose2d(channels_in, channels_out,
                                                       kernel_size=kernel_size,
                                                       padding=0, # No padding
                                                       bias=not use_bn,  # No bias if Batch Normalization is used
                                                       dilation=dilation,
                                                       stride=stride))
        else:
            self.add_module('conv', nn.Conv2d(channels_in, channels_out,
                                              kernel_size=kernel_size,
                                              padding=padding,
                                              bias=not use_bn,  # No bias if Batch Normalization is used
                                              dilation=dilation,
                                              stride=stride))
        if use_bn:
            self.add_module('bn', nn.BatchNorm2d(channels_out))
        self.add_module('act', activation)


def get_backbone_from_model(model, n=1):
    """Creates a new backbone network from the original in parameter where last n layers from the original backbone are removed.
    This function is useful to adapt a pretrained model from the torchvision.models package.

    Args:
        model (nn.Module): the base classification network that contains the last FC layer to remove
        n (int, optional): the number of last layers that will be removed. Default to 1 (because generally, the last layer is a Dense one)

    Returns:
        nn.Module: the newly created backbone module without an FC layer as the last layer.
    """
    return torch.nn.Sequential(*(list(model.children())[:-n]))
