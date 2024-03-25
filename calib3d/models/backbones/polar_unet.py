# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmengine.model import BaseModule
from torch import Tensor, nn

try:
    from dropblock import DropBlock2D
except ImportError:
    DropBlock2D = None

from calib3d.registry import MODELS
from calib3d.utils import ConfigType, OptConfigType, OptMultiConfig


class DoubleConv(BaseModule):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='LeakyReLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super(DoubleConv, self).__init__(init_cfg=init_cfg)
        self.conv = nn.Sequential(
            build_conv_layer(
                conv_cfg, in_channels, out_channels, 3, padding=1),
            build_norm_layer(norm_cfg, out_channels)[1],
            build_activation_layer(act_cfg),
            build_conv_layer(
                conv_cfg, out_channels, out_channels, 3, padding=1),
            build_norm_layer(norm_cfg, out_channels)[1],
            build_activation_layer(act_cfg))

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return x


class DoubleConvCircular(BaseModule):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='LeakyReLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super(DoubleConvCircular, self).__init__(init_cfg=init_cfg)
        self.conv1 = nn.Sequential(
            build_conv_layer(
                conv_cfg, in_channels, out_channels, 3, padding=(1, 0)),
            build_norm_layer(norm_cfg, out_channels)[1],
            build_activation_layer(act_cfg))
        self.conv2 = nn.Sequential(
            build_conv_layer(
                conv_cfg, out_channels, out_channels, 3, padding=(1, 0)),
            build_norm_layer(norm_cfg, out_channels)[1],
            build_activation_layer(act_cfg))

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = self.conv1(x)
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = self.conv2(x)
        return x


class StemConv(BaseModule):

    def __init__(self,
                 block: nn.Module,
                 in_channels: int,
                 out_channels: int,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='LeakyReLU', inplace=True),
                 pre_norm: bool = False,
                 init_cfg: OptMultiConfig = None) -> None:
        super(StemConv, self).__init__(init_cfg=init_cfg)
        if pre_norm:
            self.conv = nn.Sequential(
                build_norm_layer(norm_cfg, in_channels)[1],
                block(in_channels, out_channels, conv_cfg, norm_cfg, act_cfg))
        else:
            self.conv = block(in_channels, out_channels, conv_cfg, norm_cfg,
                              act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return x


class DownConv(BaseModule):

    def __init__(self,
                 block: nn.Module,
                 in_channels: int,
                 out_channels: int,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='LeakyReLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super(DownConv, self).__init__(init_cfg=init_cfg)
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            block(in_channels, out_channels, conv_cfg, norm_cfg, act_cfg))

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return x


class UpConv(BaseModule):

    def __init__(self,
                 block: nn.Module,
                 in_channels: int,
                 out_channels: int,
                 use_dropblock: bool = False,
                 dropout_ratio: float = 0.5,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='LeakyReLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super(UpConv, self).__init__(init_cfg=init_cfg)

        self.up_conv = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = block(in_channels, out_channels, conv_cfg, norm_cfg,
                          act_cfg)

        self.use_dropblock = use_dropblock
        if self.use_dropblock:
            if DropBlock2D is None:
                raise ImportError('Please run "pip install dropblock" to '
                                  'install dropblock first.')
            self.dropblock = DropBlock2D(block_size=7, drop_prob=dropout_ratio)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up_conv(x)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(
            x,
            (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat((skip, x), dim=1)
        x = self.conv(x)
        if self.use_dropblock:
            x = self.dropblock(x)
        return x


@MODELS.register_module()
class PolarUNet(BaseModule):
    """Backbone for PolarNet.

    Args:
        in_channels (int): Channels of input features.
        output_shape (Tuple[int, int]): Required output shape of features.
    """

    def __init__(self,
                 in_channels: int,
                 output_shape: Tuple[int, int],
                 stem_channels: int = 64,
                 num_stages: int = 4,
                 down_channels: Sequence[int] = (128, 256, 512, 512),
                 up_channels: Sequence[int] = (256, 128, 64, 64),
                 use_dropblock: bool = False,
                 dropout_ratio: float = 0.5,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='LeakyReLU', inplace=True),
                 pre_norm: bool = False,
                 circular_padding: bool = True,
                 init_cfg: OptMultiConfig = None) -> None:
        super(PolarUNet, self).__init__(init_cfg=init_cfg)
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        self.in_channels = in_channels
        assert len(down_channels) == len(up_channels) == 4

        if circular_padding:
            block = DoubleConvCircular
        else:
            block = DoubleConv

        self.stem_conv = StemConv(block, in_channels, stem_channels, conv_cfg,
                                  norm_cfg, act_cfg, pre_norm)

        self.down_block_list = nn.ModuleList()
        down_channels = [stem_channels] + list(down_channels)
        for i in range(num_stages):
            in_channels = down_channels[i]
            out_channels = down_channels[i + 1]
            self.down_block_list.append(
                DownConv(block, in_channels, out_channels, conv_cfg, norm_cfg,
                         act_cfg))

        self.up_block_list = nn.ModuleList()
        up_channels = [down_channels[-1]] + list(up_channels)
        for i in range(num_stages):
            in_channels = down_channels[-i - 2] + up_channels[i]
            out_channels = up_channels[i + 1]
            self.up_block_list.append(
                UpConv(block, in_channels, out_channels, use_dropblock,
                       dropout_ratio, conv_cfg, norm_cfg, act_cfg))

    def forward(self, voxel_features: Tensor, coors: Tensor,
                batch_size: int) -> Tensor:
        voxel_coors = coors.long()

        features = torch.zeros(
            (batch_size, self.ny, self.nx, self.in_channels),
            dtype=voxel_features.dtype,
            device=voxel_features.device)

        features[voxel_coors[:, 0], voxel_coors[:, 1],
                 voxel_coors[:, 2]] = voxel_features
        features = features.permute(0, 3, 1, 2).contiguous()

        features = self.stem_conv(features)

        skip_features = [features]
        for conv in self.down_block_list:
            features = conv(features)
            skip_features.append(features)

        for i, conv in enumerate(self.up_block_list):
            skip = skip_features[-i - 2]
            features = conv(features, skip)

        return features