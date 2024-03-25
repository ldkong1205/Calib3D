# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Tuple

from mmengine.model import BaseModule
from torch import Tensor, nn

from mmdet3d.registry import MODELS
from mmdet3d.utils import OptMultiConfig


class BasicBlock(BaseModule):

    def __init__(self,
                 inplanes: int,
                 planes: Sequence[int],
                 bn_d: float = 0.1,
                 init_cfg: OptMultiConfig = None) -> None:
        super(BasicBlock, self).__init__(init_cfg)
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0], momentum=bn_d)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1], momentum=bn_d)
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out += residual
        return out


@MODELS.register_module()
class RangeNet(BaseModule):

    def __init__(self,
                 in_channels: int = 5,
                 init_cfg: OptMultiConfig = None) -> None:
        super(RangeNet, self).__init__(init_cfg)

        self.bn_d = 0.01
        self.blocks = [1, 2, 8, 8, 4]
        self.strides = [2, 2, 2, 2, 2]

        # input layer
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, momentum=self.bn_d)
        self.relu1 = nn.LeakyReLU(0.1)

        self.enc1 = self._make_enc_layer(BasicBlock, [32, 64], self.blocks[0], stride=self.strides[0], bn_d=self.bn_d)
        self.enc2 = self._make_enc_layer(BasicBlock, [64, 128], self.blocks[1], stride=self.strides[1], bn_d=self.bn_d)
        self.enc3 = self._make_enc_layer(BasicBlock, [128, 256], self.blocks[2], stride=self.strides[2], bn_d=self.bn_d)
        self.enc4 = self._make_enc_layer(BasicBlock, [256, 512], self.blocks[3], stride=self.strides[3], bn_d=self.bn_d)
        self.enc5 = self._make_enc_layer(BasicBlock, [512, 1024], self.blocks[4], stride=self.strides[4], bn_d=self.bn_d)

        self.dec5 = self._make_dec_layer(BasicBlock, [1024, 512], bn_d=self.bn_d, stride=self.strides[4])
        self.dec4 = self._make_dec_layer(BasicBlock, [512, 256], bn_d=self.bn_d, stride=self.strides[3])
        self.dec3 = self._make_dec_layer(BasicBlock, [256, 128], bn_d=self.bn_d, stride=self.strides[2])
        self.dec2 = self._make_dec_layer(BasicBlock, [128, 64], bn_d=self.bn_d, stride=self.strides[1])
        self.dec1 = self._make_dec_layer(BasicBlock, [64, 32], bn_d=self.bn_d, stride=self.strides[0])

        self.dropout = nn.Dropout2d(0.05)

    def _make_enc_layer(self, block: nn.Module, planes: Sequence[int], blocks: int, stride: int, bn_d: float = 0.1) -> nn.Sequential:
        layers = []

        layers.append(nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=[1, stride], dilation=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(planes[1], momentum=bn_d))
        layers.append(nn.LeakyReLU(0.1))

        inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(block(inplanes, planes, bn_d))
        return nn.Sequential(*layers)

    def _make_dec_layer(self, block: nn.Module, planes: Sequence[int], bn_d: float = 0.1, stride=2) -> nn.Sequential:
        layers = []

        if stride == 2:
            layers.append(nn.ConvTranspose2d(planes[0], planes[1], kernel_size=[1, 4], stride=[1, 2], padding=[0, 1]))
        else:
            layers.append(nn.Conv2d(planes[0], planes[1], kernel_size=3, padding=1))

        layers.append(nn.BatchNorm2d(planes[1], momentum=bn_d))
        layers.append(nn.LeakyReLU(0.1))

        layers.append(block(planes[1], planes, bn_d))
        return nn.Sequential(*layers)

    def run_enc(self, x: Tensor, layer: nn.Module, skips: dict, os: int) -> tuple:
        y = layer(x)
        if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
            skips[os] = x.detach()
            os *= 2
        x = y
        return x, skips, os

    def run_dec(self, x: Tensor, layer: nn.Module, skips: dict, os: int) -> tuple:
        feats = layer(x)  # up
        if feats.shape[-1] > x.shape[-1]:
            os //= 2
            feats = feats + skips[os].detach()
        x = feats
        return x, skips, os

    def forward(self, x: Tensor) -> Tuple[Tensor]:

        skips = {}
        os = 1

        # first layer
        x, skips, os = self.run_enc(x, self.conv1, skips, os)
        x, skips, os = self.run_enc(x, self.bn1, skips, os)
        x, skips, os = self.run_enc(x, self.relu1, skips, os)

        # all encoder blocks with intermediate dropouts
        x, skips, os = self.run_enc(x, self.enc1, skips, os)
        x, skips, os = self.run_enc(x, self.dropout, skips, os)
        x, skips, os = self.run_enc(x, self.enc2, skips, os)
        x, skips, os = self.run_enc(x, self.dropout, skips, os)
        x, skips, os = self.run_enc(x, self.enc3, skips, os)
        x, skips, os = self.run_enc(x, self.dropout, skips, os)
        x, skips, os = self.run_enc(x, self.enc4, skips, os)
        x, skips, os = self.run_enc(x, self.dropout, skips, os)
        x, skips, os = self.run_enc(x, self.enc5, skips, os)
        x, skips, os = self.run_enc(x, self.dropout, skips, os)

        x, skips, os = self.run_dec(x, self.dec5, skips, os)
        x, skips, os = self.run_dec(x, self.dec4, skips, os)
        x, skips, os = self.run_dec(x, self.dec3, skips, os)
        x, skips, os = self.run_dec(x, self.dec2, skips, os)
        x, skips, os = self.run_dec(x, self.dec1, skips, os)

        x = self.dropout(x)

        return tuple([x])