# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
from mmengine.model import BaseModule
from torch import Tensor, nn

from calib3d.registry import MODELS
from calib3d.utils import OptMultiConfig


class ResContextBlock(BaseModule):

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 init_cfg: OptMultiConfig = None) -> None:
        super(ResContextBlock, self).__init__(init_cfg)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(
            planes, planes, kernel_size=3, dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(planes)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA = self.bn2(resA)

        resA = self.conv3(resA)
        resA = self.act3(resA)
        resA = self.bn3(resA)
        output = shortcut + resA
        return output


class ResBlock(BaseModule):

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 dropout_rate: float,
                 kernel_size: int = 3,
                 stride: int = 1,
                 pooling: bool = True,
                 drop_out: bool = True,
                 init_cfg: OptMultiConfig = None) -> None:
        super(ResBlock, self).__init__(init_cfg)
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(
            planes, planes, kernel_size=3, dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(planes)

        self.conv4 = nn.Conv2d(
            planes, planes, kernel_size=2, dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(planes)

        self.conv5 = nn.Conv2d(planes * 3, planes, kernel_size=1)
        self.act5 = nn.LeakyReLU()
        self.bn5 = nn.BatchNorm2d(planes)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(
                kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn2(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn3(resA)

        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA3 = self.bn4(resA)

        concat = torch.cat([resA1, resA2, resA3], dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn5(resA)
        resA = shortcut + resA

        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)
            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB


class UpBlock(BaseModule):

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 dropout_rate: float,
                 drop_out: bool = True,
                 init_cfg: OptMultiConfig = None) -> None:
        super(UpBlock, self).__init__(init_cfg)
        self.drop_out = drop_out

        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(
            inplanes // 4 + 2 * planes, planes, kernel_size=3, padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(
            planes, planes, kernel_size=2, dilation=2, padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(planes)

        self.conv4 = nn.Conv2d(planes * 3, planes, kernel_size=1)
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(planes)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = torch.cat((upA, skip), dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = self.conv2(upE1)
        upE = self.act2(upE)
        upE2 = self.bn2(upE)

        upE = self.conv3(upE2)
        upE = self.act3(upE)
        upE3 = self.bn3(upE)

        concat = torch.cat((upE1, upE2, upE3), dim=1)
        upE = self.conv4(concat)
        upE = self.act4(upE)
        upE = self.bn4(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE


@MODELS.register_module()
class SalsaNext(BaseModule):

    def __init__(self,
                 in_channels: int = 5,
                 init_cfg: OptMultiConfig = None) -> None:
        super(SalsaNext, self).__init__(init_cfg)

        self.downCntx = ResContextBlock(in_channels, 32)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)

        self.resBlock1 = ResBlock(
            32, 2 * 32, 0.2, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True)
        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False)

        self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, 0.2)
        self.upBlock2 = UpBlock(4 * 32, 4 * 32, 0.2)
        self.upBlock3 = UpBlock(4 * 32, 2 * 32, 0.2)
        self.upBlock4 = UpBlock(2 * 32, 32, 0.2, drop_out=False)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        downCntx = self.downCntx(x)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)

        down0c, down0b = self.resBlock1(downCntx)
        down1c, down1b = self.resBlock2(down0c)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down5c = self.resBlock5(down3c)

        up4e = self.upBlock1(down5c, down3b)
        up3e = self.upBlock2(up4e, down2b)
        up2e = self.upBlock3(up3e, down1b)
        up1e = self.upBlock4(up2e, down0b)
        return tuple([up1e])