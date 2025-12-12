from typing import List, Type, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.cpools import TopPool, BottomPool, LeftPool, RightPool


CornerNetPool = Type[Union[TopPool, BottomPool, LeftPool, RightPool]]


class Conv(nn.Module):
    """
    Typical Conv - BN (optional) - ReLU
    """

    def __init__(
        self,
        kernel: int,
        inp_dim: int,
        out_dim: int,
        stride: int = 1,
        with_bn: bool = True,
    ):
        super().__init__()

        self.pad = (kernel - 1) // 2
        self.conv = nn.Conv2d(
            in_channels=inp_dim,
            out_channels=out_dim,
            kernel_size=kernel,
            stride=stride,
            padding=self.pad,
            bias=not with_bn,
        )
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        return self.relu(self.bn(self.conv(x)))


class Pool(nn.Module):
    def __init__(self, dim: int, pool1: CornerNetPool, pool2: CornerNetPool):
        super().__init__()

        # x -> conv -> pool
        self.pool1 = nn.Sequential(Conv(kernel=3, inp_dim=dim, out_dim=128), pool1())
        self.pool2 = nn.Sequential(Conv(kernel=3, inp_dim=dim, out_dim=128), pool2())

        # pool1 + pool2 -> post_pool
        self.post_pool = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=dim, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(dim),
        )

        self.skip = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=3, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.conv = Conv(kernel=3, inp_dim=3, out_dim=3)

    def forward(self, x: torch.Tensor):

        pooled_x = self.post_pool(self.pool1(x) + self.pool2(x))
        skip_x = self.skip(x)
        return self.conv(F.relu(pooled_x + skip_x, inplace=True))


class Residual(nn.Module):
    """
    Residual Module

    |------------ conv or Identity-----|
    |                                  |
    x ---> conv-bn-relu ---> conv-bn --(+) -> relu -> out
    """

    def __init__(
        self,
        kernel: int,
        inp_dim: int,
        out_dim: int,
        stride: int = 1,
        with_bn: bool = True,
    ):
        super().__init__()

        self.pre_conv = Conv(
            kernel=kernel,
            inp_dim=inp_dim,
            out_dim=out_dim,
            stride=stride,
            with_bn=with_bn,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=out_dim,
                out_channels=out_dim,
                kernel_size=kernel,
                padding=self.pre_conv.pad,
                bias=False,
            ),
            nn.BatchNorm2d(out_dim),
        )

        if stride == 1 and inp_dim == out_dim:
            self.skip = nn.Identity()
        else:
            # we need transformation to match the channels
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels=inp_dim,
                    out_channels=out_dim,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_dim),
            )

    def forward(self, x: torch.Tensor):
        out = self.pre_conv(x)
        out = self.conv(out)

        return F.relu(out + self.skip(x), inplace=True)


def make_layer(
    kernel: int,
    inp_dim: int,
    out_dim: int,
    num_modules: int,
    layer: Type[Conv | Residual],
    stride: int = 1,
):
    # inp_dim -> out_dim -> ....... -> out_dim
    layers = [layer(kernel, inp_dim, out_dim, stride=stride)]
    layers += [
        layer(kernel, out_dim, out_dim, stride=1) for _ in range(num_modules - 1)
    ]
    return nn.Sequential(*layers)


def make_layer_revr(
    kernel: int,
    inp_dim: int,
    out_dim: int,
    num_modules: int,
    layer: Type[Conv | Residual],
):
    # inp_dim -> inp_dim -> .......... -> inp_dim -> out_dim
    layers = [layer(kernel, inp_dim, inp_dim) for _ in range(num_modules - 1)]
    layers.append(layer(kernel, inp_dim, out_dim))
    return nn.Sequential(*layers)
