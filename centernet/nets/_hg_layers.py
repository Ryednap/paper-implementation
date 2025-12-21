from typing import List, Type, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class CornerPool(nn.Module):

    cummax_dim_flip = {
        "bottom": (2, False),
        "left": (3, True),
        "right": (3, False),
        "top": (2, True),
    }

    def __init__(self, mode):
        super().__init__()
        self.dim, self.flip = self.cummax_dim_flip[mode]

    def forward(self, x: torch.Tensor):
        if self.flip:
            x = x.flip(self.dim)

        pool_tensor, _ = torch.cummax(x, dim=self.dim)

        if self.flip:
            pool_tensor = pool_tensor.flip(self.dim)

        return pool_tensor


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


class CascadeCornerPool(nn.Module):
    def __init__(self, dim: int, pool1: CornerPool, pool2: CornerPool):
        super().__init__()

        self.conv1 = Conv(kernel=3, inp_dim=dim, out_dim=128)
        self.look_conv1 = Conv(kernel=3, inp_dim=dim, out_dim=128)
        self.p1_look_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)

        self.conv2 = Conv(kernel=3, inp_dim=dim, out_dim=128)
        self.look_conv2 = Conv(kernel=3, inp_dim=dim, out_dim=128)
        self.p2_look_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)

        self.pool1 = pool1
        self.pool2 = pool2

        self.pooled_conv = nn.Conv2d(128, dim, kernel_size=3, padding=1, bias=False)
        self.pooled_bn = nn.BatchNorm2d(dim)

        self.skip_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.skip_bn = nn.BatchNorm2d(dim)

        self.post_pool_conv = Conv(kernel=3, inp_dim=dim, out_dim=dim)

    def forward(self, x: torch.Tensor):
        # MY DOUBTS
        # the `self.pool2(self.look_conv1(x))` looks like standard conv-bn-relu-pool
        # the `self.conv1(x)` is also fine and then we concatenate.
        # why are we have `p1_look_conv` as just conv? why not remove this or why not trivial conv-bn-relu?
        # I think we cannot remove p1-look_conv because in functional form it would correspond to two consecutive pool
        # i.e `self.pool1(self.conv(x)) + self.pool1(self.pool2(...))`

        pool1 = self.conv1(x) + self.pool2(self.look_conv1(x))
        pool1 = self.pool1(self.p1_look_conv(pool1))

        pool2 = self.conv2(x) + self.pool1(self.look_conv2(x))
        pool2 = self.pool2(self.p2_look_conv(pool2))

        pooled_x = self.pooled_bn(self.pooled_conv(pool1 + pool2))
        skip_x = self.skip_bn(self.skip_conv(x))

        out = F.relu(pooled_x + skip_x, inplace=True)
        out = self.post_pool_conv(out)
        return out


class CenterPool(nn.Module):
    """
    Faster implementation of center pool.
    It's faster than consecutive top-bottom and left-right.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.pool1_conv = Conv(kernel=3, inp_dim=dim, out_dim=128)
        self.pool2_conv = Conv(kernel=3, inp_dim=dim, out_dim=128)

        self.pooled_conv = nn.Sequential(
            nn.Conv2d(128, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.skip = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.post_pool_conv = Conv(kernel=3, inp_dim=dim, out_dim=dim)

    def forward(self, x: torch.Tensor):
        # vertical pool
        p1 = self.pool1_conv(x)
        pool1 = torch.max(p1, dim=2, keepdim=True)[0].expand_as(p1)

        # horizontal pool
        p2 = self.pool2_conv(x)
        pool2 = torch.max(p2, dim=3, keepdim=True)[0].expand_as(p2)

        pooled_x = self.pooled_conv(pool1 + pool2)
        skip_x = self.skip(x)

        return self.post_pool_conv(F.relu(pooled_x + skip_x, inplace=True))


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


class Hourglass(nn.Module):
    def __init__(self, n: int, dims: List[int], num_modules: List[int]):
        super().__init__()

        self.n = n

        curr_num_modules = num_modules[0]
        next_num_modules = num_modules[1]
        curr_dim = dims[0]
        next_dim = dims[1]

        # skip connection from current recursion to next recursion
        # Note that each recursion does all the down-sampling and what not
        # but the output has the same spatial dim as the input
        self.skip = make_layer(
            kernel=3,
            inp_dim=curr_dim,
            out_dim=curr_dim,
            num_modules=curr_num_modules,
            layer=Residual,
        )

        # ideally this should be max-pool and in
        # self.low1 the stride should be 1 instead of 2
        self.down = nn.Identity()

        self.low1 = make_layer(
            kernel=3,
            inp_dim=curr_dim,
            out_dim=next_dim,
            num_modules=curr_num_modules,
            stride=2,
            layer=Residual,
        )

        if self.n > 1:
            self.low2 = Hourglass(n - 1, dims[1:], num_modules[1:])
        else:
            self.low2 = make_layer(
                kernel=3,
                inp_dim=next_dim,
                out_dim=next_dim,
                num_modules=next_num_modules,
                layer=Residual,
            )

        self.low3 = make_layer_revr(
            kernel=3,
            inp_dim=next_dim,
            out_dim=curr_dim,
            num_modules=curr_num_modules,
            layer=Residual,
        )

        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x: torch.Tensor):

        down = self.down(x)
        low1 = self.low1(down)  # [curr_dim -> next_dim]
        low2 = self.low2(low1)  # [next_dim -> ... -> next_dim]
        low3 = self.low3(low2)  # [next_dim -> curr_dim]
        up = self.up(low3)

        return self.skip(x) + up


class HourglassStack(nn.Module):
    """
    A single stack of centernet hourglass which support attaching a head
    and also interstack connect layer to connect two stacks using skip connection.
    x ----> stack ---> stack2
        |          ^
        |__________|
    """

    def __init__(
        self,
        n: int,
        dims: List[int],
        num_modules: List[int],
        num_classes: int = 80,
        conv_dim=256,
        inter_stack_connect=False,
    ):

        super().__init__()
        curr_dim = dims[0]
        self.inter_stack_connect = inter_stack_connect
        self.hg_backbone = Hourglass(n, dims, num_modules)
        self.conv = Conv(kernel=3, inp_dim=curr_dim, out_dim=conv_dim)

        if inter_stack_connect:
            self.inter_ = nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim),
            )
            self.cnvs_ = nn.Sequential(
                nn.Conv2d(conv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim),
            )
            self.inter = Residual(3, curr_dim, curr_dim)

        self.tl_pool = CascadeCornerPool(
            dim=conv_dim,
            pool1=CornerPool(mode="top"),
            pool2=CornerPool(mode="left"),
        )
        self.br_pool = CascadeCornerPool(
            dim=conv_dim,
            pool1=CornerPool(mode="bottom"),
            pool2=CornerPool(mode="right"),
        )
        self.ct_pool = CenterPool(dim=conv_dim)

        # Heatmap Heads
        self.tl_hmap = self.make_head(conv_dim, curr_dim, out_dim=num_classes)
        self.br_hmap = self.make_head(conv_dim, curr_dim, out_dim=num_classes)
        self.ct_hmap = self.make_head(conv_dim, curr_dim, out_dim=num_classes)

        self.tl_hmap[-1].bias.data.fill_(-2.19)
        self.br_hmap[-1].bias.data.fill_(-2.19)
        self.ct_hmap[-1].bias.data.fill_(-2.19)

        # Offset heads
        self.tl_regs = self.make_head(conv_dim, curr_dim, out_dim=2)
        self.br_regs = self.make_head(conv_dim, curr_dim, out_dim=2)
        self.ct_regs = self.make_head(conv_dim, curr_dim, out_dim=2)

        # Embeddings Head
        self.tl_embd = self.make_head(conv_dim, curr_dim, out_dim=1)
        self.br_embd = self.make_head(conv_dim, curr_dim, out_dim=1)

    @staticmethod
    def make_head(inp_dim: int, mid_dim: int, out_dim: int):
        return nn.Sequential(
            Conv(kernel=3, inp_dim=inp_dim, out_dim=mid_dim, with_bn=False),
            nn.Conv2d(mid_dim, out_dim, (1, 1)),
        )

    def forward(self, x: torch.Tensor, attach_head=False):

        out = self.hg_backbone(x)  # curr_dim --> ....... -> curr_dim
        out = self.conv(out)  # curr_dim --> conv_dim

        head_out = None
        if attach_head:
            tl_out = self.tl_pool(out)
            br_out = self.br_pool(out)
            ct_out = self.ct_pool(out)
            head_out = {
                "tl_hmap": self.tl_hmap(tl_out),
                "br_hmap": self.br_hmap(br_out),
                "ct_hmap": self.ct_hmap(ct_out),
                "tl_regs": self.tl_regs(tl_out),
                "br_regs": self.br_regs(br_out),
                "ct_regs": self.ct_regs(ct_out),
                "tl_embd": self.tl_embd(tl_out),
                "br_embd": self.br_embd(br_out),
            }

        if self.inter_stack_connect:
            # Merge the information from previous stack (x) and current stack (out)
            out = F.relu(self.inter_(x) + self.cnvs_(out), inplace=True)
            out = self.inter(out)

        return out, head_out
