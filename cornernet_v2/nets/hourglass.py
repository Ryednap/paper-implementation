from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.cpools import RightPool, TopPool, LeftPool, BottomPool
from .layers import make_layer, make_layer_revr, Residual, Conv, Pool
from loss import Loss


class Hourglass(nn.Module):
    """
    Recursive HourGlass recursive architecture

    Args:
        n (int): Number of recursive layers
        dims (int): The number of conv channels to process in each layer
        num_modules (int): Number of modules of Residual unit in each layer
    """

    def __init__(self, n: int, dims: List[int], num_modules: List[int]):
        super().__init__()

        self.n = n

        curr_num_modules = num_modules[0]
        next_num_modules = num_modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        # skip connection for residual connection from current
        # recurssion to next recurssion
        self.skip = make_layer(
            kernel=3,
            inp_dim=curr_dim,
            out_dim=curr_dim,
            num_modules=curr_num_modules,
            layer=Residual,
        )

        # ideally this should be max-pool and
        # in self.low1 the stride should be 1 instead of 2
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


class CornerNetStack(nn.Module):
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

        self.inter_stack_connect = inter_stack_connect
        curr_dim = dims[0]

        self.hg_backbone = Hourglass(n, dims, num_modules)
        self.conv = Conv(kernel=3, inp_dim=curr_dim, out_dim=conv_dim)

        if inter_stack_connect:
            # For stack level residual connection.
            self.inter_ = nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim),
            )
            self.cnvs_ = nn.Sequential(
                nn.Conv2d(conv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim),
            )
            self.inter = Residual(3, curr_dim, curr_dim)

        self.tl_pool = Pool(dim=conv_dim, pool1=TopPool, pool2=LeftPool)
        self.br_pool = Pool(dim=conv_dim, pool1=BottomPool, pool2=RightPool)

        # Output Head
        self.tl_hmap = nn.Sequential(
            Conv(kernel=3, inp_dim=conv_dim, out_dim=conv_dim, with_bn=False),
            nn.Conv2d(curr_dim, num_classes, (1, 1)),
        )
        self.br_hmap = nn.Sequential(
            Conv(kernel=3, inp_dim=conv_dim, out_dim=conv_dim, with_bn=False),
            nn.Conv2d(curr_dim, num_classes, (1, 1)),
        )

        self.tl_embd = nn.Sequential(
            Conv(kernel=3, inp_dim=conv_dim, out_dim=curr_dim, with_bn=False),
            nn.Conv2d(curr_dim, 1, (1, 1)),
        )
        self.br_embd = nn.Sequential(
            Conv(kernel=3, inp_dim=conv_dim, out_dim=curr_dim, with_bn=False),
            nn.Conv2d(curr_dim, 1, (1, 1)),
        )

        self.tl_regs = nn.Sequential(
            Conv(kernel=3, inp_dim=conv_dim, out_dim=curr_dim, with_bn=False),
            nn.Conv2d(curr_dim, 2, (1, 1)),
        )
        self.br_regs = nn.Sequential(
            Conv(kernel=3, inp_dim=conv_dim, out_dim=curr_dim, with_bn=False),
            nn.Conv2d(curr_dim, 2, (1, 1)),
        )

    def forward(self, x: torch.Tensor, attach_head=False):

        out = self.hg_backbone(x)  # curr_dim --> ....... -> curr_dim
        out = self.conv(out)  # curr_dim --> conv_dim

        if self.inter_stack_connect:
            # Merge the information from previous stack (x) and current stack (out)
            out = F.relu(self.inter_(x) + self.cnvs_(out), inplace=True)
            out = self.inter(out)

        if attach_head:
            return out, {
                "tl_hmap": self.tl_hmap(out),
                "br_hmap": self.br_hmap(out),
                "tl_regs": self.tl_regs(out),
                "br_regs": self.br_regs(out),
                "tl_embd": self.tl_embd(out),
                "br_embd": self.br_embd(out),
            }

        return out, None


class CornerNet(nn.Module):
    def __init__(
        self,
        n: int,
        nstack: int,
        dims: List[int],
        num_modules: List[int],
        num_classes=80,
        conv_dim=256,
        deep_supervision=True,
    ):
        super().__init__()

        self.nstack = nstack
        self.deep_supervision = deep_supervision

        curr_dim = dims[0]

        # AlexNet style larger initial kernel
        self.pre = nn.Sequential(
            Conv(kernel=7, inp_dim=3, out_dim=128, stride=2),
            Residual(kernel=3, inp_dim=128, out_dim=curr_dim, stride=2),
        )

        self.stacks = nn.ModuleList(
            [
                CornerNetStack(
                    n=n,
                    dims=dims,
                    num_modules=num_modules,
                    num_classes=num_classes,
                    conv_dim=conv_dim,
                    inter_stack_connect=True,
                )
                for _ in range(nstack - 1)
            ]
        )
        self.final_stack = CornerNetStack(
            n=n,
            dims=dims,
            num_modules=num_modules,
            num_classes=num_classes,
            conv_dim=conv_dim,
            inter_stack_connect=True,
        )

        self.loss = Loss()

    def forward(self, batch: Dict[str, torch.Tensor]):
        out = self.pre(batch["image"])

        outputs = []
        for stack in self.stacks:
            out, stack_feats = stack(
                out, attach_head=self.training and self.deep_supervision
            )
            if stack_feats:  # when deeep supervision enabled
                outputs.append(stack_feats)

        _, stack_feat = self.final_stack(out, attach_head=True)
        outputs.append(stack_feat)

        if self.training:
            loss = self.loss(batch, outputs)
            return {"outputs": outputs, "loss": loss}

        return {"outputs": outputs}
