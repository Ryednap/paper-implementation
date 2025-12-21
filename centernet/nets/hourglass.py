import torch
import torch.nn as nn

from typing import Any, List, cast
import lightning as L

from configs.config import Config
from ._hg_layers import Conv, Residual, HourglassStack
from loss import centernet_loss


class CenterNetHourglass(nn.Module):
    def __init__(
        self,
        n: int,
        nstack: int,
        dims: List[int],
        num_modules: List[int],
        num_classes: int,
        conv_dim: int,
        deep_supervision: bool = True,
    ):
        super().__init__()

        self.dims = dims
        self.deep_supervision = deep_supervision

        self.pre_conv = nn.Sequential(
            Conv(kernel=7, inp_dim=3, out_dim=128, stride=2),
            Residual(kernel=3, inp_dim=128, out_dim=self.dims[0], stride=2),
        )

        self.hg_stacks = nn.ModuleList(
            [
                HourglassStack(
                    n=n,
                    dims=cast(list, dims),
                    num_modules=cast(list, num_modules),
                    num_classes=num_classes,
                    conv_dim=conv_dim,
                    inter_stack_connect=True,
                )
                for _ in range(nstack - 1)
            ]
        )

        self.final_stack = HourglassStack(
            n=n,
            dims=cast(list, dims),
            num_modules=cast(list, num_modules),
            num_classes=num_classes,
            conv_dim=conv_dim,
            inter_stack_connect=False,
        )


    def forward(self, x: torch.Tensor):
        out = self.pre_conv(x)

        outputs = []
        attach_head = self.training and self.deep_supervision
        for hg_stack in self.hg_stacks:
            out, stack_out_feat = hg_stack(out, attach_head=attach_head)
            if stack_out_feat is not None:
                outputs.append(stack_out_feat)

        # always attach head to the final stack
        _, stack_out_feat = self.final_stack(out, attach_head=True)
        outputs.append(stack_out_feat)
        return outputs

