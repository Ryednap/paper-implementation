import torch
import torch.nn as nn

from lib._cpools import TopPool, LeftPool, RightPool, BottomPool


class pool(nn.Module):
    """
    x --------------------(1x1 Conv-BN) ---------------------------
    |                                                             |
    |--->  (3x3 Conv-bn-relu) -> pool1 _                           |
    |                                   \                          |
    |                                   (+) --> (1x1 Conv-BN) ---> (+) --> ReLU -> (3x3 Conv-bn-relu)
    |                                   /
    |--->  (3x3 Conv-bn-relu) -> pool2 -

    """

    def __init__(self, dim: int, pool1, pool2):
        super().__init__()

        self.p1_conv1 = convolution(k=3, inp_dim=dim, out_dim=128)
        self.p2_conv1 = convolution(k=3, inp_dim=dim, out_dim=128)

        self.p_conv1 = nn.Conv2d(128, dim, kernel_size=(1, 1), bias=False)
        self.p_bn1 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(k=3, inp_dim=dim, out_dim=dim)

        self.pool1 = pool1()
        self.pool2 = pool2()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # pool1
        p1_conv1 = self.p1_conv1(x)
        pool1 = self.pool1(p1_conv1)

        # pool2
        p2_conv1 = self.p2_conv1(x)
        pool2 = self.pool2(p2_conv1)

        # pool1 + pool2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1 = self.p_bn1(p_conv1)

        # shortcut
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        return self.conv2(relu1)


class convolution(nn.Module):
    """
    Typical CONV-BN-RELU
    Convolution module with same padding meaning the spatial dimension is unchanged.
    """

    def __init__(self, k: int, inp_dim: int, out_dim: int, stride=1, with_bn=True):
        super().__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(
            inp_dim,
            out_dim,
            kernel_size=(k, k),
            padding=(pad, pad),
            stride=(stride, stride),
            bias=not with_bn,
        )
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu


class residual(nn.Module):
    """
    x --------------------->(3x3 Conv-BN-ReLU) ------------->(3x3 Conv-BN) -----------------
    |                                                                                       |
    |                                                                                       |
    |----------------------->(Skip Connection) -------------------------------------------- (+) --> ReLU -->
    """

    def __init__(self, k: int, inp_dim: int, out_dim: int, stride=1, with_bn=True):
        super().__init__()

        self.conv1 = nn.Conv2d(
            inp_dim, out_dim, (3, 3), padding=(1, 1), stride=stride, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)

        if stride != 1 or inp_dim != out_dim:
            self.skip = nn.Sequential(
                nn.Conv2d(inp_dim, out_dim, (1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_dim),
            )
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        skip = self.skip(x)

        return self.relu(bn2 + skip)


# inp_dim -> out_dim -> ..... -> out_dim
def make_layer(k: int, inp_dim: int, out_dim: int, modules: int, layer, stride=1):
    layers = [layer(k, inp_dim, out_dim, stride=stride)]
    layers += [layer(k, out_dim, out_dim) for _ in range(modules - 1)]
    return nn.Sequential(*layers)


# inp_dim -> inp_dim -> ..... -> out_dim
def make_layer_revr(k: int, inp_dim: int, out_dim: int, modules: int, layer):
    layers = [layer(k, inp_dim, inp_dim) for _ in range(modules - 1)]
    layers.append(layer(k, inp_dim, out_dim))
    return nn.Sequential(*layers)


def make_kp_layer(cnv_dim: int, curr_dim: int, out_dim: int):
    return nn.Sequential(
        convolution(k=3, inp_dim=cnv_dim, out_dim=curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1)),
    )


class kp_module(nn.Module):
    def __init__(self, n: int, dims: list[int], modules: list[int]):
        super().__init__()

        self.n = n

        curr_modules = modules[0]
        next_modules = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        # creates curr_modules times residual layers with curr_dim channels.
        # Note that this uses identity function to skip
        self.top = make_layer(3, curr_dim, curr_dim, curr_modules, residual)

        # The resolution should have been halved here
        self.down = nn.Sequential()

        # curr_dim -> next_dim -> ...... -> next_dim times curr_modules residual unit
        # The actual resolution is reduced in the first residual unit. Rest use identity skip
        self.low1 = make_layer(3, curr_dim, next_dim, curr_modules, residual, stride=2)

        if self.n > 1:
            # There's still an hourglass in the middle
            self.low2 = kp_module(n - 1, dims[1:], modules[1:])
        else:
            self.low2 = make_layer(3, next_dim, next_dim, next_modules, residual)

        # This bring backs the number of channels to curr_dim
        self.low3 = make_layer_revr(3, next_dim, curr_dim, curr_modules, residual)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upper branch
        up1 = self.top(x)  # Residual of ther upper branch

        # Lower branch
        low1 = self.low1(x)  # Residual of the lower branch after downsampling
        low2 = self.low2(low1)  # Hourglass of the lower branch
        low3 = self.low3(low2)  # Residual of the lower branch before upsampling
        up2 = self.up(low3)  # Upsampling of the lower branch

        return up1 + up2


class exkp(nn.Module):
    def __init__(
        self,
        n: int,
        nstack: int,
        dims: list[int],
        modules: list[int],
        cnv_dim: int,
        num_classes: int,
    ):
        super().__init__()

        self.nstack = nstack

        curr_dim = dims[0]
        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, curr_dim, stride=2),
        )

        self.kps = nn.ModuleList([kp_module(n, dims, modules) for _ in range(nstack)])

        self.cnvs = nn.ModuleList(
            [convolution(3, curr_dim, cnv_dim) for _ in range(nstack)]
        )

        self.inters = nn.ModuleList(
            [convolution(3, curr_dim, curr_dim) for _ in range(nstack - 1)]
        )

        self.inters_ = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                    nn.BatchNorm2d(curr_dim),
                )
                for _ in range(nstack - 1)
            ]
        )

        self.cnvs_ = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(curr_dim, cnv_dim, (1, 1), bias=False),
                    nn.BatchNorm2d(cnv_dim),
                )
                for _ in range(nstack - 1)
            ]
        )

        self.cnvs_tl = nn.ModuleList(
            [pool(cnv_dim, TopPool, LeftPool) for _ in range(nstack)]
        )
        self.cnvs_br = nn.ModuleList(
            [pool(cnv_dim, BottomPool, RightPool) for _ in range(nstack)]
        )

        # heatmap layers
        self.hmap_tl = nn.ModuleList(
            [make_kp_layer(cnv_dim, curr_dim, num_classes) for _ in range(nstack)]
        )
        self.hmap_br = nn.ModuleList(
            [make_kp_layer(cnv_dim, curr_dim, num_classes) for _ in range(nstack)]
        )

        # embedding layers
        self.embd_tl = nn.ModuleList(
            [make_kp_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)]
        )
        self.embd_br = nn.ModuleList(
            [make_kp_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)]
        )

        for hmap_tl, hmap_br in zip(self.hmap_tl, self.hmap_br):
            hmap_tl[-1].bias.data.fill_(-2.19)
            hmap_br[-1].bias.data.fill_(-2.19)

        # regression layers
        self.regs_tl = nn.ModuleList(
            [make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)]
        )
        self.regs_br = nn.ModuleList(
            [make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)]
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        inter = self.pre(inputs)  # (B, curr_dim, H//2, W//2)

        outs = []

        for i in range(self.nstack):
            kp = self.kps[i](inter)  # (B, curr_dim, H//2, W//2)
            cnv = self.cnvs[i](kp)  # (B, cnv_dim, H//2, W//2)

            if self.training or i == self.nstack - 1:
                cnv_tl = self.cnvs_tl[i](cnv)
                cnv_br = self.cnvs_br[i](cnv)

                hmap_tl, hmap_br = self.hmap_tl[i](cnv_tl), self.hmap_br[i](cnv_br)
                embd_tl, embd_br = self.embd_tl[i](cnv_tl), self.embd_br[i](cnv_br)
                regs_tl, regs_br = self.regs_tl[i](cnv_tl), self.regs_br[i](cnv_br)

                outs.append([hmap_tl, hmap_br, embd_tl, embd_br, regs_tl, regs_br])

            if i < self.nstack - 1:
                inter = self.inters_[i](inter) + self.cnvs_[i](cnv)
                inter = self.relu(inter)
                inter = self.inters[i](inter)

        return outs


get_hourglass = {
    "large_hourglass": exkp(
        n=5,
        nstack=2,
        dims=[256, 256, 384, 384, 384, 512],
        modules=[2, 2, 2, 2, 2, 4],
        cnv_dim=256,
        num_classes=80,
    ),
    "small_hourglass": exkp(
        n=5,
        nstack=1,
        dims=[256, 256, 384, 384, 384, 512],
        modules=[2, 2, 2, 2, 2, 4],
        cnv_dim=256,
        num_classes=80,
    ),
    "tiny_hourglass": exkp(
        n=5,
        nstack=1,
        dims=[256, 128, 256, 256, 256, 384],
        modules=[2, 2, 2, 2, 2, 4],
        cnv_dim=256,
        num_classes=80,
    ),
}


