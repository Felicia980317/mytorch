import math

import torch.nn as nn
from mytorch import activations

"""
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.choices = nn.ModuleDict({
                'conv': nn.Conv2d(10, 10, 3),
                'pool': nn.MaxPool2d(3)
        })
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['prelu', nn.PReLU()]
        ])

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x
"""

_activations = nn.ModuleDict(
    [
        ["lrelu", nn.LeakyReLU()],
        ["prelu", nn.PReLU()],
        ["relu", nn.ReLU(inplace=True)],
        ["relu6", nn.ReLU6(inplace=True)],
        ["hswish", activations.h_swish()],
        ["hsigmoid", activations.h_sigmoid()],
    ]
)


class BatchNorm2d(nn.Module):
    def __init__(self, D, momentum=0.01):
        super(BatchNorm2d, self).__init__()
        slef.D = D
        self.momentun = momentun
        self.batch_norm = nn.BatchNorm2d(self.D, momentun=self.momentun)

    def forward(self, x):
        return self.batch_norm(x)


def batchnorm2d(D, momentum=0.01):
    return nn.BatchNorm2d(D, momentum=momentum)


class Conv2d(nn.Module):
    def __init__(
        self,
        in_cha,
        out_cha,
        kernel_size,
        stride=1,
        dilation=1,
        use_bn=True,
        use_bias=False,
        use_activation=True,
        activation="relu6",
    ):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_cha,
            out_cha,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            dilation=dilation,
            groups=1,
            bias=use_bias,
        )
        self.bn = None
        self.activation = None
        if use_activation:
            self.activation = _activations[activation]
        if use_bn:
            self.bn = batchnorm2d(out_cha)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


def conv2d(
    in_cha,
    out_cha,
    kernel_size,
    stride=1,
    dilation=1,
    use_bn=True,
    use_bias=False,
    use_activation=True,
    activation="relu6",
):
    layers = []
    layers.append(
        nn.Conv2d(
            in_cha,
            out_cha,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            dilation=dilation,
            groups=1,
            bias=use_bias,
        )
    )
    if use_bn:
        layers.append(batchnorm2d(out_cha))
    if use_activation:
        layers.append(_activations[activation])

    return nn.Sequential(*layers)


class DepConv2d(nn.Module):
    def __init__(
        self,
        in_cha,
        out_cha,
        kernel_size,
        stride=1,
        dilation=1,
        use_bn=True,
        use_bias=False,
        use_activation=True,
        activation="relu6",
    ):
        super(DepConv2d, self).__init__()
        if in_cha != out_cha:
            raise ValueError("DepConv2d: only support in_cha==out_cha.")
        self.conv = nn.Conv2d(
            in_cha,
            out_cha,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            dilation=dilation,
            groups=out_cha,
            bias=use_bias,
        )

        self.bn = None
        self.activation = None
        if use_activation:
            self.activation = _activations[activation]
        if use_bn:
            self.bn = batchnorm2d(out_cha)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class SepConv2d(nn.Module):
    def __init__(
        self,
        in_cha,
        out_cha,
        kernel_size,
        stride=1,
        dilation=1,
        use_bn=True,
        use_bias=False,
        use_activation=True,
        activation="relu6",
    ):
        super(SepConv2d, self).__init__()
        self.convdw = nn.Conv2d(
            in_cha,
            in_cha,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            dilation=dilation,
            groups=in_cha,
            bias=use_bias,
        )
        self.conv1x1 = nn.Conv2d(
            in_cha,
            out_cha,
            1,
            stride=1,
            padding=0,
            dilation=dilation,
            groups=1,
            bias=use_bias,
        )

        self.bn1 = batchnorm2d(in_cha)
        self.bn2 = None

        self.activation1 = _activations[activation]
        self.activation2 = None

        if use_activation:
            self.activation2 = _activations[activation]
        if use_bn:
            self.bn2 = batchnorm2d(out_cha)

    def forward(self, x):
        x = self.convdw(x)
        x = self.bn1(x)
        x = _activations[self.activation](x)

        x = self.conv1x1(x)
        if self.bn2:
            x = self.bn2(x)
        if self.activation2:
            x = self.activation2(x)
        return x


def sepconv2d(
    in_cha,
    out_cha,
    kernel_size,
    stride=1,
    dilation=1,
    use_bn=True,
    use_bias=False,
    use_activation=True,
    activation="relu6",
):
    layers = []
    layers.append(
        nn.Conv2d(
            in_cha,
            in_cha,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            dilation=dilation,
            groups=in_cha,
            bias=use_bias,
        )
    )
    layers.append(batchnorm2d(in_cha))
    layers.append(_activations[activation])

    layers.append(
        nn.Conv2d(
            in_cha,
            out_cha,
            1,
            stride=1,
            padding=0,
            dilation=dilation,
            groups=1,
            bias=use_bias,
        )
    )
    if use_bn:
        layers.append(batchnorm2d(out_cha))
    if use_activation:
        layers.append(_activations[activation])

    return nn.Sequential(*layers)


def upsample2d(x, size=None, scale_factor=None):
    if not size is None and not scale_factor is None:
        raise ValueError("upsample2d, only support size or scale_factor")
    if size is None and scale_factor is None:
        raise ValueError("upsample2d, set size or set scale_factor")

    return nn.Upsample(
        size=size, scale_factor=scale_factor, mode="bilinear", align_corners=False
    )(x)


def sperable_upsample2d(x, scale_factor):
    up_times = int(math.log(scale_factor, 2))
    if up_times % 1 != 0:
        raise ValueError("sperable_upsample2d, the scale factor != 2 to the nth power")

    upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
    for _ in range(up_times):
        x = upsample(x)
    return x


# * squeeze and excitation layer
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        assert (
            channel > reduction
        ), "Make sure your input channel bigger than reduction which equals to {}".format(
            reduction
        )
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            _activations["relu"],
            nn.Linear(channel // reduction, channel),
            _activations["hsigmoid"],
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
