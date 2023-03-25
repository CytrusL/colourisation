import math

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class EqualizedConv2d(nn.Module):
    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            kernel: int,
            stride=1,
            pad=0,
            bias=True,
            groups=1
    ):
        super(EqualizedConv2d, self).__init__()

        self.weight = nn.Parameter(
            torch.randn(out_ch, in_ch // groups, kernel, kernel)
        )
        self.scale = 1 / math.sqrt(in_ch * kernel ** 2)

        self.stride = stride
        self.pad = pad
        self.groups = groups

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))

        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.conv2d(
            x,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.pad,
            groups=self.groups
        )

        return out


class EqualizedLinear(nn.Module):
    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            bias=True,
            bias_init=0,
            lr_mul=1,
    ):
        super(EqualizedLinear, self).__init__()

        self.weight = nn.Parameter(torch.randn(out_ch, in_ch).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_ch).fill_(bias_init))

        else:
            self.bias = None

        self.scale = (1 / math.sqrt(in_ch)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out


class DropPath(nn.Module):
    def __init__(
            self,
            drop_prob,
            training=True
    ):
        super(DropPath, self).__init__()

        self.keep_prob = 1 - drop_prob
        self.training = training

    def forward(self, x):
        if self.training:
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = self.keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            x = x.div(self.keep_prob) * random_tensor

        return x


class EncoderBlock(nn.Module):
    def __init__(
            self,
            dim,
            layer_scale_init_value,
            drop_rate,
            training=True,
            res=True
    ):
        super(EncoderBlock, self).__init__()
        self.res = res
        self.main = nn.Sequential(
            EqualizedConv2d(dim, dim, 7, 1, 3),
            nn.InstanceNorm2d(dim),
            EqualizedConv2d(dim, dim, 3, 1, 1),
            nn.GELU(),
            EqualizedConv2d(dim, dim, 3, 1, 1)
        )
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate, training=training) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        h = self.main(x)
        if self.gamma is not None:
            h = self.gamma * h
        h = self.drop_path(h)
        return (x + h, x) if self.res else x + h


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()

        self.main = nn.Sequential(
            nn.InstanceNorm2d(in_ch),
            EqualizedConv2d(in_ch, out_ch, 4, 2, 1)
        )
        self.skip = nn.Sequential(
            nn.AvgPool2d(3, 2, 1),
            EqualizedConv2d(in_ch, out_ch, 1, 1, 0)
        )

    def forward(self, x, res):
        return self.main(x) + self.skip(res)


class ConvNeXtBlock(nn.Module):
    def __init__(
            self,
            dim,
            drop_rate=0.,
            layer_scale_init_value=1e-6,
            scale=4,
            training=True,
    ):
        super(ConvNeXtBlock, self).__init__()
        self.main = nn.Sequential(
            self.groupconv(dim),
            nn.InstanceNorm2d(dim),
            EqualizedConv2d(dim, scale * dim, 1, 1, 0),
            nn.GELU(),
            EqualizedConv2d(scale * dim, dim, 1, 1, 0),
        )
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate, training=training) if drop_rate > 0. else nn.Identity()

    @staticmethod
    def groupconv(dim):
        conv = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        scale = 1 / math.sqrt(dim * 3 ** 2)
        torch.nn.init.normal_(conv.weight.data, std=scale)

        return conv

    def forward(self, x):
        h = self.main(x)
        if self.gamma is not None:
            h = self.gamma * h
        h = self.drop_path(h)

        return x + h


class SACatConvNextBlock(nn.Module):
    def __init__(
            self,
            dim,
            layer_scale_init_value,
            drop_rate,
            scale=4,
            training=True
    ):
        super(SACatConvNextBlock, self).__init__()

        self.main = ConvNeXtBlock(dim, layer_scale_init_value, drop_rate, scale, training)
        self.sa = nn.Sequential(
            EqualizedConv2d(dim * 2, dim, 1, 1, 0, bias=False),
            nn.ReLU(),
            EqualizedConv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, features):
        h = self.main(x)
        h = h * self.sa(torch.cat([h, features], dim=1))

        return h + x


class FeatureExtractor(nn.Module):
    def __init__(
            self,
            mid_ch=512,
            out_ch=1024,
            n_layers=2,
            scale=4,
    ):
        super(FeatureExtractor, self).__init__()

        self.main = nn.Sequential(
            ResNet34(),
            EqualizedConv2d(mid_ch, out_ch, 1, 1, 0),
            *[ConvNeXtBlock(out_ch, scale=scale) for _ in range(n_layers)]
        )

    def forward(self, x):
        x = self.main(x)

        return x


class Generator(nn.Module):
    """
    Generator Class
    U-net architecture, based on ConvNeXt blocks and Concatenation and Spatial Attention blocks.
    Parameters:
        in_ch: int, default: 7
          the dimensions of the input, RGB sketch and RGBA color hint.
        depth: tuple, default: (1, 1, 1, 1, 9, 9, 6, 3, 3)
          Number of ConvNeXt blocks in each layer, separated by down and up sample blocks
        dims: tuple, default: (64, 128, 256, 512, 1024)
          Dimensions of each layer.
        drop_path_rate: float, default: 0.2
          Maximum drop path rate, grows linearly as depth increases.
        layer_scale_init_value: float, default: 1e-5
          Beta value used to scale the input in the ConvNeXt blocks.
        scale: int, default: 2
          The dimensions scaled up in 1x1 convs in ConvNeXt blocks
        training: bool, default: True
          Activate drop path layers.
    """

    def __init__(
            self,
            in_ch=7,
            depth=(1, 1, 1, 1, 8, 6, 6, 3, 1),
            dims=(64, 128, 256, 512, 1024),
            attn=1,
            drop_path_rate=0.2,
            layer_scale_init_value=1e-2,
            scale=3,
            training=True,
    ):
        super(Generator, self).__init__()

        assert (len(depth) - 1) % 2 == 0
        n_down = (len(depth) - 1) // 2
        assert n_down == len(dims) - 1

        self.attn = attn
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.downsample = nn.ModuleList()

        # Feature Extractor
        self.feature = FeatureExtractor(out_ch=dims[-1], scale=scale).eval()

        # Encoder
        self.encoder = nn.ModuleList([EqualizedConv2d(in_ch, dims[0], 7, 1, 3)])
        dp_rate = [x.item() for x in torch.linspace(0, drop_path_rate, steps=sum(depth))]
        cur = 0
        for i in range(n_down):
            self.modules = []
            for j in range(depth[i]):
                self.modules.append(
                    EncoderBlock(dims[i], drop_rate=dp_rate[cur + j],
                                 layer_scale_init_value=layer_scale_init_value,
                                 training=training, res=True if j == depth[i] - 1 else False)
                )
            self.encoder.append(nn.Sequential(*self.modules))
            self.downsample.append(
                DownBlock(dims[i], dims[i + 1])
            )
            cur += depth[i]

        # Bottleneck
        self.bottleneck = nn.ModuleList()
        for j in range(depth[n_down]):
            if (j + 1) % attn != 0:
                self.bottleneck.append(
                    ConvNeXtBlock(dims[-1], drop_rate=dp_rate[cur + j],
                                  layer_scale_init_value=layer_scale_init_value,
                                  scale=scale, training=training)
                )
            else:
                self.bottleneck.append(
                    SACatConvNextBlock(dims[-1], drop_rate=dp_rate[cur + j],
                                       layer_scale_init_value=layer_scale_init_value,
                                       scale=scale, training=training)
                )
        self.bottleneck.append(EqualizedConv2d(dims[-1], dims[-2], 1, 1, 0))
        cur += depth[n_down]

        # Decoder
        self.decoder = nn.ModuleList()
        cur_dim = len(dims) - 2
        for i in range(n_down + 1, len(depth)):
            layer = [EqualizedConv2d(dims[cur_dim] * 2, dims[cur_dim], 1, 1, 0)]
            for j in range(depth[i]):
                layer.append(
                    ConvNeXtBlock(dims[cur_dim], drop_rate=dp_rate[cur + j],
                                  layer_scale_init_value=layer_scale_init_value,
                                  scale=scale, training=training)
                )
            if i != len(depth) - 1:
                layer.append(EqualizedConv2d(dims[cur_dim], dims[cur_dim - 1], 1, 1, 0))
            self.decoder.append(nn.Sequential(*layer))
            cur_dim -= 1
            cur += depth[i]

        self.out = nn.Sequential(
            EqualizedConv2d(dims[0], 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x, h):
        feature = self.feature(x)

        x = torch.cat([x, h], dim=1)
        mid_layer_list = []

        x = self.encoder[0](x)
        for i, layer in enumerate(self.encoder[1:]):
            x, res = layer(x)
            mid_layer_list.append(x)
            x = self.downsample[i](x, res)

        for i, layer in enumerate(self.bottleneck):
            if (i + 1) % self.attn == 0 and i != len(self.bottleneck) - 1:
                x = layer(x, feature)
            else:
                x = layer(x)

        for i, layer in enumerate(self.decoder):
            x = self.upsample(x)
            x = layer(torch.cat([x, mid_layer_list.pop()], dim=1))

        x = self.out(x)

        return x


# Discriminator V3

class Discriminator(nn.Module):
    """
    Discriminator Class
    Based on ConvNeXt, uses PatchGan and Multiscale strategy.
    Parameters:
        in_ch: int, default: 3
          the dimensions of the input, RGB image.
        depth: tuple, default: (1, 1, 3, 1)
          Number of ConvNeXt blocks in each layer, separated by down and up sample blocks
        dims: tuple, default: (96, 192, 384, 768)
          Dimensions of each layer.
        drop_path_rate: float, default: 0.2
          Maximum drop path rate, grows linearly as depth increases.
        layer_scale_init_value: float, default: 1e-5
          Beta value used to scale the input in the ConvNeXt blocks.
        num_D: int, default: 3
          Number of discriminators used.
        patch: bool, default: False
          Use PatchGAN strategy.
    """

    def __init__(
            self,
            in_ch=6,
            depth=(2, 2, 6, 2),
            dims=(96, 192, 384, 768),
            drop_path_rate=0.2,
            layer_scale_init_value=1e-5,
            scale=2,
            num_D=3,
            patch=True,
            im_size=512,
            training=True,
    ):
        super(Discriminator, self).__init__()

        n_layers = len(depth)
        assert im_size % (2 ** (n_layers + num_D + 1)) == 0

        self.patch = patch
        self.discs = nn.ModuleList()

        self.main = nn.ModuleList([EqualizedConv2d(in_ch, dims[0], 4, 4)])

        dp_rate = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        disc_blk = []

        for i in range(n_layers):
            for j in range(depth[i]):
                disc_blk.append(
                    ConvNeXtBlock(dims[i], drop_rate=dp_rate[cur + j],
                                  layer_scale_init_value=layer_scale_init_value,
                                  scale=scale,
                                  training=training)
                )
            if i != n_layers - 1:
                disc_blk.append(nn.InstanceNorm2d(dims[i]))
                disc_blk.append(EqualizedConv2d(dims[i], dims[i + 1], 4, 2, 1))

            cur += depth[i]

            self.main.append(nn.Sequential(*disc_blk))
            disc_blk = []

        self.out = nn.ModuleList()
        if self.patch:
            self.main.append(EqualizedConv2d(dims[-1], 1, 1, 1, 0, bias=False))
        else:
            for i in range(num_D):
                self.out.append(
                    EqualizedLinear(dims[-1] * (im_size // (2 ** (n_layers + 1 + i))) ** 2, 1)
                )

        for i in range(num_D):
            self.discs.append(self.main)

        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        results = []
        mid_points = []

        for i, disc in enumerate(self.discs):
            h = x
            for j, layer in enumerate(disc):
                x = layer(x)
                if (self.patch and j != len(disc) - 1) or not self.patch:
                    mid_points.append(x)

            if not self.patch:
                x = self.out[i](x.view(x.size(0), -1))

            results.append(x)
            x = self.downsample(h)

        return results, mid_points


# VGG19

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()

        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

        return out


class ResNet34(nn.Module):
    def __init__(self, require_grad=False):
        super(ResNet34, self).__init__()

        self.model = torch.hub.load('RF5/danbooru-pretrained', 'resnet34')
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        if not require_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        x = self.up(x)

        return x

