import torch
from torch import nn
from torchvision import models


class FeatureExtractor(nn.Module):
    def __init__(self, mid_ch=512, out_ch=1024, n_layers=3):
        super(FeatureExtractor, self).__init__()

        self.main = nn.Sequential(
            ResNet34(),
            nn.Conv2d(mid_ch, out_ch, 1, 1, 0),
            *[ConvNeXtBlock(out_ch) for _ in range(n_layers)]
        )

    def forward(self, x):
        x = self.main(x)

        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob, training=True):
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


class ConvNeXtBlock(nn.Module):
    def __init__(self,
                 dim,
                 drop_rate=0.,
                 layer_scale_init_value=1e-6,
                 scale=4,
                 training=True,
                 ):
        super(ConvNeXtBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(dim, dim, 7, 1, 3, groups=dim),
            nn.InstanceNorm2d(dim),
            nn.Conv2d(dim, scale * dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(scale * dim, dim, 1, 1, 0),
        )
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate, training=training) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        h = self.main(x)
        if self.gamma is not None:
            h = self.gamma * h
        h = self.drop_path(h)

        return x + h


class SACatConvNextBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value, drop_rate, training):
        super(SACatConvNextBlock, self).__init__()

        self.main = ConvNeXtBlock(dim, layer_scale_init_value, drop_rate, training=training)
        self.sa = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, 1, 0, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, features):
        h = self.main(x)
        h = h * self.sa(torch.cat([h, features], dim=1))

        return h + x


class Generator(nn.Module):
    def __init__(self,
                 in_channels=7,
                 depth=[1, 1, 1, 1, 27, 9, 9, 3, 3, 3],
                 n_down=4,
                 dims=[64, 128, 256, 512, 1024],
                 attn=3,
                 drop_path_rate=0.2,
                 layer_scale_init_value=1e-5,
                 training=True,
                 ):
        super(Generator, self).__init__()

        assert n_down * 2 + 2 == len(depth)
        assert depth[n_down] % attn == 0

        self.attn = attn
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.downsample = nn.ModuleList()

        # Feature Extractor
        self.feature = FeatureExtractor(out_ch=dims[-1])

        # Encoder
        self.encoder = nn.ModuleList([nn.Conv2d(in_channels, dims[0], 3, 1, 1)])
        dp_rate = [x.item() for x in torch.linspace(0, drop_path_rate, steps=sum(depth))]
        cur = 0
        for i in range(n_down):
            for j in range(depth[i]):
                self.encoder.append(
                    ConvNeXtBlock(dims[i], drop_rate=dp_rate[cur + j],
                                  layer_scale_init_value=layer_scale_init_value,
                                  training=training)
                )
            self.downsample.append(
                nn.Sequential(
                    nn.InstanceNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], 2, 2, 0)
                )
            )
            cur += depth[i]

        # Bottleneck
        self.bottleneck = nn.ModuleList()
        for j in range(depth[n_down]):
            if (j + 1) % attn != 0:
                self.bottleneck.append(
                    ConvNeXtBlock(dims[-1], drop_rate=dp_rate[cur + j],
                                  layer_scale_init_value=layer_scale_init_value,
                                  training=training)
                )
            else:
                self.bottleneck.append(
                    SACatConvNextBlock(dims[-1], drop_rate=dp_rate[cur + j],
                                       layer_scale_init_value=layer_scale_init_value,
                                       training=training)
                )
        cur += depth[n_down]

        # Decoder
        self.decoder = nn.ModuleList()
        cur_dim = -1
        for i in range(n_down + 1, len(depth)):
            if i == n_down + 1:
                layer = [nn.Conv2d(dims[cur_dim], dims[cur_dim - 1], 1, 1, 0)]
            else:
                layer = [nn.Conv2d(dims[cur_dim] * 2, dims[cur_dim - 1], 1, 1, 0)]
            cur_dim -= 1
            for j in range(depth[i]):
                layer.append(
                    ConvNeXtBlock(dims[cur_dim], drop_rate=dp_rate[cur + j],
                                  layer_scale_init_value=layer_scale_init_value,
                                  training=training)
                )
            self.decoder.append(nn.Sequential(*layer))
            cur += depth[i]

        self.out = nn.Sequential(
            *[ConvNeXtBlock(dims[0], drop_rate=dp_rate[cur+j],
                            layer_scale_init_value=layer_scale_init_value,
                            training=training) for j in range(3)],
            nn.Tanh()
        )

    def forward(self, x, h):
        style = x
        x = torch.cat([x, h], dim=1)
        mid_layer_list = []

        x = self.encoder[0](x)
        for i, layer in enumerate(self.encoder[1:]):
            x = layer(x)
            mid_layer_list.append(x)
            x = self.downsample[i](x)

        feature = self.feature(style)
        for i, layer in enumerate(self.bottleneck):
            if (i + 1) % self.attn == 0:
                x = layer(x, feature)
            else:
                x = layer(x)

        for i, layer in enumerate(self.decoder):
            if i == 0:
                x = layer(x)
            else:
                x = layer(torch.cat([x, mid_layer_list.pop()], dim=1))
            x = self.upsample(x)

        x = self.out(x)

        return x


# Discriminator V3

class NLDiscriminator(nn.Module):
    def __init__(self, in_ch, depth, dims, drop_path_rate, layer_scale_init_value):
        super(NLDiscriminator, self).__init__()

        n_layers = len(depth)

        self.modules = [
            nn.Conv2d(in_ch, dims[0], kernel_size=4, stride=4),
        ]

        dp_rate = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0

        for i in range(n_layers):
            for j in range(depth[i]):
                self.modules.append(
                    ConvNeXtBlock(dims[i], drop_rate=dp_rate[cur + j],
                                  layer_scale_init_value=layer_scale_init_value)
                )
            if i != n_layers-1:
                self.modules.append(nn.InstanceNorm2d(dims[i]))
                self.modules.append(nn.Conv2d(dims[i], dims[i+1], 2, 2))
            cur += depth[i]

        self.main = nn.Sequential(*self.modules)

    def forward(self, x):
        return self.main(x).mean([-2, -1])


class MultiscaleDiscriminator(nn.Module):
    def __init__(self,
                 num_D=3,
                 in_ch=3,
                 depth=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0.2,
                 layer_scale_init_value=1e-5,
                 ):
        super(MultiscaleDiscriminator, self).__init__()

        self.num_D = num_D
        self.discs = nn.ModuleList()

        for i in range(num_D):
            self.discs.append(NLDiscriminator(in_ch, depth, dims, drop_path_rate, layer_scale_init_value))

        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        results = []
        for i in range(self.num_D):
            model = self.discs[i]
            results.append(model(x))
            if i != self.num_D - 1:
                x = self.downsample(x)

        return results  # [loss1, loss2, loss3]


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

    def forward(self, X):
        h_relu1 = self.slice1(X)
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


if __name__ == '__main__':
    model = Generator()
    data = torch.ones((1, 3, 512, 512))
    data2 = torch.ones((1, 4, 512, 512))
    print(model(data, data2))