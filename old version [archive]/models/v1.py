import torch
from torch import nn
from torchvision import models


class FeatureExtractor(nn.Module):
    def __init__(self, mid_channels=512, out_channels=512):
        super(FeatureExtractor, self).__init__()

        self.main = nn.Sequential(
            ResNet34(),
            ConvBaseBlock(mid_channels, out_channels, 1, 1, 0, bias=False),
            ResBlock(out_channels, out_channels),
            ResBlock(out_channels, out_channels)
        )

    def forward(self, x):
        x = self.main(x)

        return x


class ConvBaseBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 bias=True):
        super(ConvBaseBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        x = self.main(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResBlock, self).__init__()

        self.main = nn.Sequential(
            ConvBaseBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ConvBaseBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        res = self.main(x)

        return res + x


class DownResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, down=True):
        super(DownResBlock, self).__init__()

        self.main = nn.Sequential(
            ConvBaseBlock(in_channels, out_channels, 4, 2, 1),
            ConvBaseBlock(out_channels, out_channels, 3, 1, 1)
        )
        self.downsample = nn.AvgPool2d(3, 2, 1) if down else nn.Identity()
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        res = self.main(x)
        x = self.downsample(x)
        x = self.skip(x)

        return res + x


class SEBlock(nn.Module):
    def __init__(self, in_channels: int):
        super(SEBlock, self).__init__()

        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, int(in_channels/16), 1, 1, 0, bias=False),
            nn.ReLU(),
            nn.Conv2d(int(in_channels/16), in_channels, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.main(x)


class SEResNextBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 cardinality: int,
                 norm='in',
                 activation='lrelu'):
        super(SEResNextBlock, self).__init__()

        self.modules = []
        self.modules.append(
            ConvBaseBlock(in_channels, out_channels, 1, 1, 0, bias=False),
        )
        self.modules.append(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=cardinality)
        )

        self.modules = self._norm(self.modules, norm, out_channels)
        self.modules = self._activation(self.modules, activation)

        self.modules.append(
            SEBlock(out_channels)
        )

        self.main = nn.Sequential(*self.modules)

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False) \
            if in_channels != out_channels else nn.Identity()

        self.relu = nn.LeakyReLU(0.2, True)

    @staticmethod
    def _norm(modules, norm, out_channels):
        if norm == 'in':
            modules.append(nn.InstanceNorm2d(out_channels))
        elif norm == 'bn':
            modules.append(nn.BatchNorm2d(out_channels))

        return modules

    @staticmethod
    def _activation(modules, activation):
        if activation == 'lrelu':
            modules.append(nn.LeakyReLU(0.2, True))
        elif activation == 'relu':
            modules.append(nn.ReLU())

        return modules

    def forward(self, x):
        skip = self.shortcut(x)
        x = self.main(x)
        x = self.relu(x)

        return x + skip


class UpBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 cardinality: int,
                 n_layers: int,
                 ):
        super(UpBlock, self).__init__()

        if n_layers != 1:
            modules = [SEResNextBlock(in_channels, out_channels, cardinality=cardinality)]
            for _ in range(n_layers-1):
                modules.append(SEResNextBlock(out_channels, out_channels, cardinality=cardinality))
            modules.append(nn.UpsamplingBilinear2d(scale_factor=2))

            self.main = nn.Sequential(*modules)

        else:
            self.main = nn.Sequential(
                SEResNextBlock(in_channels, out_channels, cardinality=cardinality),
                nn.UpsamplingBilinear2d(scale_factor=2)
            )

    def forward(self, x):
        x = self.main(x)

        return x


class SACat(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(SACat, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, 1, 1, 0, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, feature):
        h = torch.cat([x, feature], dim=1)
        h = self.main(h)

        return h


class SACatResNextBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(SACatResNextBlock, self).__init__()

        self.se = SEResNextBlock(in_channels, out_channels, cardinality=32)
        self.sa = SACat(out_channels, out_channels)

    def forward(self, x, feature):
        h = self.se(x)
        h = h * self.sa(h, feature)

        return h + x


class Generator(nn.Module):
    def __init__(self,
                 in_channels=5,
                 feature=64,
                 num_layers=10,
                 up_layers=[10, 5, 5, 3]):

        super(Generator, self).__init__()

        self.feature = FeatureExtractor(out_channels=feature * 8)

        # Content Encoder
        self.c1 = ConvBaseBlock(in_channels, feature, 3, 1, 1)

        self.d1 = DownResBlock(feature, feature * 2)
        self.d2 = DownResBlock(feature * 2, feature * 4)
        self.d3 = DownResBlock(feature * 4, feature * 8)
        self.d4 = DownResBlock(feature * 8, feature * 8)

        self.d_res = nn.Sequential(
            ResBlock(feature*8, feature*8),
            ResBlock(feature*8, feature*8),
        )

        # Feature Extractor
        self.feature = FeatureExtractor()

        # Bottleneck
        modules = [SACatResNextBlock(feature*8, feature*8) for _ in range(num_layers)]
        self.bottleneck = nn.ModuleList(modules)

        # Decoder
        self.u1 = UpBlock(feature * 16, feature * 8, cardinality=16, n_layers=up_layers[0])
        self.u2 = UpBlock(feature * 16, feature * 4, cardinality=16, n_layers=up_layers[1])
        self.u3 = UpBlock(feature * 8, feature * 2, cardinality=16, n_layers=up_layers[2])
        self.u4 = UpBlock(feature * 4, feature, cardinality=16, n_layers=up_layers[3])

        self.out = nn.Sequential(
            nn.Conv2d(feature, feature, 3, 1, 1),
            nn.InstanceNorm2d(feature),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x, h, style):
        x = torch.cat([x, h], dim=1)

        x = self.c1(x)
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        x = self.d_res(d4)

        feature = self.feature(style)
        for layer in self.bottleneck:
            x = layer(x, feature)

        x = self.u1(torch.cat([x, d4], dim=1))
        x = self.u2(torch.cat([x, d3], dim=1))
        x = self.u3(torch.cat([x, d2], dim=1))
        x = self.u4(torch.cat([x, d1], dim=1))

        x = self.out(x)

        return x


# Discriminator

class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=2):
        super(CNNBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class NLDiscriminator(nn.Module):
    def __init__(self, in_channels=4, features=[64, 128, 256, 512]):  # 512 -> 58
        super(NLDiscriminator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )

        modules = []
        prev_channels = features[0]
        for feature in features[1:]:
            modules.append(
                CNNBlock(prev_channels, feature, stride=1 if feature == features[-1] else 2),
            )
            prev_channels = feature

        modules.append(
            nn.Conv2d(prev_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'),
        )

        self.main = nn.Sequential(*modules)

    def forward(self, x):
        x = self.initial(x)
        return self.main(x)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, num_D=3, in_channels=4, features=[64, 128, 256, 512]):
        super(MultiscaleDiscriminator, self).__init__()

        self.num_D = num_D
        self.discs = nn.ModuleList()

        for i in range(num_D):
            netD = NLDiscriminator(in_channels, features)
            self.discs.append(netD)

        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x, y):
        num_D = self.num_D
        result = []
        x_down = torch.cat([x, y], dim=1)  # 1 + 3

        for i in range(num_D):
            model = self.discs[i]
            result.append(model(x_down))
            if i != num_D - 1:
                x_down = self.downsample(x_down)

        return result  # [loss1, loss2, loss3]


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