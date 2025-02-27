import torch.nn as nn
import torch

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size//2), bias=bias)


class FusionAtt(nn.Module):
    def __init__(self, in_channels, height=2, reduction=4, bias=False):
        super(FusionAtt, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)
        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU()))
        self.conv = nn.Conv2d(d * height, height, 1, 1, 0, bias=bias)

    def forward(self, inp_feats):
        fusion_at = [fc(inp_feats[idx]) for idx, fc in enumerate(self.fcs)]
        fusion_at = torch.cat(fusion_at, dim=1)  # N*(height*d)*H*W
        fusion_at = self.conv(fusion_at).softmax(dim=1).permute(1, 0, 2, 3).unsqueeze(-3)  # N*height*H*W

        return fusion_at

class ResidualBlock(nn.Module):
    def __init__(self, channel):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channel, channel, 3,padding=(3//2), bias=True),
            nn.InstanceNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3,padding=(3//2), bias=True),
            nn.InstanceNorm2d(channel),
        )

    def forward(self, x):
        return x + self.block(x)

class ResnetModule(nn.Module):
    def __init__(self, channel, num_residual_blocks=9):
        super(ResnetModule, self).__init__()

        model = []
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(channel)]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)


class CINRBlock(nn.Module):
    def __init__(self, inchanel):
        super(CINRBlock, self).__init__()
        model = [
            nn.Conv2d(3, inchanel, 7, stride=1, padding=3),
            nn.InstanceNorm2d(inchanel),
            nn.ReLU(inplace=True),
        ]
        for _ in range(2):
            inchanelt = inchanel*2
            model += [
                nn.Conv2d(inchanel, inchanelt, 3, stride=2, padding=1),
                nn.InstanceNorm2d(inchanelt),
                nn.ReLU(inplace=True),
            ]
            inchanel = inchanelt
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

class DINRBlock(nn.Module):
    def __init__(self, inchanel):
        super(DINRBlock, self).__init__()
        model = []
        for _ in range(2):
            inchanelt = inchanel // 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(inchanel, inchanelt, 3, stride=1, padding=1),
                nn.InstanceNorm2d(inchanelt),
                nn.ReLU(inplace=True),
                ]
            inchanel = inchanelt
        model += [
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.InstanceNorm2d(inchanel),
            nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

class FusionBlock(nn.Module):
    def __init__(self):
        super(FusionBlock, self).__init__()
        model = [
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)


class YGmodel(nn.Module):
    def __init__(self):
        super(YGmodel, self).__init__()
        self.res = ResnetModule(256)
        self.down = CINRBlock(64)
        self.up = DINRBlock(256)
        self.initconv = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.bla = nn.Sequential(
            nn.Conv2d(256, 128, 1, padding=0, bias=True),
            nn.PReLU(),
            nn.Conv2d(128 , 64, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.tran = nn.Sequential(
            nn.Conv2d(64, 64 // 8, 3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64 // 8, 64, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

        self.af = FusionAtt(64,height=2)
        self.fb = FusionBlock()
    def forward(self, x):

        downx = self.down(x)
        resx = self.res(downx)
        out1 = self.up(resx)

        px = self.initconv(x)
        a = self.avg_pool(downx)
        a = self.bla(a)
        t = self.tran(px)
        out2 = torch.mul((1 - t), a) + torch.mul(t, px)

        v = self.af([out1, out2])
        outf = out1 * v[0] + out2 * v[1]
        out = self.fb(outf)

        return out
