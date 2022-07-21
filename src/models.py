import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, active=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if active:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, active=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, active=False)
        )
        
    def forward(self, x):
        return self.main(x) + x
    

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(out_channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, num_res = 8):
        super(UNet, self).__init__()
        base_channel = 32
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, active=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, active=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, active=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, active=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, active=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, active=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, active=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, active=True, stride=1),
        ])

    def forward(self, x):

        z = self.feat_extract[0](x)
        res1 = self.Encoder[0](z)

        z = self.feat_extract[1](res1)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.Encoder[2](z)

        z = self.Decoder[0](z)
        z = self.feat_extract[3](z)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z = self.feat_extract[4](z)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)

        return z+x
    
