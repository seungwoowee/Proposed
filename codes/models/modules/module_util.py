import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from utils import util


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


## STARnet module baseed

class DenseBlock(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation='relu', norm='batch'):
        super(DenseBlock, self).__init__()
        self.fc = nn.Linear(input_size, output_size, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm1d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm1d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.1, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.fc(x))
        else:
            out = self.fc(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.1, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ConvBlock3D(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(input_size, output_size, kernel_size, stride, padding)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.1, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DeconvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.1, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ResnetBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(num_filter)
        elif norm == 'instance':
            self.bn = nn.InstanceNorm2d(num_filter)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.1, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)

        out = torch.add(out, residual)

        if self.activation is not None:
            out = self.act(out)

        return out


class ResnetBlock3D(nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(ResnetBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(num_filter, num_filter, kernel_size, stride, padding)
        self.conv2 = nn.Conv3d(num_filter, num_filter, kernel_size, stride, padding)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(num_filter)
        elif norm == 'instance':
            self.bn = nn.InstanceNorm2d(num_filter)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.1, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)

        out = torch.add(out, residual)

        if self.activation is not None:
            out = self.act(out)

        return out


class UpBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class UpBlockPix(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, scale=4, bias=True, activation='prelu',
                 norm=None):
        super(UpBlockPix, self).__init__()
        self.up_conv1 = Upsampler_STAR(scale, num_filter)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.up_conv3 = Upsampler_STAR(scale, num_filter)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class D_UpBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu',
                 norm=None):
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class D_UpBlockPix(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, scale=4, bias=True,
                 activation='prelu', norm=None):
        super(D_UpBlockPix, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = Upsampler_STAR(scale, num_filter)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.up_conv3 = Upsampler_STAR(scale, num_filter)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class DownBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class DownBlockPix(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, scale=4, bias=True, activation='prelu',
                 norm=None):
        super(DownBlockPix, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.down_conv2 = Upsampler_STAR(scale, num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class D_DownBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu',
                 norm=None):
        super(D_DownBlock, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, bias, activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class D_DownBlockPix(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, scale=4, bias=True,
                 activation='prelu', norm=None):
        super(D_DownBlockPix, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, bias, activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.down_conv2 = Upsampler_STAR(scale, num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class PSBlock(nn.Module):
    def __init__(self, input_size, output_size, scale_factor, kernel_size=3, stride=1, padding=1, bias=True,
                 activation='prelu', norm='batch'):
        super(PSBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size * scale_factor ** 2, kernel_size, stride, padding, bias=bias)
        self.ps = nn.PixelShuffle(scale_factor)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.1, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.ps(self.conv(x)))
        else:
            out = self.ps(self.conv(x))

        if self.activation is not None:
            out = self.act(out)
        return out


class Upsampler_STAR(nn.Module):
    def __init__(self, scale, n_feat, bn=False, act='prelu', bias=True):
        super(Upsampler_STAR, self).__init__()
        modules = []
        for _ in range(int(math.log(scale, 2))):
            modules.append(ConvBlock(n_feat, 4 * n_feat, 3, 1, 1, bias, activation=None, norm=None))
            modules.append(nn.PixelShuffle(2))
            if bn: modules.append(nn.BatchNorm2d(n_feat))
            # modules.append(nn.PReLU())
        self.up = nn.Sequential(*modules)

        self.activation = act
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.1, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        out = self.up(x)
        if self.activation is not None:
            out = self.act(out)
        return out


class Upsample2xBlock(nn.Module):
    def __init__(self, input_size, output_size, bias=True, upsample='deconv', activation='relu', norm='batch'):
        super(Upsample2xBlock, self).__init__()
        scale_factor = 2
        # 1. Deconvolution (Transposed convolution)
        if upsample == 'deconv':
            self.upsample = DeconvBlock(input_size, output_size,
                                        kernel_size=4, stride=2, padding=1,
                                        bias=bias, activation=activation, norm=norm)

        # 2. Sub-pixel convolution (Pixel shuffler)
        elif upsample == 'ps':
            self.upsample = PSBlock(input_size, output_size, scale_factor=scale_factor,
                                    bias=bias, activation=activation, norm=norm)

        # 3. Resize and Convolution
        elif upsample == 'rnc':
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                ConvBlock(input_size, output_size,
                          kernel_size=3, stride=1, padding=1,
                          bias=bias, activation=activation, norm=norm)
            )

    def forward(self, x):
        out = self.upsample(x)
        return out


class PyramidModule(nn.Module):
    def __init__(self, num_inchannels, activation='prelu'):
        super(PyramidModule, self).__init__()

        self.l1_1 = ResnetBlock(num_inchannels, kernel_size=3, stride=1, padding=1, bias=True, activation=activation,
                                norm=None)
        self.l1_2 = ResnetBlock(num_inchannels, kernel_size=3, stride=1, padding=1, bias=True, activation=activation,
                                norm=None)
        self.l1_3 = ResnetBlock(num_inchannels, kernel_size=3, stride=1, padding=1, bias=True, activation=activation,
                                norm=None)
        self.l1_4 = ResnetBlock(num_inchannels, kernel_size=3, stride=1, padding=1, bias=True, activation=activation,
                                norm=None)
        self.l1_5 = ResnetBlock(num_inchannels, kernel_size=3, stride=1, padding=1, bias=True, activation=activation,
                                norm=None)

        self.l2_1 = ResnetBlock(num_inchannels * 2, kernel_size=3, stride=1, padding=1, bias=True,
                                activation=activation, norm=None)
        self.l2_2 = ResnetBlock(num_inchannels * 2, kernel_size=3, stride=1, padding=1, bias=True,
                                activation=activation, norm=None)
        self.l2_3 = ResnetBlock(num_inchannels * 2, kernel_size=3, stride=1, padding=1, bias=True,
                                activation=activation, norm=None)
        self.l2_4 = ResnetBlock(num_inchannels * 2, kernel_size=3, stride=1, padding=1, bias=True,
                                activation=activation, norm=None)

        self.l3_1 = ResnetBlock(num_inchannels * 4, kernel_size=3, stride=1, padding=1, bias=True,
                                activation=activation, norm=None)
        self.l3_2 = ResnetBlock(num_inchannels * 4, kernel_size=3, stride=1, padding=1, bias=True,
                                activation=activation, norm=None)
        self.l3_3 = ResnetBlock(num_inchannels * 4, kernel_size=3, stride=1, padding=1, bias=True,
                                activation=activation, norm=None)

        self.down1 = ConvBlock(num_inchannels, num_inchannels * 2, 4, 2, 1, bias=True, activation=activation, norm=None)
        self.down2 = ConvBlock(num_inchannels * 2, num_inchannels * 4, 4, 2, 1, bias=True, activation=activation,
                               norm=None)

        self.up1 = DeconvBlock(num_inchannels * 2, num_inchannels, 4, 2, 1, bias=True, activation=activation, norm=None)
        self.up2 = DeconvBlock(num_inchannels * 4, num_inchannels * 2, 4, 2, 1, bias=True, activation=activation,
                               norm=None)

        self.final = ConvBlock(num_inchannels, num_inchannels, 3, 1, 1, bias=True, activation=activation, norm=None)

    def forward(self, x):
        out1_1 = self.l1_1(x)
        out2_1 = self.l2_1(self.down1(out1_1))
        out3_1 = self.l3_1(self.down2(out2_1))

        out1_2 = self.l1_2(out1_1 + self.up1(out2_1))
        out2_2 = self.l2_2(out2_1 + self.down1(out1_2) + self.up2(out3_1))
        out3_2 = self.l3_2(out3_1 + self.down2(out2_2))

        out1_3 = self.l1_3(out1_2 + self.up1(out2_2))
        out2_3 = self.l2_3(out2_2 + self.down1(out1_3) + self.up2(out3_2))
        out3_3 = self.l3_3(out3_2 + self.down2(out2_3))

        out1_4 = self.l1_4(out1_3 + self.up1(out2_3))
        out2_4 = self.l2_4(out2_3 + self.down1(out1_4) + self.up2(out3_3))

        out1_5 = self.l1_5(out1_4 + self.up1(out2_4))

        final = self.final(out1_5)

        return final


### DRB_Net module


class PixelShufflePack(nn.Module):
    """ Pixel Shuffle upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, scale):
        super(PixelShufflePack, self).__init__()
        self.upsample_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                       kernel_size=3, padding=1)
        self.relu = nn.PReLU()
        self.scale = scale

        util.initialize_weights([self.upsample_conv])

    def forward(self, x):
        """Forward function for PixelShufflePack.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale)
        x = self.relu(x)

        return x


class CALayer(nn.Module):
    def __init__(self, inout_ch, ch_reduction_ratio):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(inout_ch, inout_ch // ch_reduction_ratio, 1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(inout_ch // ch_reduction_ratio, inout_ch, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        util.initialize_weights([self.conv_du])

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ConvModule(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(out_ch)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(out_ch)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

        util.initialize_weights([self.conv])

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


# class DownUpModule(torch.nn.Module): # up 부분 고치기
#     def __init__(self, inout_ch, kernel_size=3, stride=1, padding=1, bias=False, activation='prelu', norm=None,
#                  block_n=1):
#         super(DownUpModule, self).__init__()
#         # conv first
#         self.conv_first = ConvModule(in_ch=inout_ch * block_n, out_ch=inout_ch, kernel_size=1, stride=1,
#                                      padding=0, bias=bias, activation=activation, norm=norm)
#         # down & up & down
#         self.conv1_1 = ConvModule(in_ch=inout_ch, out_ch=inout_ch, kernel_size=kernel_size, stride=stride,
#                                   padding=padding, bias=bias, activation=activation, norm=norm)
#         self.down1_2 = nn.MaxPool2d(2)
#         self.up1_3 = PixelShufflePack(in_channels=inout_ch, scale=2)
#         self.conv1_4 = ConvModule(in_ch=inout_ch, out_ch=inout_ch, kernel_size=kernel_size, stride=stride,
#                                   padding=padding, bias=bias, activation=activation, norm=norm)
#         self.down1_5 = nn.MaxPool2d(2)
#
#         # up & down & up
#         self.up2_1 = PixelShufflePack(in_channels=inout_ch, scale=2)
#         self.conv2_2 = ConvModule(in_ch=inout_ch, out_ch=inout_ch, kernel_size=kernel_size, stride=stride,
#                                   padding=padding, bias=bias, activation=activation, norm=norm)
#         self.down2_3 = nn.MaxPool2d(2)
#         self.up2_4 = PixelShufflePack(in_channels=inout_ch, scale=2)
#
#     def forward(self, x):
#         # down & up & down
#         res = self.down1_2(self.conv1_1(x))
#         out = self.up1_3(res)
#         sub = self.down1_5(self.conv1_4(out - x)) + res
#
#         # up & down & up
#         res = self.up2_1(sub)
#         out = self.down2_3(self.conv2_2(res))
#         out = self.up2_4(out - sub)
#
#         return out + res


class DRB_Block1(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, activation='prelu',
                 use_norm=None):
        super(DRB_Block1, self).__init__()
        self.conv1 = ConvModule(in_ch=in_ch, out_ch=out_ch, kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=bias, activation=activation, norm=use_norm)
        self.conv2 = ConvModule(in_ch=out_ch, out_ch=out_ch, kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=bias, activation=activation, norm=use_norm)
        self.conv3 = ConvModule(in_ch=out_ch, out_ch=out_ch, kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=bias, activation=activation, norm=use_norm)
        self.conv4 = ConvModule(in_ch=out_ch, out_ch=out_ch, kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=bias, activation=activation, norm=use_norm)

    def forward(self, x):
        out1 = self.conv1(x)
        res = self.conv2(out1)
        out2 = self.conv3(res)
        out = self.conv4(out2 - out1)
        return out + res


class DRB_Block2(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, activation='prelu',
                 use_norm=None):
        super(DRB_Block2, self).__init__()
        # down & up & down
        self.conv1 = ConvModule(in_ch=in_ch, out_ch=4 * out_ch, kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=bias, activation=activation, norm=use_norm)
        self.down1 = nn.MaxPool2d(2)

        self.up1 = PixelShufflePack(in_channels=4 * out_ch, scale=2)
        self.conv2 = ConvModule(in_ch=in_ch + out_ch, out_ch=out_ch, kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=bias, activation=activation, norm=use_norm)

    def forward(self, x):
        # down & down & up
        out = self.down1(self.conv1(x))
        out = self.up1(out)
        out = self.conv2(torch.cat((x, out), 1))
        return out


class DRB_Block3(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, activation='prelu',
                 use_norm=None):
        super(DRB_Block3, self).__init__()
        # down & up & down
        self.conv1 = ConvModule(in_ch=in_ch, out_ch=4 * out_ch, kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=bias, activation=activation, norm=use_norm)
        self.down1 = nn.MaxPool2d(2)

        # self.up1 = DeconvBlock(input_size=4 * out_ch, output_size=out_ch, kernel_size=kernel_size,
        #                        stride=2, padding=1, bias=bias, activation=activation, norm=use_norm)
        self.up1 = nn.ConvTranspose2d(4 * out_ch, out_ch, 3, stride=2, padding=1, bias=bias)

        self.conv2 = ConvModule(in_ch=in_ch + out_ch, out_ch=out_ch, kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=bias, activation=activation, norm=use_norm)

    def forward(self, x):
        # down & down & up
        out = self.down1(self.conv1(x))

        out = self.up1(out, output_size=x.size())
        out = self.conv2(torch.cat((x, out), 1))
        return out


class DRB_Block4(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, activation='prelu',
                 use_norm=None):
        super(DRB_Block4, self).__init__()
        self.conv1 = ConvModule(in_ch=in_ch, out_ch=in_ch, kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=bias, activation=activation, norm=use_norm)
        self.down1 = nn.MaxPool2d(2)

        # self.up1 = DeconvBlock(input_size=4 * out_ch, output_size=out_ch, kernel_size=kernel_size,
        #                        stride=2, padding=1, bias=bias, activation=activation, norm=use_norm)
        self.up1 = nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, bias=bias)

        self.conv2 = ConvModule(in_ch=in_ch + out_ch, out_ch=out_ch, kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=bias, activation=activation, norm=use_norm)

    def forward(self, x):
        out = self.down1(self.conv1(x))
        out = self.up1(out, output_size=x.size())
        out = self.conv2(torch.cat((x, out), 1))
        return out


class DRBModule(nn.Module):
    def __init__(self, inout_ch, ch_reduction_ratio, bias, act, block_n):
        super(DRBModule, self).__init__()
        self.conv1_0 = DRB_Block1(in_ch=inout_ch * block_n, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                  bias=bias, activation=act, use_norm=None)
        self.conv1_1 = DRB_Block1(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_2 = DRB_Block1(in_ch=inout_ch * block_n + 2 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_3 = DRB_Block1(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_4 = DRB_Block1(in_ch=inout_ch * block_n, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                  bias=bias, activation=act, use_norm=None)

        self.conv2_1 = DRB_Block1(in_ch=2 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1, bias=bias,
                                  activation=act, use_norm=None)
        self.conv2_2 = DRB_Block1(in_ch=5 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1, bias=bias,
                                  activation=act, use_norm=None)
        self.conv2_3 = DRB_Block1(in_ch=2 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1, bias=bias,
                                  activation=act, use_norm=None)

        self.conv3_2 = DRB_Block1(in_ch=3 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1, bias=bias,
                                  activation=act, use_norm=None)

        self.CA = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)

        self.conv_out = nn.Conv2d(in_channels=inout_ch + inout_ch * block_n, out_channels=inout_ch, kernel_size=3,
                                  stride=1, padding=1, bias=False)

    def forward(self, x):
        x1_0 = self.conv1_0(x[0])
        x1_1 = self.conv1_1(torch.cat((x1_0, x[1]), 1))
        x1_4 = self.conv1_4(x[4])
        x1_3 = self.conv1_3(torch.cat((x1_4, x[3]), 1))
        x1_2 = self.conv1_2(torch.cat((x1_1, x1_3, x[2]), 1))

        x2_1 = self.conv2_1(torch.cat((x1_0, x1_1), 1))
        x2_3 = self.conv2_3(torch.cat((x1_4, x1_3), 1))
        x2_2 = self.conv2_2(torch.cat((x1_1, x1_2, x1_3, x2_1, x2_3), 1))

        x3_2 = self.conv3_2(torch.cat((x2_1, x2_2, x2_3), 1))

        out_0 = self.CA(self.conv_out(torch.cat((x[0], x1_0), 1)))
        out_1 = self.CA(self.conv_out(torch.cat((x[1], x2_1), 1)))
        out_2 = self.CA(self.conv_out(torch.cat((x[2], x3_2), 1)))
        out_3 = self.CA(self.conv_out(torch.cat((x[3], x2_3), 1)))
        out_4 = self.CA(self.conv_out(torch.cat((x[4], x1_4), 1)))

        x[0] = torch.cat((out_0, x[0]), 1)
        x[1] = torch.cat((out_1, x[1]), 1)
        x[2] = torch.cat((out_2, x[2]), 1)
        x[3] = torch.cat((out_3, x[3]), 1)
        x[4] = torch.cat((out_4, x[4]), 1)

        return x


class DRBModule2(nn.Module):
    def __init__(self, inout_ch, ch_reduction_ratio, bias, act, block_n):
        super(DRBModule2, self).__init__()
        self.inout_ch = inout_ch
        self.conv1_0 = DRB_Block2(in_ch=inout_ch * block_n, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                  bias=bias, activation=act, use_norm=None)
        self.conv1_1 = DRB_Block2(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_2 = DRB_Block2(in_ch=inout_ch * block_n + 2 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_3 = DRB_Block2(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_4 = DRB_Block2(in_ch=inout_ch * block_n, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                  bias=bias, activation=act, use_norm=None)

        self.conv2_2 = DRB_Block2(in_ch=5 * inout_ch, out_ch=5 * inout_ch, kernel_size=3, stride=1, padding=1,
                                  bias=bias, activation=act, use_norm=None)

        self.conv_out0 = DRB_Block2(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out1 = DRB_Block2(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out2 = DRB_Block2(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out3 = DRB_Block2(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out4 = DRB_Block2(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.CA0 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA1 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA2 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA3 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA4 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)

    def forward(self, x):
        x1_0 = self.conv1_0(x[0])
        x1_1 = self.conv1_1(torch.cat((x1_0, x[1]), 1))
        x1_4 = self.conv1_4(x[4])
        x1_3 = self.conv1_3(torch.cat((x1_4, x[3]), 1))
        x1_2 = self.conv1_2(torch.cat((x1_1, x1_3, x[2]), 1))

        x2_2 = self.conv2_2(torch.cat((x1_0, x1_1, x1_2, x1_3, x1_4), 1))
        tmp = torch.split(x2_2[0], self.inout_ch)
        x_out = []
        for k in range(len(tmp)):
            x_out.append(tmp[k].unsqueeze(0))

        out_0 = self.CA0(self.conv_out0(torch.cat((x[0], x_out[0]), 1)))
        out_1 = self.CA1(self.conv_out1(torch.cat((x[1], x_out[1]), 1)))
        out_2 = self.CA2(self.conv_out2(torch.cat((x[2], x_out[2]), 1)))
        out_3 = self.CA3(self.conv_out3(torch.cat((x[3], x_out[3]), 1)))
        out_4 = self.CA4(self.conv_out4(torch.cat((x[4], x_out[4]), 1)))

        x[0] = torch.cat((out_0, x[0]), 1)
        x[1] = torch.cat((out_1, x[1]), 1)
        x[2] = torch.cat((out_2, x[2]), 1)
        x[3] = torch.cat((out_3, x[3]), 1)
        x[4] = torch.cat((out_4, x[4]), 1)

        # x.append(out_2)

        return x


class DRBModule4(nn.Module):
    def __init__(self, inout_ch, ch_reduction_ratio, bias, act, block_n):
        super(DRBModule4, self).__init__()
        self.inout_ch = inout_ch
        self.conv1_0 = DRB_Block2(in_ch=inout_ch * block_n, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                  bias=bias, activation=act, use_norm=None)
        self.conv1_1 = DRB_Block2(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_2 = DRB_Block2(in_ch=inout_ch * block_n + 2 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_3 = DRB_Block2(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_4 = DRB_Block2(in_ch=inout_ch * block_n, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                  bias=bias, activation=act, use_norm=None)

        self.conv_out0 = DRB_Block2(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out1 = DRB_Block2(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out2 = DRB_Block2(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out3 = DRB_Block2(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out4 = DRB_Block2(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.CA0 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA1 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA2 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA3 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA4 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)

    def forward(self, x):
        x1_0 = self.conv1_0(x[0])
        x1_1 = self.conv1_1(torch.cat((x1_0, x[1]), 1))
        x1_4 = self.conv1_4(x[4])
        x1_3 = self.conv1_3(torch.cat((x1_4, x[3]), 1))
        x1_2 = self.conv1_2(torch.cat((x1_1, x1_3, x[2]), 1))

        out_0 = self.CA0(self.conv_out0(torch.cat((x[0], x1_0), 1)))
        out_1 = self.CA1(self.conv_out1(torch.cat((x[1], x1_1), 1)))
        out_2 = self.CA2(self.conv_out2(torch.cat((x[2], x1_2), 1)))
        out_3 = self.CA3(self.conv_out3(torch.cat((x[3], x1_3), 1)))
        out_4 = self.CA4(self.conv_out4(torch.cat((x[4], x1_4), 1)))

        x[0] = torch.cat((out_0, x[0]), 1)
        x[1] = torch.cat((out_1, x[1]), 1)
        x[2] = torch.cat((out_2, x[2]), 1)
        x[3] = torch.cat((out_3, x[3]), 1)
        x[4] = torch.cat((out_4, x[4]), 1)

        # x.append(out_2)

        return x


class DRBModule5(nn.Module):
    def __init__(self, inout_ch, ch_reduction_ratio, bias, act, block_n):
        super(DRBModule5, self).__init__()
        self.conv1_0 = DRB_Block2(in_ch=inout_ch * block_n, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                  bias=bias, activation=act, use_norm=None)
        self.conv1_1 = DRB_Block2(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_2 = DRB_Block2(in_ch=inout_ch * block_n + 2 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_3 = DRB_Block2(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_4 = DRB_Block2(in_ch=inout_ch * block_n, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                  bias=bias, activation=act, use_norm=None)

        self.conv2_1 = DRB_Block2(in_ch=2 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1, bias=bias,
                                  activation=act, use_norm=None)
        self.conv2_2 = DRB_Block2(in_ch=5 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1, bias=bias,
                                  activation=act, use_norm=None)
        self.conv2_3 = DRB_Block2(in_ch=2 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1, bias=bias,
                                  activation=act, use_norm=None)

        self.conv3_2 = DRB_Block2(in_ch=3 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1, bias=bias,
                                  activation=act, use_norm=None)

        self.conv_out0 = DRB_Block2(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out1 = DRB_Block2(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out2 = DRB_Block2(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out3 = DRB_Block2(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out4 = DRB_Block2(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.CA0 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA1 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA2 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA3 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA4 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)

    def forward(self, x):
        x1_0 = self.conv1_0(x[0])
        x1_1 = self.conv1_1(torch.cat((x1_0, x[1]), 1))
        x1_4 = self.conv1_4(x[4])
        x1_3 = self.conv1_3(torch.cat((x1_4, x[3]), 1))
        x1_2 = self.conv1_2(torch.cat((x1_1, x1_3, x[2]), 1))

        x2_1 = self.conv2_1(torch.cat((x1_0, x1_1), 1))
        x2_3 = self.conv2_3(torch.cat((x1_4, x1_3), 1))
        x2_2 = self.conv2_2(torch.cat((x1_1, x1_2, x1_3, x2_1, x2_3), 1))

        x3_2 = self.conv3_2(torch.cat((x2_1, x2_2, x2_3), 1))

        out_0 = self.CA0(self.conv_out0(x1_0))
        out_1 = self.CA1(self.conv_out1(x2_1))
        out_2 = self.CA2(self.conv_out2(x3_2))
        out_3 = self.CA3(self.conv_out3(x2_3))
        out_4 = self.CA4(self.conv_out4(x1_4))

        x[0] = torch.cat((out_0, x[0]), 1)
        x[1] = torch.cat((out_1, x[1]), 1)
        x[2] = torch.cat((out_2, x[2]), 1)
        x[3] = torch.cat((out_3, x[3]), 1)
        x[4] = torch.cat((out_4, x[4]), 1)

        return x


class DRBModule6(nn.Module):
    def __init__(self, inout_ch, ch_reduction_ratio, bias, act, block_n):
        super(DRBModule6, self).__init__()
        self.conv1_0 = DRB_Block1(in_ch=inout_ch * block_n, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                  bias=bias, activation=act, use_norm=None)
        self.conv1_1 = DRB_Block1(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_2 = DRB_Block1(in_ch=inout_ch * block_n + 2 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_3 = DRB_Block1(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_4 = DRB_Block1(in_ch=inout_ch * block_n, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                  bias=bias, activation=act, use_norm=None)

        self.conv2_1 = DRB_Block1(in_ch=2 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1, bias=bias,
                                  activation=act, use_norm=None)
        self.conv2_2 = DRB_Block1(in_ch=5 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1, bias=bias,
                                  activation=act, use_norm=None)
        self.conv2_3 = DRB_Block1(in_ch=2 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1, bias=bias,
                                  activation=act, use_norm=None)

        self.conv3_2 = DRB_Block1(in_ch=3 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1, bias=bias,
                                  activation=act, use_norm=None)

        self.conv_out0 = DRB_Block1(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out1 = DRB_Block1(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out2 = DRB_Block1(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out3 = DRB_Block1(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out4 = DRB_Block1(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.CA0 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA1 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA2 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA3 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA4 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)

    def forward(self, x):
        x1_0 = self.conv1_0(x[0])
        x1_1 = self.conv1_1(torch.cat((x1_0, x[1]), 1))
        x1_4 = self.conv1_4(x[4])
        x1_3 = self.conv1_3(torch.cat((x1_4, x[3]), 1))
        x1_2 = self.conv1_2(torch.cat((x1_1, x1_3, x[2]), 1))

        x2_1 = self.conv2_1(torch.cat((x1_0, x1_1), 1))
        x2_3 = self.conv2_3(torch.cat((x1_4, x1_3), 1))
        x2_2 = self.conv2_2(torch.cat((x1_1, x1_2, x1_3, x2_1, x2_3), 1))

        x3_2 = self.conv3_2(torch.cat((x2_1, x2_2, x2_3), 1))

        out_0 = self.CA0(self.conv_out0(x1_0))
        out_1 = self.CA1(self.conv_out1(x2_1))
        out_2 = self.CA2(self.conv_out2(x3_2))
        out_3 = self.CA3(self.conv_out3(x2_3))
        out_4 = self.CA4(self.conv_out4(x1_4))

        x[0] = torch.cat((out_0, x[0]), 1)
        x[1] = torch.cat((out_1, x[1]), 1)
        x[2] = torch.cat((out_2, x[2]), 1)
        x[3] = torch.cat((out_3, x[3]), 1)
        x[4] = torch.cat((out_4, x[4]), 1)

        return x


class DRBModule7(nn.Module):
    def __init__(self, inout_ch, ch_reduction_ratio, bias, act, block_n):
        super(DRBModule7, self).__init__()
        self.conv1_0 = DRB_Block3(in_ch=inout_ch * block_n, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                  bias=bias, activation=act, use_norm=None)
        self.conv1_1 = DRB_Block3(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_2 = DRB_Block3(in_ch=inout_ch * block_n + 2 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_3 = DRB_Block3(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_4 = DRB_Block3(in_ch=inout_ch * block_n, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                  bias=bias, activation=act, use_norm=None)

        self.conv2_1 = DRB_Block3(in_ch=2 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1, bias=bias,
                                  activation=act, use_norm=None)
        self.conv2_2 = DRB_Block3(in_ch=5 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1, bias=bias,
                                  activation=act, use_norm=None)
        self.conv2_3 = DRB_Block3(in_ch=2 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1, bias=bias,
                                  activation=act, use_norm=None)

        self.conv3_2 = DRB_Block3(in_ch=3 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1, bias=bias,
                                  activation=act, use_norm=None)

        self.conv_out0 = DRB_Block3(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out1 = DRB_Block3(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out2 = DRB_Block3(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out3 = DRB_Block3(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out4 = DRB_Block3(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.CA0 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA1 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA2 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA3 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA4 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)

    def forward(self, x):
        x1_0 = self.conv1_0(x[0])
        x1_1 = self.conv1_1(torch.cat((x1_0, x[1]), 1))
        x1_4 = self.conv1_4(x[4])
        x1_3 = self.conv1_3(torch.cat((x1_4, x[3]), 1))
        x1_2 = self.conv1_2(torch.cat((x1_1, x1_3, x[2]), 1))

        x2_1 = self.conv2_1(torch.cat((x1_0, x1_1), 1))
        x2_3 = self.conv2_3(torch.cat((x1_4, x1_3), 1))
        x2_2 = self.conv2_2(torch.cat((x1_1, x1_2, x1_3, x2_1, x2_3), 1))

        x3_2 = self.conv3_2(torch.cat((x2_1, x2_2, x2_3), 1))

        out_0 = self.CA0(self.conv_out0(x1_0))
        out_1 = self.CA1(self.conv_out1(x2_1))
        out_2 = self.CA2(self.conv_out2(x3_2))
        out_3 = self.CA3(self.conv_out3(x2_3))
        out_4 = self.CA4(self.conv_out4(x1_4))

        x[0] = torch.cat((out_0, x[0]), 1)
        x[1] = torch.cat((out_1, x[1]), 1)
        x[2] = torch.cat((out_2, x[2]), 1)
        x[3] = torch.cat((out_3, x[3]), 1)
        x[4] = torch.cat((out_4, x[4]), 1)

        return x


class DRBModule8(nn.Module):
    def __init__(self, inout_ch, ch_reduction_ratio, bias, act, block_n):
        super(DRBModule8, self).__init__()
        self.conv1_0 = DRB_Block3(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                  bias=bias, activation=act, use_norm=None)
        self.conv1_1 = DRB_Block3(in_ch=3 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_2 = DRB_Block3(in_ch=inout_ch * block_n + 4 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_3 = DRB_Block3(in_ch=3 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_4 = DRB_Block3(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                  bias=bias, activation=act, use_norm=None)

        self.conv_out0 = DRB_Block3(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out1 = DRB_Block3(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out2 = DRB_Block3(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out3 = DRB_Block3(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out4 = DRB_Block3(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.CA0 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA1 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA2 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA3 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA4 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)

    def forward(self, x):
        x1_0 = self.conv1_0(x[0])
        x1_1 = self.conv1_1(torch.cat((x1_0, x[0], x[1]), 1))
        x1_4 = self.conv1_4(x[4])
        x1_3 = self.conv1_3(torch.cat((x1_4, x[4], x[3]), 1))

        x1_2 = self.conv1_2(torch.cat((x1_1, x1_3, x[1], x[2], x[3]), 1))

        x[0] = self.CA0(self.conv_out0(x1_0))
        x[1] = self.CA1(self.conv_out1(x1_1))
        out_2 = self.CA2(self.conv_out2(x1_2))
        x[3] = self.CA3(self.conv_out3(x1_3))
        x[4] = self.CA4(self.conv_out4(x1_4))

        x[2] = torch.cat((out_2, x[2]), 1)

        return x


class DRBModule9(nn.Module):
    def __init__(self, inout_ch, ch_reduction_ratio, bias, act, block_n):
        super(DRBModule9, self).__init__()
        self.inout_ch = inout_ch
        self.conv1_0 = DRB_Block4(in_ch=inout_ch * block_n, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                  bias=bias, activation=act, use_norm=None)
        self.conv1_1 = DRB_Block4(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_2 = DRB_Block4(in_ch=inout_ch * block_n + 2 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_3 = DRB_Block4(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_4 = DRB_Block4(in_ch=inout_ch * block_n, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                  bias=bias, activation=act, use_norm=None)

        self.conv_out0 = DRB_Block4(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out1 = DRB_Block4(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out2 = DRB_Block4(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out3 = DRB_Block4(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out4 = DRB_Block4(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.CA0 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA1 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA2 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA3 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA4 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)

    def forward(self, x):
        x1_0 = self.conv1_0(x[0])
        x1_1 = self.conv1_1(torch.cat((x1_0, x[1]), 1))
        x1_4 = self.conv1_4(x[4])
        x1_3 = self.conv1_3(torch.cat((x1_4, x[3]), 1))
        x1_2 = self.conv1_2(torch.cat((x1_1, x1_3, x[2]), 1))

        out_0 = self.CA0(self.conv_out0(torch.cat((x[0], x1_0), 1)))
        out_1 = self.CA1(self.conv_out1(torch.cat((x[1], x1_1), 1)))
        out_2 = self.CA2(self.conv_out2(torch.cat((x[2], x1_2), 1)))
        out_3 = self.CA3(self.conv_out3(torch.cat((x[3], x1_3), 1)))
        out_4 = self.CA4(self.conv_out4(torch.cat((x[4], x1_4), 1)))

        x[0] = torch.cat((out_0, x[0]), 1)
        x[1] = torch.cat((out_1, x[1]), 1)
        x[2] = torch.cat((out_2, x[2]), 1)
        x[3] = torch.cat((out_3, x[3]), 1)
        x[4] = torch.cat((out_4, x[4]), 1)

        # x.append(out_2)

        return x


class DRBModule9_1(nn.Module):
    def __init__(self, inout_ch, ch_reduction_ratio, bias, act, block_n):
        super(DRBModule9_1, self).__init__()
        self.inout_ch = inout_ch
        self.conv1_0 = DRB_Block4(in_ch=inout_ch * block_n, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                  bias=bias, activation=act, use_norm=None)
        self.conv1_1 = DRB_Block4(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_2 = DRB_Block4(in_ch=inout_ch * block_n + 2 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_3 = DRB_Block4(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_4 = DRB_Block4(in_ch=inout_ch * block_n, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                  bias=bias, activation=act, use_norm=None)

        self.conv_out0 = DRB_Block4(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out1 = DRB_Block4(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out2 = DRB_Block4(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out3 = DRB_Block4(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out4 = DRB_Block4(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.CA0 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA1 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA2 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA3 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA4 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)

    def forward(self, x):
        x1_0 = self.conv1_0(x[0])
        x1_1 = self.conv1_1(torch.cat((x1_0, x[1]), 1))
        x1_4 = self.conv1_4(x[4])
        x1_3 = self.conv1_3(torch.cat((x1_4, x[3]), 1))
        x1_2 = self.conv1_2(torch.cat((x1_1, x1_3, x[2]), 1))

        out_0 = self.CA0(self.conv_out0(x1_0))
        out_1 = self.CA1(self.conv_out1(x1_1))
        out_2 = self.CA2(self.conv_out2(x1_2))
        out_3 = self.CA3(self.conv_out3(x1_3))
        out_4 = self.CA4(self.conv_out4(x1_4))

        x[0] = torch.cat((out_0, x[0]), 1)
        x[1] = torch.cat((out_1, x[1]), 1)
        x[2] = torch.cat((out_2, x[2]), 1)
        x[3] = torch.cat((out_3, x[3]), 1)
        x[4] = torch.cat((out_4, x[4]), 1)

        # x.append(out_2)

        return x


class DRBModule9_2(nn.Module):
    def __init__(self, inout_ch, ch_reduction_ratio, bias, act, block_n):
        super(DRBModule9_2, self).__init__()
        self.inout_ch = inout_ch
        self.conv1_0 = DRB_Block4(in_ch=inout_ch * block_n, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                  bias=bias, activation=act, use_norm=None)
        self.conv1_1 = DRB_Block4(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_2 = DRB_Block4(in_ch=inout_ch * block_n + 2 * inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_3 = DRB_Block4(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_4 = DRB_Block4(in_ch=inout_ch * block_n, out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                  bias=bias, activation=act, use_norm=None)

        self.conv_out0 = DRB_Block4(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out1 = DRB_Block4(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out2 = DRB_Block4(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out3 = DRB_Block4(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out4 = DRB_Block4(in_ch=inout_ch * block_n + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.CA0 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA1 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA2 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA3 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA4 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)

    def forward(self, x):
        x1_0 = self.conv1_0(x[0])
        x1_1 = self.conv1_1(torch.cat((x1_0, x[1]), 1))
        x1_4 = self.conv1_4(x[4])
        x1_3 = self.conv1_3(torch.cat((x1_4, x[3]), 1))
        x1_2 = self.conv1_2(torch.cat((x1_1, x1_3, x[2]), 1))

        out_0 = self.CA0(self.conv_out0(torch.cat((x[0], x1_0), 1)))
        out_1 = self.CA1(self.conv_out1(torch.cat((x[1], x1_1), 1)))
        out_2 = self.CA2(self.conv_out2(torch.cat((x[2], x1_2), 1)))
        out_3 = self.CA3(self.conv_out3(torch.cat((x[3], x1_3), 1)))
        out_4 = self.CA4(self.conv_out4(torch.cat((x[4], x1_4), 1)))

        x[0] = torch.cat((out_0, x[0]), 1)
        x[1] = torch.cat((out_1, x[1]), 1)
        x[2] = torch.cat((out_2, x[2]), 1)
        x[3] = torch.cat((out_3, x[3]), 1)
        x[4] = torch.cat((out_4, x[4]), 1)

        # x.append(out_2)

        return x


class DRBModule9_3(nn.Module):
    def __init__(self, inout_ch, ch_reduction_ratio, bias, act, block_n):
        super(DRBModule9_3, self).__init__()
        self.inout_ch = inout_ch
        self.conv1_0 = DRB_Block4(in_ch=inout_ch * min(2, block_n), out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                  bias=bias, activation=act, use_norm=None)
        self.conv1_1 = DRB_Block4(in_ch=inout_ch * min(2, block_n) + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_2 = DRB_Block4(in_ch=inout_ch * min(2, block_n) + 2 * inout_ch, out_ch=inout_ch, kernel_size=3,
                                  stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_3 = DRB_Block4(in_ch=inout_ch * min(2, block_n) + inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                  padding=1, bias=bias, activation=act, use_norm=None)
        self.conv1_4 = DRB_Block4(in_ch=inout_ch * min(2, block_n), out_ch=inout_ch, kernel_size=3, stride=1, padding=1,
                                  bias=bias, activation=act, use_norm=None)

        self.conv_out0 = DRB_Block4(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out1 = DRB_Block4(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out2 = DRB_Block4(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out3 = DRB_Block4(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.conv_out4 = DRB_Block4(in_ch=inout_ch, out_ch=inout_ch, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=bias, activation=act, use_norm=None)
        self.CA0 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA1 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA2 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA3 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.CA4 = CALayer(inout_ch=inout_ch, ch_reduction_ratio=ch_reduction_ratio)

    def forward(self, x):
        x1_0 = self.conv1_0(x[0])
        x1_1 = self.conv1_1(torch.cat((x1_0, x[1]), 1))
        x1_4 = self.conv1_4(x[4])
        x1_3 = self.conv1_3(torch.cat((x1_4, x[3]), 1))
        x1_2 = self.conv1_2(torch.cat((x1_1, x1_3, x[2]), 1))

        out_0 = self.CA0(self.conv_out0(x1_0))
        out_1 = self.CA1(self.conv_out1(x1_1))
        out_2 = self.CA2(self.conv_out2(x1_2))
        out_3 = self.CA3(self.conv_out3(x1_3))
        out_4 = self.CA4(self.conv_out4(x1_4))

        x[0] = torch.cat((out_0, x1_0), 1)
        x[1] = torch.cat((out_1, x1_1), 1)
        x[2] = torch.cat((out_2, x1_2), 1)
        x[3] = torch.cat((out_3, x1_3), 1)
        x[4] = torch.cat((out_4, x1_4), 1)

        # x.append(out_2)

        return x
