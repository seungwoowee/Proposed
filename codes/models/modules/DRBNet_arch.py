# import sys
# sys.path.append('models\modules\GMA_core')

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
from utils import util
from data.util import read_img

from models.modules.GMA_core.GMA_network import RAFTGMA
from models.modules.GMA_core.GMA_utils.utils import InputPadder

import numpy as np
from PIL import Image
import cv2

import torchvision.transforms as transforms

import functools
import models.modules.module_util as mutil


def load_image(path):
    img = np.array(Image.open(path)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()

    return img[None].to('cuda')


def show_PIL_image(imgs):
    tf = transforms.ToPILImage()

    dst = Image.new('RGB', (imgs.shape[-1], imgs.shape[-2] * len(imgs)))
    dst.paste(tf(imgs.detach()[0].float().cpu()), (0, 0))
    # dst.show()
    for i in range(len(imgs) - 1):
        img = tf(imgs.detach()[i + 1].float().cpu())
        dst.paste(img, (0, img.height * (i + 1)))

    return dst


def flow_cal(ref_img, src_img, model):
    padder = InputPadder(ref_img.shape)
    ref_img, src_img = padder.pad(ref_img, src_img)
    _, flow = model(ref_img, src_img, iters=12, test_mode=True)
    return padder.unpad(flow)


def flow_cal_backwarp(ref_img, src_img, model):
    padder = InputPadder(ref_img.shape)
    ref_img, src_img = padder.pad(ref_img, src_img)
    _, flow = model(ref_img, src_img, iters=12, test_mode=True)
    out = backwarp(padder.unpad(src_img), padder.unpad(flow))
    return out


def backwarp(img, flow):
    _, _, H, W = img.size()

    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]

    gridW, gridH = np.meshgrid(np.arange(W), np.arange(H))

    gridW = torch.tensor(gridW, requires_grad=False, device=torch.device('cuda'))
    gridH = torch.tensor(gridH, requires_grad=False, device=torch.device('cuda'))
    x = gridW.unsqueeze(0).expand_as(u).float() + u
    y = gridH.unsqueeze(0).expand_as(v).float() + v
    # range -1 to 1
    x = 2 * (x / W - 0.5)
    y = 2 * (y / H - 0.5)
    # stacking X and Y
    grid = torch.stack((x, y), dim=3)
    # Sample pixels using bilinear interpolation.
    imgOut = torch.nn.functional.grid_sample(img, grid, align_corners=True)  # I2 , F12
    return imgOut


flow_parser = argparse.ArgumentParser()
flow_parser.add_argument('--model', help="restore checkpoint",
                         default="models/modules/GMA_checkpoints/gma-sintel.pth")
flow_parser.add_argument('--model_name', help="define model name", default="GMA")
flow_parser.add_argument('--path', help="dataset for evaluation",
                         default="models/modules/GMA_imgs")
flow_parser.add_argument('--num_heads', default=1, type=int,
                         help='number of heads in attention and aggregation')
flow_parser.add_argument('--position_only', default=False, action='store_true',
                         help='only use position-wise attention')
flow_parser.add_argument('--position_and_content', default=False, action='store_true',
                         help='use position and content-wise attention')
flow_parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
flow_args = flow_parser.parse_args()


class ResModule(nn.Module):
    def __init__(self, inout_ch, kernel_size=3, bias=False, bn=False, res_scale=1):
        super(ResModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inout_ch, out_channels=inout_ch, kernel_size=kernel_size, stride=1,
                               padding=(kernel_size // 2), bias=bias)
        self.conv2 = nn.Conv2d(in_channels=inout_ch * 2, out_channels=inout_ch, kernel_size=kernel_size, stride=1,
                               padding=(kernel_size // 2), bias=bias)
        self.conv3 = nn.Conv2d(in_channels=inout_ch * 3, out_channels=inout_ch, kernel_size=kernel_size, stride=1,
                               padding=(kernel_size // 2), bias=bias)
        self.conv4 = nn.Conv2d(in_channels=inout_ch * 4, out_channels=inout_ch, kernel_size=kernel_size, stride=1,
                               padding=(kernel_size // 2), bias=bias)
        self.conv5 = nn.Conv2d(in_channels=inout_ch * 5, out_channels=inout_ch, kernel_size=kernel_size, stride=1,
                               padding=(kernel_size // 2), bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.res_scale = res_scale

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5 * self.res_scale + x


class ResModule_down(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, bias=False, bn=False,
                 act=nn.LeakyReLU(negative_slope=0.1, inplace=True), res_scale=1):
        super(ResModule_down, self).__init__()
        module_temp_down = [nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=kernel_size, stride=2,
                                      padding=(kernel_size // 2), bias=bias), act]
        self.module_down = nn.Sequential(*module_temp_down)

        module_temp = []
        for i in range(2):
            module_temp.append(nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=kernel_size, stride=1,
                                         padding=(kernel_size // 2), bias=bias))
            if i == 0:
                if bn:
                    module_temp.append(nn.BatchNorm2d(in_ch))
                module_temp.append(act)

        self.module = nn.Sequential(*module_temp)
        self.res_scale = res_scale

        self.conv_out = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1,
                                  padding=0, bias=bias)

    def forward(self, x):
        x = self.module_down(x)
        res = self.module(x) * self.res_scale
        out = self.conv_out(res + x)

        return out


class scale_up_x2_deconv(nn.Module):
    def __init__(self, ch):
        super(scale_up_x2_deconv, self).__init__()
        self.deconv = nn.ConvTranspose2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        out = self.deconv(x, output_size=[x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2])

        return out


class PixelShufflePack(nn.Module):
    """ Pixel Shuffle upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels):
        super(PixelShufflePack, self).__init__()
        self.upsample_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels * 4,
                                       kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        """Forward function for PixelShufflePack.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, 2)
        x = self.lrelu(x)

        return x


class CALayer(nn.Module):
    def __init__(self, inout_ch, ch_reduction_ratio):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(inout_ch, inout_ch // ch_reduction_ratio, 1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(inout_ch // ch_reduction_ratio, inout_ch, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class U_shaped_Net_with_CA_dense(nn.Module):
    def __init__(self, in_ch, out_ch, bias, bn, act, res_scale, ch_reduction_ratio):
        super(U_shaped_Net_with_CA_dense, self).__init__()
        self.resB_down = ResModule_down(in_ch=in_ch, out_ch=4 * in_ch, bias=bias, bn=bn, act=act, res_scale=res_scale)
        self.CA = CALayer(inout_ch=2 * in_ch, ch_reduction_ratio=ch_reduction_ratio)
        self.resB = ResModule(inout_ch=2 * in_ch, bias=bias, bn=bn, res_scale=res_scale)
        self.conv_out = nn.Conv2d(in_channels=2 * in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1,
                                  bias=False)
        self.res_scale = res_scale

    def forward(self, x):
        out = self.resB_down(x)
        out = F.pixel_shuffle(out, 2)
        out = torch.cat((x, out), 1)
        out = self.CA(out)
        out = self.resB(out)
        out = self.conv_out(out)
        out = out * self.res_scale + x

        return out


class img_to_feat(nn.Module):
    def __init__(self, in_ch, out_ch, use_bias):
        super(img_to_feat, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=use_bias)
        )

    def forward(self, x):
        out = self.features(x)
        return out


### final model
class DRBNet_mid(nn.Module):
    def __init__(self):
        super(DRBNet_mid, self).__init__()

        ### optical flow: GMA   ###################################
        self.GMA_model = torch.nn.DataParallel(RAFTGMA(flow_args))
        self.GMA_model.load_state_dict(torch.load(flow_args.model))
        self.GMA_model = self.GMA_model.module
        self.GMA_model.to('cuda')
        self.GMA_model.eval()
        ###########################################################

        use_act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        use_bias = True
        use_bn = False

        res_scale = 0.2
        block_n = 4

        in_ch = 64
        ch_reduction_ratio = 16
        self.in_feat0 = img_to_feat(in_ch=8, out_ch=in_ch, use_bias=use_bias)
        self.in_feat1 = img_to_feat(in_ch=8, out_ch=in_ch, use_bias=use_bias)
        self.in_feat2 = img_to_feat(in_ch=3, out_ch=in_ch, use_bias=use_bias)
        self.in_feat3 = img_to_feat(in_ch=8, out_ch=in_ch, use_bias=use_bias)
        self.in_feat4 = img_to_feat(in_ch=8, out_ch=in_ch, use_bias=use_bias)

        self.CA = CALayer(inout_ch=5 * in_ch, ch_reduction_ratio=ch_reduction_ratio)

        self.conv_1d_in = nn.Conv2d(in_channels=5 * in_ch, out_channels=in_ch, kernel_size=1, stride=1, padding=0,
                                    bias=use_bias)

        module_U_net = [
            U_shaped_Net_with_CA_dense(in_ch=in_ch, out_ch=in_ch, bias=use_bias, bn=use_bn, act=use_act,
                                       res_scale=res_scale, ch_reduction_ratio=ch_reduction_ratio)
            for _ in range(block_n)]

        self.U_net = nn.Sequential(*module_U_net)

        self.conv_1d_out = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, stride=1, padding=0,
                                     bias=use_bias)

        self.upsample1 = PixelShufflePack(in_channels=in_ch, out_channels=in_ch)
        self.upsample2 = PixelShufflePack(in_channels=in_ch, out_channels=in_ch)

        self.HRconv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1,
                                bias=use_bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_out = nn.Conv2d(in_channels=in_ch, out_channels=3, kernel_size=3, stride=1, padding=1,
                                  bias=use_bias)
        self.img_upsample_x4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        util.initialize_weights(
            [self.in_feat0, self.in_feat1, self.in_feat2, self.in_feat3, self.in_feat4, self.CA, self.U_net,
             self.conv_1d_in, self.conv_1d_out, self.upsample1, self.upsample2, self.HRconv, self.conv_out])
        util.initialize_weights([self.U_net], scale=0.1)

    def forward(self, x):
        src_img = x[2]

        ref_img = x[0]
        flow_ = flow_cal(ref_img, src_img, self.GMA_model)
        x0 = torch.cat((ref_img, flow_, x[2]), 1)

        ref_img = x[1]
        flow_ = flow_cal(ref_img, src_img, self.GMA_model)
        x1 = torch.cat((ref_img, flow_, x[2]), 1)

        ref_img = x[3]
        flow_ = flow_cal(ref_img, src_img, self.GMA_model)
        x3 = torch.cat((ref_img, flow_, x[2]), 1)

        ref_img = x[4]
        flow_ = flow_cal(ref_img, src_img, self.GMA_model)
        x4 = torch.cat((ref_img, flow_, x[2]), 1)

        x0 = self.in_feat0(x0)
        x1 = self.in_feat1(x1)
        x2 = self.in_feat2(x[2])
        x3 = self.in_feat3(x3)
        x4 = self.in_feat4(x4)

        ## show image
        # show_PIL_image(x[0][:, [2, 1, 0], :, :]).show()

        out = self.CA(torch.cat((x0, x1, x2, x3, x4), 1))
        out = self.conv_1d_in(out)
        out = self.U_net(out)
        out = self.conv_1d_out(out)
        out = self.upsample1(out)  # conv shuffle lrelu
        out = self.upsample2(out)  # conv shuffle lrelu

        out = self.conv_out(self.lrelu(self.HRconv(out)))
        src_img = self.img_upsample_x4(src_img)
        out += src_img
        return out
