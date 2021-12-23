# import sys
# sys.path.append('models\modules\GMA_core')

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from utils import util

from models.modules.GMA_core.GMA_network import RAFTGMA
from models.modules.GMA_core.GMA_utils.utils import InputPadder

import numpy as np
from PIL import Image

import torchvision.transforms as transforms
from models.modules import module_util


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


def flow_cal(src_img, ref_img, model):
    padder = InputPadder(src_img.shape)
    src_img_pad, ref_img_pad = padder.pad(src_img, ref_img)
    _, flow = model(src_img_pad, ref_img_pad, iters=12, test_mode=True)
    return padder.unpad(flow)


def flow_cal_backwarp(src_img, ref_img, model):
    padder = InputPadder(src_img.shape)
    src_img_pad, ref_img_pad = padder.pad(src_img, ref_img)
    _, flow = model(src_img_pad, ref_img_pad, iters=12, test_mode=True)
    out = backwarp(padder.unpad(ref_img_pad), padder.unpad(flow), src_img)
    return out


def flow_cal_backwarp2(src_img, ref_img, model):
    padder = InputPadder(src_img.shape)
    src_img_pad, ref_img_pad = padder.pad(src_img, ref_img)
    _, flow = model(src_img_pad, ref_img_pad, iters=12, test_mode=True)
    out = backwarp(padder.unpad(src_img_pad), padder.unpad(flow), src_img)
    return out


def backwarp(ref_img, flow, src_img):
    _, _, H, W = ref_img.size()

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

    # x = 2 * ((x - x.min()) / (x.max() - x.min()) - 0.5)
    # y = 2 * ((y - y.min()) / (y.max() - y.min()) - 0.5)
    # stacking X and Y
    grid = torch.stack((x, y), dim=3)
    # Sample pixels using bilinear interpolation.
    imgOut = torch.nn.functional.grid_sample(ref_img, grid)  # I2 , F12
    # show_PIL_image(x[0][:, [2, 1, 0], :, :]).show()
    a = imgOut == 0
    imgOut = imgOut + src_img.mul(a)
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


### final model
class DRBNet1(nn.Module):
    def __init__(self, rgb_range, scale):
        super(DRBNet1, self).__init__()

        ### optical flow: GMA   ###################################
        self.GMA_model = torch.nn.DataParallel(RAFTGMA(flow_args))
        self.GMA_model.load_state_dict(torch.load(flow_args.model))
        self.GMA_model = self.GMA_model.module
        self.GMA_model.to('cuda')
        self.GMA_model.eval()
        ###########################################################

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        use_bias = True
        use_act = 'prelu'  # 'relu' 'prelu'  'lrelu' 'tanh' 'sigmoid'
        use_norm = None  # batch / instance

        feat = 256
        unit_feat = 64

        self.feat0 = module_util.ConvModule(in_ch=3 * 5, out_ch=feat * 5, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat1 = module_util.ConvModule(in_ch=feat * 5, out_ch=unit_feat * 5, kernel_size=1, stride=1, padding=0,
                                            bias=use_bias, activation=use_act, norm=use_norm)

        ch_reduction_ratio = 16
        block_n = 6

        module_DRB = [
            module_util.DRBModule(inout_ch=unit_feat, ch_reduction_ratio=ch_reduction_ratio, bias=use_bias, act=use_act,
                                  block_n=k + 1)
            for k in range(block_n)]

        self.dense_DRB = nn.Sequential(*module_DRB)

        self.HRconv = nn.Conv2d(in_channels=unit_feat + block_n * unit_feat, out_channels=unit_feat, kernel_size=3,
                                stride=1, padding=1,
                                bias=use_bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_out = nn.Conv2d(in_channels=unit_feat, out_channels=scale ** 2 * 3, kernel_size=3, stride=1,
                                  padding=1,
                                  bias=use_bias)

        self.upsample = module_util.PixelShufflePack(in_channels=scale ** 2 * 3, scale=scale)

        self.img_upsample_x4 = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)

    def forward(self, x):
        x[0] = self.sub_mean(x[0])
        x[1] = self.sub_mean(x[1])
        x[2] = self.sub_mean(x[2])
        x[3] = self.sub_mean(x[3])
        x[4] = self.sub_mean(x[4])

        src_img = x[2]
        x[0] = flow_cal_backwarp(src_img, x[0], self.GMA_model)
        x[1] = flow_cal_backwarp(src_img, x[1], self.GMA_model)
        x[3] = flow_cal_backwarp(src_img, x[3], self.GMA_model)
        x[4] = flow_cal_backwarp(src_img, x[4], self.GMA_model)

        x = self.feat0(torch.cat((x[0], x[1], x[2], x[3], x[4]), 1))
        x = self.feat1(x)

        tmp = torch.split(x[0], 64)
        x = []
        for k in range(len(tmp)):
            x.append(tmp[k].unsqueeze(0))

        out = self.dense_DRB(x)

        ## show image
        # show_PIL_image(x[0][:, [2, 1, 0], :, :]).show()

        out = self.conv_out(self.lrelu(self.HRconv(out[2])))

        out = self.upsample(out)  # conv shuffle lrelu

        out = self.add_mean(out)

        src_img = self.img_upsample_x4(src_img)

        out += src_img

        return out


class DRBNet2(nn.Module):
    def __init__(self, rgb_range, scale):
        super(DRBNet2, self).__init__()

        ### optical flow: GMA   ###################################
        self.GMA_model = torch.nn.DataParallel(RAFTGMA(flow_args))
        self.GMA_model.load_state_dict(torch.load(flow_args.model))
        self.GMA_model = self.GMA_model.module
        self.GMA_model.to('cuda')
        self.GMA_model.eval()
        ###########################################################

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)

        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        use_bias = True
        use_act = 'prelu'  # 'relu' 'prelu'  'lrelu' 'tanh' 'sigmoid'
        use_norm = None  # batch / instance

        feat = 256
        unit_feat = 64

        self.feat0 = module_util.ConvModule(in_ch=3 * 5, out_ch=feat * 5, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat1 = module_util.ConvModule(in_ch=feat * 5, out_ch=unit_feat * 5, kernel_size=1, stride=1, padding=0,
                                            bias=use_bias, activation=use_act, norm=use_norm)

        ch_reduction_ratio = 16
        block_n = 2

        module_DRB = [
            module_util.DRBModule2(inout_ch=unit_feat, ch_reduction_ratio=ch_reduction_ratio, bias=use_bias,
                                   act=use_act,
                                   block_n=k + 1)
            for k in range(block_n)]

        self.dense_DRB = nn.Sequential(*module_DRB)

        self.HRconv = nn.Conv2d(in_channels=unit_feat + block_n * unit_feat, out_channels=unit_feat, kernel_size=3,
                                stride=1, padding=1,
                                bias=use_bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_out = nn.Conv2d(in_channels=unit_feat, out_channels=scale ** 2 * 3, kernel_size=3, stride=1,
                                  padding=1,
                                  bias=use_bias)

        self.upsample = module_util.PixelShufflePack(in_channels=scale ** 2 * 3, scale=scale)

        self.img_upsample_x4 = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)

        # self.down_up = module_util.DownUpModule(inout_ch=inout_ch, kernel_size=3, stride=1, padding=1, bias=use_bias, activation=act,
        #                             norm=None, block_n=block_n)

    def forward(self, x):
        x[0] = self.sub_mean(x[0])
        x[1] = self.sub_mean(x[1])
        x[2] = self.sub_mean(x[2])
        x[3] = self.sub_mean(x[3])
        x[4] = self.sub_mean(x[4])

        src_img = x[2]
        x[0] = flow_cal_backwarp(src_img, x[0], self.GMA_model)
        x[1] = flow_cal_backwarp(src_img, x[1], self.GMA_model)
        x[3] = flow_cal_backwarp(src_img, x[3], self.GMA_model)
        x[4] = flow_cal_backwarp(src_img, x[4], self.GMA_model)

        x = self.feat0(torch.cat((x[0], x[1], x[2], x[3], x[4]), 1))
        x = self.feat1(x)

        tmp = torch.split(x[0], 64)
        x = []
        for k in range(len(tmp)):
            x.append(tmp[k].unsqueeze(0))

        out = self.dense_DRB(x)

        ## show image
        # show_PIL_image(x[0][:, [2, 1, 0], :, :]).show()

        out = self.conv_out(self.lrelu(self.HRconv(out[2])))

        out = self.upsample(out)  # conv shuffle lrelu

        out = self.add_mean(out)

        src_img = self.img_upsample_x4(src_img)

        out += src_img

        return out


class DRBNet3(nn.Module):
    def __init__(self, rgb_range, scale):
        super(DRBNet3, self).__init__()

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)

        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        use_bias = True
        use_act = 'prelu'  # 'relu' 'prelu'  'lrelu' 'tanh' 'sigmoid'
        use_norm = None  # batch / instance

        feat = 256
        unit_feat = 64

        self.feat0 = module_util.ConvModule(in_ch=3 * 5, out_ch=feat * 5, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat1 = module_util.ConvModule(in_ch=feat * 5, out_ch=unit_feat * 5, kernel_size=1, stride=1, padding=0,
                                            bias=use_bias, activation=use_act, norm=use_norm)

        ch_reduction_ratio = 16
        block_n = 6

        module_DRB = [
            module_util.DRBModule2(inout_ch=unit_feat, ch_reduction_ratio=ch_reduction_ratio, bias=use_bias,
                                   act=use_act,
                                   block_n=k + 1)
            for k in range(block_n)]

        self.dense_DRB = nn.Sequential(*module_DRB)

        self.HRconv = nn.Conv2d(in_channels=unit_feat + block_n * unit_feat, out_channels=unit_feat, kernel_size=3,
                                stride=1, padding=1,
                                bias=use_bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_out = nn.Conv2d(in_channels=unit_feat, out_channels=scale ** 2 * 3, kernel_size=3, stride=1,
                                  padding=1,
                                  bias=use_bias)

        self.upsample = module_util.PixelShufflePack(in_channels=scale ** 2 * 3, scale=scale)

        self.img_upsample_x4 = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)

        # self.down_up = module_util.DownUpModule(inout_ch=inout_ch, kernel_size=3, stride=1, padding=1, bias=use_bias, activation=act,
        #                             norm=None, block_n=block_n)

    def forward(self, x):
        x[0] = self.sub_mean(x[0])
        x[1] = self.sub_mean(x[1])
        x[2] = self.sub_mean(x[2])
        x[3] = self.sub_mean(x[3])
        x[4] = self.sub_mean(x[4])

        src_img = x[2]

        x = self.feat0(torch.cat((x[0], x[1], x[2], x[3], x[4]), 1))
        x = self.feat1(x)

        tmp = torch.split(x[0], 64)
        x = []
        for k in range(len(tmp)):
            x.append(tmp[k].unsqueeze(0))

        out = self.dense_DRB(x)

        ## show image
        # show_PIL_image(x[0][:, [2, 1, 0], :, :]).show()

        out = self.conv_out(self.lrelu(self.HRconv(out[2])))

        out = self.upsample(out)  # conv shuffle lrelu

        out = self.add_mean(out)

        src_img = self.img_upsample_x4(src_img)

        out += src_img

        return out


class DRBNet4(nn.Module):
    def __init__(self, rgb_range, scale):
        super(DRBNet4, self).__init__()

        ### optical flow: GMA   ###################################
        self.GMA_model = torch.nn.DataParallel(RAFTGMA(flow_args))
        self.GMA_model.load_state_dict(torch.load(flow_args.model))
        self.GMA_model = self.GMA_model.module
        self.GMA_model.to('cuda')
        self.GMA_model.eval()
        ###########################################################

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        use_bias = True
        use_act = 'prelu'  # 'relu' 'prelu'  'lrelu' 'tanh' 'sigmoid'
        use_norm = None  # batch / instance

        feat = 256
        unit_feat = 64

        self.feat0 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat1 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat2 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat3 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat4 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)

        self.feat01 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat11 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat21 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat31 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat41 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)

        ch_reduction_ratio = 16
        block_n = 6

        module_DRB = [
            module_util.DRBModule4(inout_ch=unit_feat, ch_reduction_ratio=ch_reduction_ratio, bias=use_bias,
                                   act=use_act,
                                   block_n=k + 1)
            for k in range(block_n)]

        self.dense_DRB = nn.Sequential(*module_DRB)

        self.HRconv = nn.Conv2d(in_channels=unit_feat + block_n * unit_feat, out_channels=unit_feat, kernel_size=3,
                                stride=1, padding=1,
                                bias=use_bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_out = nn.Conv2d(in_channels=unit_feat, out_channels=scale ** 2 * 3, kernel_size=3, stride=1,
                                  padding=1,
                                  bias=use_bias)

        self.upsample = module_util.PixelShufflePack(in_channels=scale ** 2 * 3, scale=scale)

        self.img_upsample_x4 = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)

    def forward(self, x):
        x[0] = self.sub_mean(x[0])
        x[1] = self.sub_mean(x[1])
        x[2] = self.sub_mean(x[2])
        x[3] = self.sub_mean(x[3])
        x[4] = self.sub_mean(x[4])

        src_img = x[2]
        x[0] = flow_cal_backwarp(src_img, x[0], self.GMA_model)
        x[1] = flow_cal_backwarp(src_img, x[1], self.GMA_model)
        x[3] = flow_cal_backwarp(src_img, x[3], self.GMA_model)
        x[4] = flow_cal_backwarp(src_img, x[4], self.GMA_model)

        x[0] = self.feat01(self.feat0(x[0]))
        x[1] = self.feat11(self.feat1(x[1]))
        x[2] = self.feat21(self.feat2(x[2]))
        x[3] = self.feat31(self.feat3(x[3]))
        x[4] = self.feat41(self.feat4(x[4]))

        # tmp = torch.split(x[0], 64)
        # x = []
        # for k in range(len(tmp)):
        #     x.append(tmp[k].unsqueeze(0))

        out = self.dense_DRB(x)

        out = self.conv_out(self.lrelu(self.HRconv(out[2])))

        out = self.upsample(out)  # conv shuffle lrelu

        out = self.add_mean(out)

        src_img = self.img_upsample_x4(src_img)

        out += src_img

        ## show image
        # show_PIL_image(out[:, [2, 1, 0], :, :]).show()

        return out


class DRBNet5(nn.Module):
    def __init__(self, rgb_range, scale):
        super(DRBNet5, self).__init__()

        ### optical flow: GMA   ###################################
        self.GMA_model = torch.nn.DataParallel(RAFTGMA(flow_args))
        self.GMA_model.load_state_dict(torch.load(flow_args.model))
        self.GMA_model = self.GMA_model.module
        self.GMA_model.to('cuda')
        self.GMA_model.eval()
        ###########################################################

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        use_bias = True
        use_act = 'prelu'  # 'relu' 'prelu'  'lrelu' 'tanh' 'sigmoid'
        use_norm = None  # batch / instance

        feat = 256
        unit_feat = 32

        self.feat0 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat1 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat2 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat3 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat4 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)

        self.feat01 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat11 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat21 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat31 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat41 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)

        ch_reduction_ratio = 16
        block_n = 6

        module_DRB = [
            module_util.DRBModule5(inout_ch=unit_feat, ch_reduction_ratio=ch_reduction_ratio, bias=use_bias,
                                   act=use_act,
                                   block_n=k + 1)
            for k in range(block_n)]

        self.dense_DRB = nn.Sequential(*module_DRB)

        self.HRconv = nn.Conv2d(in_channels=unit_feat + block_n * unit_feat, out_channels=unit_feat, kernel_size=3,
                                stride=1, padding=1,
                                bias=use_bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_out = nn.Conv2d(in_channels=unit_feat, out_channels=scale ** 2 * 3, kernel_size=3, stride=1,
                                  padding=1,
                                  bias=use_bias)

        self.upsample = module_util.PixelShufflePack(in_channels=scale ** 2 * 3, scale=scale)

        self.img_upsample_x4 = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)

    def forward(self, x):
        x[0] = self.sub_mean(x[0])
        x[1] = self.sub_mean(x[1])
        x[2] = self.sub_mean(x[2])
        x[3] = self.sub_mean(x[3])
        x[4] = self.sub_mean(x[4])

        src_img = x[2]
        x[0] = flow_cal_backwarp(src_img, x[0], self.GMA_model)
        x[1] = flow_cal_backwarp(src_img, x[1], self.GMA_model)
        x[3] = flow_cal_backwarp(src_img, x[3], self.GMA_model)
        x[4] = flow_cal_backwarp(src_img, x[4], self.GMA_model)

        x[0] = self.feat01(self.feat0(x[0]))
        x[1] = self.feat11(self.feat1(x[1]))
        x[2] = self.feat21(self.feat2(x[2]))
        x[3] = self.feat31(self.feat3(x[3]))
        x[4] = self.feat41(self.feat4(x[4]))

        # tmp = torch.split(x[0], 64)
        # x = []
        # for k in range(len(tmp)):
        #     x.append(tmp[k].unsqueeze(0))

        out = self.dense_DRB(x)

        ## show image
        # show_PIL_image(out[:, [2, 1, 0], :, :]).show()

        out = self.conv_out(self.lrelu(self.HRconv(out[2])))

        out = self.upsample(out)  # conv shuffle lrelu

        out = self.add_mean(out)

        src_img = self.img_upsample_x4(src_img)

        out += src_img

        return out


class DRBNet6(nn.Module):
    def __init__(self, rgb_range, scale):
        super(DRBNet6, self).__init__()

        ### optical flow: GMA   ###################################
        self.GMA_model = torch.nn.DataParallel(RAFTGMA(flow_args))
        self.GMA_model.load_state_dict(torch.load(flow_args.model))
        self.GMA_model = self.GMA_model.module
        self.GMA_model.to('cuda')
        self.GMA_model.eval()
        ###########################################################

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        use_bias = True
        use_act = 'prelu'  # 'relu' 'prelu'  'lrelu' 'tanh' 'sigmoid'
        use_norm = None  # batch / instance

        feat = 256
        unit_feat = 32

        self.feat0 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat1 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat2 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat3 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat4 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)

        self.feat01 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat11 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat21 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat31 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat41 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)

        ch_reduction_ratio = 16
        block_n = 6

        module_DRB = [
            module_util.DRBModule6(inout_ch=unit_feat, ch_reduction_ratio=ch_reduction_ratio, bias=use_bias,
                                   act=use_act,
                                   block_n=k + 1)
            for k in range(block_n)]

        self.dense_DRB = nn.Sequential(*module_DRB)

        self.HRconv = nn.Conv2d(in_channels=unit_feat + block_n * unit_feat, out_channels=unit_feat, kernel_size=3,
                                stride=1, padding=1,
                                bias=use_bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_out = nn.Conv2d(in_channels=unit_feat, out_channels=scale ** 2 * 3, kernel_size=3, stride=1,
                                  padding=1,
                                  bias=use_bias)

        self.upsample = module_util.PixelShufflePack(in_channels=scale ** 2 * 3, scale=scale)

        self.img_upsample_x4 = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)

    def forward(self, x):
        x[0] = self.sub_mean(x[0])
        x[1] = self.sub_mean(x[1])
        x[2] = self.sub_mean(x[2])
        x[3] = self.sub_mean(x[3])
        x[4] = self.sub_mean(x[4])

        src_img = x[2]
        x[0] = flow_cal_backwarp(src_img, x[0], self.GMA_model)
        x[1] = flow_cal_backwarp(src_img, x[1], self.GMA_model)
        x[3] = flow_cal_backwarp(src_img, x[3], self.GMA_model)
        x[4] = flow_cal_backwarp(src_img, x[4], self.GMA_model)

        x[0] = self.feat01(self.feat0(x[0]))
        x[1] = self.feat11(self.feat1(x[1]))
        x[2] = self.feat21(self.feat2(x[2]))
        x[3] = self.feat31(self.feat3(x[3]))
        x[4] = self.feat41(self.feat4(x[4]))

        # tmp = torch.split(x[0], 64)
        # x = []
        # for k in range(len(tmp)):
        #     x.append(tmp[k].unsqueeze(0))

        out = self.dense_DRB(x)

        ## show image
        # show_PIL_image(out[:, [2, 1, 0], :, :]).show()

        out = self.conv_out(self.lrelu(self.HRconv(out[2])))

        out = self.upsample(out)  # conv shuffle lrelu

        out = self.add_mean(out)

        src_img = self.img_upsample_x4(src_img)

        out += src_img

        return out


class DRBNet7(nn.Module):
    def __init__(self, rgb_range, scale):
        super(DRBNet7, self).__init__()

        ### optical flow: GMA   ###################################
        self.GMA_model = torch.nn.DataParallel(RAFTGMA(flow_args))
        self.GMA_model.load_state_dict(torch.load(flow_args.model))
        self.GMA_model = self.GMA_model.module
        self.GMA_model.to('cuda')
        self.GMA_model.eval()
        ###########################################################

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        use_bias = True
        use_act = 'prelu'  # 'relu' 'prelu'  'lrelu' 'tanh' 'sigmoid'
        use_norm = None  # batch / instance

        feat = 256
        unit_feat = 32

        self.feat0 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat1 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat2 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat3 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat4 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)

        self.feat01 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat11 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat21 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat31 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat41 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)

        ch_reduction_ratio = 16
        block_n = 6

        module_DRB = [
            module_util.DRBModule7(inout_ch=unit_feat, ch_reduction_ratio=ch_reduction_ratio, bias=use_bias,
                                   act=use_act,
                                   block_n=k + 1)
            for k in range(block_n)]

        self.dense_DRB = nn.Sequential(*module_DRB)

        self.HRconv = nn.Conv2d(in_channels=unit_feat + block_n * unit_feat, out_channels=unit_feat, kernel_size=3,
                                stride=1, padding=1,
                                bias=use_bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_out = nn.Conv2d(in_channels=unit_feat, out_channels=scale ** 2 * 3, kernel_size=3, stride=1,
                                  padding=1,
                                  bias=use_bias)

        self.upsample = module_util.PixelShufflePack(in_channels=scale ** 2 * 3, scale=scale)

        self.img_upsample_x4 = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)

    def forward(self, x):
        x[0] = self.sub_mean(x[0])
        x[1] = self.sub_mean(x[1])
        x[2] = self.sub_mean(x[2])
        x[3] = self.sub_mean(x[3])
        x[4] = self.sub_mean(x[4])

        src_img = x[2]
        x[0] = flow_cal_backwarp(src_img, x[0], self.GMA_model)
        x[1] = flow_cal_backwarp(src_img, x[1], self.GMA_model)
        x[3] = flow_cal_backwarp(src_img, x[3], self.GMA_model)
        x[4] = flow_cal_backwarp(src_img, x[4], self.GMA_model)

        x[0] = self.feat01(self.feat0(x[0]))
        x[1] = self.feat11(self.feat1(x[1]))
        x[2] = self.feat21(self.feat2(x[2]))
        x[3] = self.feat31(self.feat3(x[3]))
        x[4] = self.feat41(self.feat4(x[4]))

        # tmp = torch.split(x[0], 64)
        # x = []
        # for k in range(len(tmp)):
        #     x.append(tmp[k].unsqueeze(0))

        out = self.dense_DRB(x)

        ## show image
        # show_PIL_image(out[:, [2, 1, 0], :, :]).show()

        out = self.conv_out(self.lrelu(self.HRconv(out[2])))

        out = self.upsample(out)  # conv shuffle lrelu

        out = self.add_mean(out)

        src_img = self.img_upsample_x4(src_img)

        out += src_img

        return out


class DRBNet8(nn.Module):
    def __init__(self, rgb_range, scale):
        super(DRBNet8, self).__init__()

        ### optical flow: GMA   ###################################
        self.GMA_model = torch.nn.DataParallel(RAFTGMA(flow_args))
        self.GMA_model.load_state_dict(torch.load(flow_args.model))
        self.GMA_model = self.GMA_model.module
        self.GMA_model.to('cuda')
        self.GMA_model.eval()
        ###########################################################

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        use_bias = True
        use_act = 'prelu'  # 'relu' 'prelu'  'lrelu' 'tanh' 'sigmoid'
        use_norm = None  # batch / instance

        feat = 256
        unit_feat = 32

        self.feat0 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat1 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat2 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat3 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat4 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)

        self.feat01 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat11 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat21 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat31 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat41 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)

        ch_reduction_ratio = 16
        block_n = 6

        module_DRB = [
            module_util.DRBModule8(inout_ch=unit_feat, ch_reduction_ratio=ch_reduction_ratio, bias=use_bias,
                                   act=use_act,
                                   block_n=k + 1)
            for k in range(block_n)]

        self.dense_DRB = nn.Sequential(*module_DRB)

        self.HRconv = nn.Conv2d(in_channels=unit_feat + block_n * unit_feat, out_channels=unit_feat, kernel_size=3,
                                stride=1, padding=1,
                                bias=use_bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_out = nn.Conv2d(in_channels=unit_feat, out_channels=scale ** 2 * 3, kernel_size=3, stride=1,
                                  padding=1,
                                  bias=use_bias)

        self.upsample = module_util.PixelShufflePack(in_channels=scale ** 2 * 3, scale=scale)

        self.img_upsample_x4 = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)

    def forward(self, x):
        x[0] = self.sub_mean(x[0])
        x[1] = self.sub_mean(x[1])
        x[2] = self.sub_mean(x[2])
        x[3] = self.sub_mean(x[3])
        x[4] = self.sub_mean(x[4])

        src_img = x[2]
        x[0] = flow_cal_backwarp(src_img, x[0], self.GMA_model)
        x[1] = flow_cal_backwarp(src_img, x[1], self.GMA_model)
        x[3] = flow_cal_backwarp(src_img, x[3], self.GMA_model)
        x[4] = flow_cal_backwarp(src_img, x[4], self.GMA_model)

        x[0] = self.feat01(self.feat0(x[0]))
        x[1] = self.feat11(self.feat1(x[1]))
        x[2] = self.feat21(self.feat2(x[2]))
        x[3] = self.feat31(self.feat3(x[3]))
        x[4] = self.feat41(self.feat4(x[4]))

        # tmp = torch.split(x[0], 64)
        # x = []
        # for k in range(len(tmp)):
        #     x.append(tmp[k].unsqueeze(0))

        out = self.dense_DRB(x)

        ## show image
        # show_PIL_image(out[:, [2, 1, 0], :, :]).show()

        out = self.conv_out(self.lrelu(self.HRconv(out[2])))

        out = self.upsample(out)  # conv shuffle lrelu

        out = self.add_mean(out)

        src_img = self.img_upsample_x4(src_img)

        out += src_img

        return out


class DRBNet9(nn.Module):
    def __init__(self, rgb_range, scale):
        super(DRBNet9, self).__init__()

        ### optical flow: GMA   ###################################
        self.GMA_model = torch.nn.DataParallel(RAFTGMA(flow_args))
        self.GMA_model.load_state_dict(torch.load(flow_args.model))
        self.GMA_model = self.GMA_model.module
        self.GMA_model.to('cuda')
        self.GMA_model.eval()
        ###########################################################

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        use_bias = True
        use_act = 'prelu'  # 'relu' 'prelu'  'lrelu' 'tanh' 'sigmoid'
        use_norm = None  # batch / instance

        feat = 256
        unit_feat = 64

        self.feat0 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat1 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat2 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat3 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat4 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)

        self.feat01 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat11 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat21 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat31 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat41 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)

        ch_reduction_ratio = 16
        block_n = 6

        module_DRB = [
            module_util.DRBModule9(inout_ch=unit_feat, ch_reduction_ratio=ch_reduction_ratio, bias=use_bias,
                                   act=use_act,
                                   block_n=k + 1)
            for k in range(block_n)]

        self.dense_DRB = nn.Sequential(*module_DRB)

        self.HRconv = nn.Conv2d(in_channels=unit_feat + block_n * unit_feat, out_channels=unit_feat, kernel_size=3,
                                stride=1, padding=1,
                                bias=use_bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_out = nn.Conv2d(in_channels=unit_feat, out_channels=scale ** 2 * 3, kernel_size=3, stride=1,
                                  padding=1,
                                  bias=use_bias)

        self.upsample = module_util.PixelShufflePack(in_channels=scale ** 2 * 3, scale=scale)

        self.img_upsample_x4 = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)

    def forward(self, x):
        x[0] = self.sub_mean(x[0])
        x[1] = self.sub_mean(x[1])
        x[2] = self.sub_mean(x[2])
        x[3] = self.sub_mean(x[3])
        x[4] = self.sub_mean(x[4])

        src_img = x[2]
        x[0] = flow_cal_backwarp(src_img, x[0], self.GMA_model)
        x[1] = flow_cal_backwarp(src_img, x[1], self.GMA_model)
        x[3] = flow_cal_backwarp(src_img, x[3], self.GMA_model)
        x[4] = flow_cal_backwarp(src_img, x[4], self.GMA_model)

        x[0] = self.feat01(self.feat0(x[0]))
        x[1] = self.feat11(self.feat1(x[1]))
        x[2] = self.feat21(self.feat2(x[2]))
        x[3] = self.feat31(self.feat3(x[3]))
        x[4] = self.feat41(self.feat4(x[4]))

        # tmp = torch.split(x[0], 64)
        # x = []
        # for k in range(len(tmp)):
        #     x.append(tmp[k].unsqueeze(0))

        out = self.dense_DRB(x)

        ## show image
        # show_PIL_image(out[:, [2, 1, 0], :, :]).show()

        out = self.conv_out(self.lrelu(self.HRconv(out[2])))

        out = self.upsample(out)  # conv shuffle lrelu

        out = self.add_mean(out)

        src_img = self.img_upsample_x4(src_img)

        out += src_img

        return out


class DRBNet9_1(nn.Module):
    def __init__(self, rgb_range, scale):
        super(DRBNet9_1, self).__init__()

        ### optical flow: GMA   ###################################
        self.GMA_model = torch.nn.DataParallel(RAFTGMA(flow_args))
        self.GMA_model.load_state_dict(torch.load(flow_args.model))
        self.GMA_model = self.GMA_model.module
        self.GMA_model.to('cuda')
        self.GMA_model.eval()
        ###########################################################

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        use_bias = True
        use_act = 'prelu'  # 'relu' 'prelu'  'lrelu' 'tanh' 'sigmoid'
        use_norm = None  # batch / instance

        feat = 256
        unit_feat = 64

        self.feat0 = module_util.ConvModule(in_ch=6, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat1 = module_util.ConvModule(in_ch=6, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat2 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat3 = module_util.ConvModule(in_ch=6, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat4 = module_util.ConvModule(in_ch=6, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)

        self.feat01 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat11 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat21 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat31 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat41 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)

        ch_reduction_ratio = 16
        block_n = 6

        module_DRB = [
            module_util.DRBModule9_1(inout_ch=unit_feat, ch_reduction_ratio=ch_reduction_ratio, bias=use_bias,
                                   act=use_act,
                                   block_n=k + 1)
            for k in range(block_n)]

        self.dense_DRB = nn.Sequential(*module_DRB)

        self.HRconv = nn.Conv2d(in_channels=unit_feat + block_n * unit_feat, out_channels=unit_feat, kernel_size=3,
                                stride=1, padding=1,
                                bias=use_bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_out = nn.Conv2d(in_channels=unit_feat, out_channels=scale ** 2 * 3, kernel_size=3, stride=1,
                                  padding=1,
                                  bias=use_bias)

        self.upsample = module_util.PixelShufflePack(in_channels=scale ** 2 * 3, scale=scale)

        self.img_upsample_x4 = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)

    def forward(self, x):
        x[0] = self.sub_mean(x[0])
        x[1] = self.sub_mean(x[1])
        x[2] = self.sub_mean(x[2])
        x[3] = self.sub_mean(x[3])
        x[4] = self.sub_mean(x[4])

        src_img = x[2]
        x[0] = flow_cal_backwarp(src_img, x[0], self.GMA_model)
        x[1] = flow_cal_backwarp(src_img, x[1], self.GMA_model)
        x[3] = flow_cal_backwarp(src_img, x[3], self.GMA_model)
        x[4] = flow_cal_backwarp(src_img, x[4], self.GMA_model)

        x[0] = self.feat01(self.feat0(torch.cat((x[0], x[2]), 1)))
        x[1] = self.feat11(self.feat1(torch.cat((x[1], x[2]), 1)))
        x[3] = self.feat31(self.feat3(torch.cat((x[3], x[2]), 1)))
        x[4] = self.feat41(self.feat4(torch.cat((x[4], x[2]), 1)))

        x[2] = self.feat21(self.feat2(x[2]))

        # tmp = torch.split(x[0], 64)
        # x = []
        # for k in range(len(tmp)):
        #     x.append(tmp[k].unsqueeze(0))

        out = self.dense_DRB(x)

        ## show image
        # show_PIL_image(out[:, [2, 1, 0], :, :]).show()

        out = self.conv_out(self.lrelu(self.HRconv(out[2])))

        out = self.upsample(out)  # conv shuffle lrelu

        out = self.add_mean(out)

        src_img = self.img_upsample_x4(src_img)

        out += src_img

        return out


class DRBNet9_2(nn.Module):
    def __init__(self, rgb_range, scale):
        super(DRBNet9_2, self).__init__()

        ### optical flow: GMA   ###################################
        self.GMA_model = torch.nn.DataParallel(RAFTGMA(flow_args))
        self.GMA_model.load_state_dict(torch.load(flow_args.model))
        self.GMA_model = self.GMA_model.module
        self.GMA_model.to('cuda')
        self.GMA_model.eval()
        ###########################################################

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        use_bias = True
        use_act = 'prelu'  # 'relu' 'prelu'  'lrelu' 'tanh' 'sigmoid'
        use_norm = None  # batch / instance

        feat = 256
        unit_feat = 64

        self.feat0 = module_util.ConvModule(in_ch=6, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat1 = module_util.ConvModule(in_ch=6, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat2 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat3 = module_util.ConvModule(in_ch=6, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat4 = module_util.ConvModule(in_ch=6, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)

        self.feat01 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat11 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat21 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat31 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat41 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)

        ch_reduction_ratio = 16
        block_n = 6

        module_DRB = [
            module_util.DRBModule9_2(inout_ch=unit_feat, ch_reduction_ratio=ch_reduction_ratio, bias=use_bias,
                                   act=use_act,
                                   block_n=k + 1)
            for k in range(block_n)]

        self.dense_DRB = nn.Sequential(*module_DRB)

        self.HRconv = nn.Conv2d(in_channels=unit_feat + block_n * unit_feat, out_channels=unit_feat, kernel_size=3,
                                stride=1, padding=1,
                                bias=use_bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_out = nn.Conv2d(in_channels=unit_feat, out_channels=scale ** 2 * 3, kernel_size=3, stride=1,
                                  padding=1,
                                  bias=use_bias)

        self.upsample = module_util.PixelShufflePack(in_channels=scale ** 2 * 3, scale=scale)

        self.img_upsample_x4 = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)

    def forward(self, x):
        x[0] = self.sub_mean(x[0])
        x[1] = self.sub_mean(x[1])
        x[2] = self.sub_mean(x[2])
        x[3] = self.sub_mean(x[3])
        x[4] = self.sub_mean(x[4])

        src_img = x[2]
        x[0] = flow_cal_backwarp(src_img, x[0], self.GMA_model)
        x[1] = flow_cal_backwarp(src_img, x[1], self.GMA_model)
        x[3] = flow_cal_backwarp(src_img, x[3], self.GMA_model)
        x[4] = flow_cal_backwarp(src_img, x[4], self.GMA_model)

        x[0] = self.feat01(self.feat0(torch.cat((x[0], x[2]), 1)))
        x[1] = self.feat11(self.feat1(torch.cat((x[1], x[2]), 1)))
        x[3] = self.feat31(self.feat3(torch.cat((x[3], x[2]), 1)))
        x[4] = self.feat41(self.feat4(torch.cat((x[4], x[2]), 1)))

        x[2] = self.feat21(self.feat2(x[2]))

        # tmp = torch.split(x[0], 64)
        # x = []
        # for k in range(len(tmp)):
        #     x.append(tmp[k].unsqueeze(0))

        out = self.dense_DRB(x)

        ## show image
        # show_PIL_image(out[:, [2, 1, 0], :, :]).show()

        out = self.conv_out(self.lrelu(self.HRconv(out[2])))

        out = self.upsample(out)  # conv shuffle lrelu

        out = self.add_mean(out)

        src_img = self.img_upsample_x4(src_img)

        out += src_img

        return out


class DRBNet9_2_b2(nn.Module):
    def __init__(self, rgb_range, scale):
        super(DRBNet9_2_b2, self).__init__()

        ### optical flow: GMA   ###################################
        self.GMA_model = torch.nn.DataParallel(RAFTGMA(flow_args))
        self.GMA_model.load_state_dict(torch.load(flow_args.model))
        self.GMA_model = self.GMA_model.module
        self.GMA_model.to('cuda')
        self.GMA_model.eval()
        ###########################################################

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        use_bias = True
        use_act = 'prelu'  # 'relu' 'prelu'  'lrelu' 'tanh' 'sigmoid'
        use_norm = None  # batch / instance

        feat = 256
        unit_feat = 64

        self.feat0 = module_util.ConvModule(in_ch=6, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat1 = module_util.ConvModule(in_ch=6, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat2 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat3 = module_util.ConvModule(in_ch=6, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat4 = module_util.ConvModule(in_ch=6, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)

        self.feat01 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat11 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat21 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat31 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat41 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)

        ch_reduction_ratio = 16
        block_n = 2

        module_DRB = [
            module_util.DRBModule9_2(inout_ch=unit_feat, ch_reduction_ratio=ch_reduction_ratio, bias=use_bias,
                                   act=use_act,
                                   block_n=k + 1)
            for k in range(block_n)]

        self.dense_DRB = nn.Sequential(*module_DRB)

        self.HRconv = nn.Conv2d(in_channels=unit_feat + block_n * unit_feat, out_channels=unit_feat, kernel_size=3,
                                stride=1, padding=1,
                                bias=use_bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_out = nn.Conv2d(in_channels=unit_feat, out_channels=scale ** 2 * 3, kernel_size=3, stride=1,
                                  padding=1,
                                  bias=use_bias)

        self.upsample = module_util.PixelShufflePack(in_channels=scale ** 2 * 3, scale=scale)

        self.img_upsample_x4 = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)

    def forward(self, x):
        x[0] = self.sub_mean(x[0])
        x[1] = self.sub_mean(x[1])
        x[2] = self.sub_mean(x[2])
        x[3] = self.sub_mean(x[3])
        x[4] = self.sub_mean(x[4])

        src_img = x[2]
        x[0] = flow_cal_backwarp(src_img, x[0], self.GMA_model)
        x[1] = flow_cal_backwarp(src_img, x[1], self.GMA_model)
        x[3] = flow_cal_backwarp(src_img, x[3], self.GMA_model)
        x[4] = flow_cal_backwarp(src_img, x[4], self.GMA_model)

        x[0] = self.feat01(self.feat0(torch.cat((x[0], x[2]), 1)))
        x[1] = self.feat11(self.feat1(torch.cat((x[1], x[2]), 1)))
        x[3] = self.feat31(self.feat3(torch.cat((x[3], x[2]), 1)))
        x[4] = self.feat41(self.feat4(torch.cat((x[4], x[2]), 1)))

        x[2] = self.feat21(self.feat2(x[2]))

        # tmp = torch.split(x[0], 64)
        # x = []
        # for k in range(len(tmp)):
        #     x.append(tmp[k].unsqueeze(0))

        out = self.dense_DRB(x)

        ## show image
        # show_PIL_image(out[:, [2, 1, 0], :, :]).show()

        out = self.conv_out(self.lrelu(self.HRconv(out[2])))

        out = self.upsample(out)  # conv shuffle lrelu

        out = self.add_mean(out)

        src_img = self.img_upsample_x4(src_img)

        out += src_img

        return out


class DRBNet9_2_b4(nn.Module):
    def __init__(self, rgb_range, scale):
        super(DRBNet9_2_b4, self).__init__()

        ### optical flow: GMA   ###################################
        self.GMA_model = torch.nn.DataParallel(RAFTGMA(flow_args))
        self.GMA_model.load_state_dict(torch.load(flow_args.model))
        self.GMA_model = self.GMA_model.module
        self.GMA_model.to('cuda')
        self.GMA_model.eval()
        ###########################################################

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        use_bias = True
        use_act = 'prelu'  # 'relu' 'prelu'  'lrelu' 'tanh' 'sigmoid'
        use_norm = None  # batch / instance

        feat = 256
        unit_feat = 64

        self.feat0 = module_util.ConvModule(in_ch=6, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat1 = module_util.ConvModule(in_ch=6, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat2 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat3 = module_util.ConvModule(in_ch=6, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)
        self.feat4 = module_util.ConvModule(in_ch=6, out_ch=feat, kernel_size=3, stride=1, padding=1,
                                            bias=use_bias, activation=use_act, norm=use_norm)

        self.feat01 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat11 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat21 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat31 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)
        self.feat41 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
                                             bias=use_bias, activation=use_act, norm=use_norm)

        ch_reduction_ratio = 16
        block_n = 4

        module_DRB = [
            module_util.DRBModule9_2(inout_ch=unit_feat, ch_reduction_ratio=ch_reduction_ratio, bias=use_bias,
                                   act=use_act,
                                   block_n=k + 1)
            for k in range(block_n)]

        self.dense_DRB = nn.Sequential(*module_DRB)

        self.HRconv = nn.Conv2d(in_channels=unit_feat + block_n * unit_feat, out_channels=unit_feat, kernel_size=3,
                                stride=1, padding=1,
                                bias=use_bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_out = nn.Conv2d(in_channels=unit_feat, out_channels=scale ** 2 * 3, kernel_size=3, stride=1,
                                  padding=1,
                                  bias=use_bias)

        self.upsample = module_util.PixelShufflePack(in_channels=scale ** 2 * 3, scale=scale)

        self.img_upsample_x4 = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)

    def forward(self, x):
        x[0] = self.sub_mean(x[0])
        x[1] = self.sub_mean(x[1])
        x[2] = self.sub_mean(x[2])
        x[3] = self.sub_mean(x[3])
        x[4] = self.sub_mean(x[4])

        src_img = x[2]
        x[0] = flow_cal_backwarp(src_img, x[0], self.GMA_model)
        x[1] = flow_cal_backwarp(src_img, x[1], self.GMA_model)
        x[3] = flow_cal_backwarp(src_img, x[3], self.GMA_model)
        x[4] = flow_cal_backwarp(src_img, x[4], self.GMA_model)

        x[0] = self.feat01(self.feat0(torch.cat((x[0], x[2]), 1)))
        x[1] = self.feat11(self.feat1(torch.cat((x[1], x[2]), 1)))
        x[3] = self.feat31(self.feat3(torch.cat((x[3], x[2]), 1)))
        x[4] = self.feat41(self.feat4(torch.cat((x[4], x[2]), 1)))

        x[2] = self.feat21(self.feat2(x[2]))

        # tmp = torch.split(x[0], 64)
        # x = []
        # for k in range(len(tmp)):
        #     x.append(tmp[k].unsqueeze(0))

        out = self.dense_DRB(x)

        ## show image
        # show_PIL_image(out[:, [2, 1, 0], :, :]).show()

        out = self.conv_out(self.lrelu(self.HRconv(out[2])))

        out = self.upsample(out)  # conv shuffle lrelu

        out = self.add_mean(out)

        src_img = self.img_upsample_x4(src_img)

        out += src_img

        return out
# class DRB_old(nn.Module):
#     def __init__(self, rgb_range, scale):
#         super(DRB_old, self).__init__()
#
#         ### optical flow: GMA   ###################################
#         self.GMA_model = torch.nn.DataParallel(RAFTGMA(flow_args))
#         self.GMA_model.load_state_dict(torch.load(flow_args.model))
#         self.GMA_model = self.GMA_model.module
#         self.GMA_model.to('cuda')
#         self.GMA_model.eval()
#         ###########################################################
#
#         # RGB mean for DIV2K
#         rgb_mean = (0.4488, 0.4371, 0.4040)
#         rgb_std = (1.0, 1.0, 1.0)
#         self.sub_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std)
#         self.add_mean = module_util.MeanShift(rgb_range, rgb_mean, rgb_std, 1)
#
#         use_bias = True
#         use_act = 'prelu'  # 'relu' 'prelu'  'lrelu' 'tanh' 'sigmoid'
#         use_norm = None  # batch / instance
#
#         feat = 256
#         unit_feat = 64
#
#         self.feat0_0 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
#                                               bias=use_bias, activation=use_act, norm=use_norm)
#         self.feat0_1 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
#                                               bias=use_bias, activation=use_act, norm=use_norm)
#         self.feat0_2 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
#                                               bias=use_bias, activation=use_act, norm=use_norm)
#         self.feat0_3 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
#                                               bias=use_bias, activation=use_act, norm=use_norm)
#         self.feat0_4 = module_util.ConvModule(in_ch=3, out_ch=feat, kernel_size=3, stride=1, padding=1,
#                                               bias=use_bias, activation=use_act, norm=use_norm)
#
#         self.feat1_0 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
#                                               bias=use_bias, activation=use_act, norm=use_norm)
#         self.feat1_1 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
#                                               bias=use_bias, activation=use_act, norm=use_norm)
#         self.feat1_2 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
#                                               bias=use_bias, activation=use_act, norm=use_norm)
#         self.feat1_3 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
#                                               bias=use_bias, activation=use_act, norm=use_norm)
#         self.feat1_4 = module_util.ConvModule(in_ch=feat, out_ch=unit_feat, kernel_size=1, stride=1, padding=0,
#                                               bias=use_bias, activation=use_act, norm=use_norm)
#
#         ch_reduction_ratio = 16
#         block_n = 6
#
#         module_DRB = [
#             module_util.DRBModule(inout_ch=unit_feat, ch_reduction_ratio=ch_reduction_ratio, bias=use_bias, act=use_act,
#                                   block_n=k + 1)
#             for k in range(block_n)]
#
#         self.dense_DRB = nn.Sequential(*module_DRB)
#
#         self.HRconv = nn.Conv2d(in_channels=unit_feat + block_n * unit_feat, out_channels=unit_feat, kernel_size=3,
#                                 stride=1, padding=1,
#                                 bias=use_bias)
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         self.conv_out = nn.Conv2d(in_channels=unit_feat, out_channels=scale ** 2 * 3, kernel_size=3, stride=1,
#                                   padding=1,
#                                   bias=use_bias)
#
#         self.upsample = module_util.PixelShufflePack(in_channels=scale ** 2 * 3, scale=scale)
#
#         self.img_upsample_x4 = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)
#
#     def forward(self, x):
#         # src_img = x[2]
#         #
#         # ref_img = x[0]
#         # x0_warp = flow_cal_backwarp(src_img, ref_img, self.GMA_model)  # back warp
#         # x0_flow = flow_cal(src_img, ref_img, self.GMA_model)  # flow
#         #
#         # ref_img = x[1]
#         # x1_warp = flow_cal_backwarp(src_img, ref_img, self.GMA_model)  # back warp
#         # x1_flow = flow_cal(src_img, ref_img, self.GMA_model)  # flow
#         #
#         # ref_img = x[3]
#         # x3_warp = flow_cal_backwarp(src_img, ref_img, self.GMA_model)  # back warp
#         # x3_flow = flow_cal(src_img, ref_img, self.GMA_model)  # flow
#         #
#         # ref_img = x[4]
#         # x4_warp = flow_cal_backwarp(src_img, ref_img, self.GMA_model)  # back warp
#         # x4_flow = flow_cal(src_img, ref_img, self.GMA_model)  # flow
#         # test = util.ShowImage()
#         # test.append(x0_flow, x1_flow, x3_flow, x4_flow)
#         # test.append(x[0], x[1], x[3], x[4])
#         # test.append(x0_warp, x1_warp, x3_warp, x4_warp)
#         # test.show(12)  # 한번에 보일 이미지 개수
#         #
#         # ref_img = x[0]
#         # x0_warp = flow_cal_backwarp(ref_img, src_img, self.GMA_model)  # back warp
#         # x0_flow = flow_cal(ref_img, src_img, self.GMA_model)  # flow
#         #
#         # ref_img = x[1]
#         # x1_warp = flow_cal_backwarp(ref_img, src_img, self.GMA_model)  # back warp
#         # x1_flow = flow_cal(ref_img, src_img, self.GMA_model)  # flow
#         #
#         # ref_img = x[3]
#         # x3_warp = flow_cal_backwarp(ref_img, src_img, self.GMA_model)  # back warp
#         # x3_flow = flow_cal(ref_img, src_img, self.GMA_model)  # flow
#         #
#         # ref_img = x[4]
#         # x4_warp = flow_cal_backwarp(ref_img, src_img, self.GMA_model)  # back warp
#         # x4_flow = flow_cal(ref_img, src_img, self.GMA_model)  # flow
#         # test = util.ShowImage()
#         # test.append(x0_flow, x1_flow, x3_flow, x4_flow)
#         # test.append(x[0], x[1], x[3], x[4])
#         # test.append(x0_warp, x1_warp, x3_warp, x4_warp)
#         # test.show(12)  # 한번에 보일 이미지 개수
#         #
#         # x0_warp = flow_cal_backwarp2(src_img, ref_img, self.GMA_model)  # back warp
#         # x0_flow = flow_cal(src_img, ref_img, self.GMA_model)  # flow
#         #
#         # ref_img = x[1]
#         # x1_warp = flow_cal_backwarp2(src_img, ref_img, self.GMA_model)  # back warp
#         # x1_flow = flow_cal(src_img, ref_img, self.GMA_model)  # flow
#         #
#         # ref_img = x[3]
#         # x3_warp = flow_cal_backwarp2(src_img, ref_img, self.GMA_model)  # back warp
#         # x3_flow = flow_cal(src_img, ref_img, self.GMA_model)  # flow
#         #
#         # ref_img = x[4]
#         # x4_warp = flow_cal_backwarp2(src_img, ref_img, self.GMA_model)  # back warp
#         # x4_flow = flow_cal(src_img, ref_img, self.GMA_model)  # flow
#         # test = util.ShowImage()
#         # test.append(x0_flow, x1_flow, x3_flow, x4_flow)
#         # test.append(x[0], x[1], x[3], x[4])
#         # test.append(x0_warp, x1_warp, x3_warp, x4_warp)
#         # test.show(12)  # 한번에 보일 이미지 개수
#         #
#         # ref_img = x[0]
#         # x0_warp = flow_cal_backwarp2(ref_img, src_img, self.GMA_model)  # back warp
#         # x0_flow = flow_cal(ref_img, src_img, self.GMA_model)  # flow
#         #
#         # ref_img = x[1]
#         # x1_warp = flow_cal_backwarp2(ref_img, src_img, self.GMA_model)  # back warp
#         # x1_flow = flow_cal(ref_img, src_img, self.GMA_model)  # flow
#         #
#         # ref_img = x[3]
#         # x3_warp = flow_cal_backwarp2(ref_img, src_img, self.GMA_model)  # back warp
#         # x3_flow = flow_cal(ref_img, src_img, self.GMA_model)  # flow
#         #
#         # ref_img = x[4]
#         # x4_warp = flow_cal_backwarp2(ref_img, src_img, self.GMA_model)  # back warp
#         # x4_flow = flow_cal(ref_img, src_img, self.GMA_model)  # flow
#         # test = util.ShowImage()
#         # test.append(x0_flow, x1_flow, x3_flow, x4_flow)
#         # test.append(x[0], x[1], x[3], x[4])
#         # test.append(x0_warp, x1_warp, x3_warp, x4_warp)
#         # test.show(12)  # 한번에 보일 이미지 개수
#
#         src_img = x[2]
#         x[0] = flow_cal_backwarp(src_img, x[0], self.GMA_model)
#         x[1] = flow_cal_backwarp(src_img, x[1], self.GMA_model)
#         x[3] = flow_cal_backwarp(src_img, x[3], self.GMA_model)
#         x[4] = flow_cal_backwarp(src_img, x[4], self.GMA_model)
#
#         x[0] = self.sub_mean(x[0])
#         x[1] = self.sub_mean(x[1])
#         x[2] = self.sub_mean(x[2])
#         x[3] = self.sub_mean(x[3])
#         x[4] = self.sub_mean(x[4])
#
#         x[0] = self.feat1_0(self.feat0_0(x[0]))
#         x[1] = self.feat1_1(self.feat0_1(x[1]))
#         x[2] = self.feat1_2(self.feat0_2(x[2]))
#         x[3] = self.feat1_3(self.feat0_3(x[3]))
#         x[4] = self.feat1_4(self.feat0_4(x[4]))
#
#
#         # x = self.feat0(torch.cat((x[0], x[1], x[2], x[3], x[4]), 1))
#         # x = self.feat1(x)
#         # x_split = []
#         # for t in range(x.size(0)):
#         #     tmp = torch.split(x[t], 64)
#         #     tmp = torch.stack(tmp)
#         #     for k in range(len(tmp)):
#         #         x_split.append(tmp[k].unsqueeze(0))
#
#         out = self.dense_DRB(x)
#
#         ## show image
#         # show_PIL_image(x[0][:, [2, 1, 0], :, :]).show()
#
#         out = self.conv_out(self.lrelu(self.HRconv(out[2])))
#
#         out = self.upsample(out)  # conv shuffle lrelu
#
#         out = self.add_mean(out)
#
#         src_img = self.img_upsample_x4(src_img)
#
#         out += src_img
#
#         return out
