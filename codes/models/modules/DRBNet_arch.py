# import sys
# sys.path.append('models\modules\GMA_core')

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from utils import util  # 디버그용

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
class DRBNet(nn.Module):
    def __init__(self, rgb_range, scale):
        super(DRBNet, self).__init__()

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
        self.feat1 = module_util.ConvModule(in_ch=feat * 5, out_ch=unit_feat * 5, kernel_size=1, stride=1,
                                            padding=0,
                                            bias=use_bias, activation=use_act, norm=use_norm)

        ch_reduction_ratio = 16
        block_n = 2

        module_DRB = [
            module_util.DRBModule(inout_ch=unit_feat, ch_reduction_ratio=ch_reduction_ratio, bias=use_bias,
                                  act=use_act, block_n=k + 1)
            for k in range(block_n)]

        self.dense_DRB = nn.Sequential(*module_DRB)
        self.recon_trunk = module_util.ResidualBlock_noBN(nf=unit_feat + block_n * unit_feat)

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
        out = self.recon_trunk(out[2])

        out = self.conv_out(self.lrelu(self.HRconv(out)))

        out = self.upsample(out)  # conv shuffle lrelu

        out = self.add_mean(out)

        src_img = self.img_upsample_x4(src_img)

        out += src_img

        return out


class DRBNet_tmp(nn.Module):
    def __init__(self, rgb_range, scale):
        super(DRBNet_tmp, self).__init__()

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
        self.feat1 = module_util.ConvModule(in_ch=feat * 5, out_ch=unit_feat * 5, kernel_size=1, stride=1,
                                            padding=0,
                                            bias=use_bias, activation=use_act, norm=use_norm)

        ch_reduction_ratio = 16
        block_n = 2

        module_DRB = [
            module_util.DRBModule(inout_ch=unit_feat, ch_reduction_ratio=ch_reduction_ratio, bias=use_bias,
                                  act=use_act, block_n=k + 1)
            for k in range(block_n)]

        self.dense_DRB = nn.Sequential(*module_DRB)
        self.recon_trunk = module_util.ResidualBlock_noBN(nf=unit_feat + block_n * unit_feat)

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
        # x[0] = self.sub_mean(x[0])
        # x[1] = self.sub_mean(x[1])
        # x[2] = self.sub_mean(x[2])
        # x[3] = self.sub_mean(x[3])
        # x[4] = self.sub_mean(x[4])

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
        out = self.recon_trunk(out[2])

        out = self.conv_out(self.lrelu(self.HRconv(out)))

        out = self.upsample(out)  # conv shuffle lrelu

        out = self.add_mean(out)

        src_img = self.img_upsample_x4(src_img)

        out += src_img

        return out
