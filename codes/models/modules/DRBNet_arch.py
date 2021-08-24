# import sys
# sys.path.append('models\modules\GMA_core')

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
from codes.utils import util
from codes.data.util import read_img

from models.modules.GMA_core.GMA_network import RAFTGMA
from models.modules.GMA_core.GMA_utils.utils import InputPadder

import numpy as np
from PIL import Image
import cv2

import torchvision.transforms as transforms


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


flow_parser = argparse.ArgumentParser()
flow_parser.add_argument('--model', help="restore checkpoint",
                         default="models/modules/GMA_checkpoints/gma-kitti.pth")
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
    def __init__(self, inout_ch, kernel_size=3, bias=False, bn=False, act=nn.PReLU(), res_scale=1):
        super(ResModule, self).__init__()
        module_temp = []
        for i in range(2):
            module_temp.append(nn.Conv2d(in_channels=inout_ch, out_channels=inout_ch, kernel_size=kernel_size, stride=1,
                                         padding=(kernel_size // 2), bias=bias))
            if i == 0:
                if bn:
                    module_temp.append(nn.BatchNorm2d(inout_ch))
                module_temp.append(act)

        self.module = nn.Sequential(*module_temp)
        self.res_scale = res_scale

        self.conv_out = nn.Conv2d(in_channels=inout_ch, out_channels=inout_ch, kernel_size=1, stride=1,
                                  padding=0, bias=bias)

    def forward(self, x):
        res = self.module(x).mul(self.res_scale)
        out = self.conv_out(res + x)
        return out


class ResModule_down(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, bias=False, bn=False, act=nn.PReLU(), res_scale=1):
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


class scale_up_x2(nn.Module):
    def __init__(self, ch):
        super(scale_up_x2, self).__init__()
        self.deconv = nn.ConvTranspose2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        out = self.deconv(x, output_size=[x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2])

        return out


class CALayer(nn.Module):
    def __init__(self, inout_ch, ch_reduction):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(inout_ch, inout_ch // ch_reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(inout_ch // ch_reduction, inout_ch, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class U_shaped_Net_with_CA_dense(nn.Module):
    def __init__(self, ch, bias, bn, act, res_scale, ch_reduction):
        super(U_shaped_Net_with_CA_dense, self).__init__()
        self.resB_down1 = ResModule_down(in_ch=ch, out_ch=4 * ch, bias=bias, bn=bn, act=act, res_scale=res_scale)
        self.resB_down2 = ResModule_down(in_ch=4 * ch, out_ch=16 * ch, bias=bias, bn=bn, act=act, res_scale=res_scale)
        self.deconv3 = nn.ConvTranspose2d(16 * ch, 4 * ch, 3, stride=2, padding=1)

        self.resB4 = ResModule(inout_ch=8 * ch, bias=bias, bn=bn, act=act, res_scale=res_scale)
        self.deconv5 = nn.ConvTranspose2d(8 * ch, 2 * ch, 3, stride=2, padding=1)
        self.resB6 = ResModule(inout_ch=3 * ch, bias=bias, bn=bn, act=act, res_scale=res_scale)
        self.CA = CALayer(inout_ch=3 * ch, ch_reduction=ch_reduction)
        self.conv_out = nn.Conv2d(in_channels=4 * ch, out_channels=ch, kernel_size=3, stride=1, padding=1,
                                  bias=False)

    def forward(self, x):
        cat1 = x[0]
        x[0] = self.resB_down1(x[0])
        cat2 = x[0]
        x[0] = self.resB_down2(x[0])
        x[0] = self.deconv3(x[0], output_size=[x[0].size(0), x[0].size(1), x[0].size(2) * 2, x[0].size(3) * 2])
        x[0] = torch.cat((cat2, x[0]), 1)
        x[0] = self.resB4(x[0])
        x[0] = self.deconv5(x[0], output_size=[x[0].size(0), x[0].size(1), x[0].size(2) * 2, x[0].size(3) * 2])
        x[0] = torch.cat((cat1, x[0]), 1)
        x[0] = self.resB6(x[0])
        x[0] = self.CA(x[0])
        out = self.conv_out(torch.cat((x[0], x[1]), 1))
        return [out, x[1]]


class img_to_feat(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(img_to_feat, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        out = self.features(x)
        return out


### final model
class DRBNet_mid(nn.Module):
    def __init__(self):
        super(DRBNet_mid, self).__init__()
        use_act = nn.PReLU()
        use_bias = False
        use_bn = False

        res_scale = 0.1
        block_n = 2

        ch = 64
        ch_reduction = 16
        self.in_feat1 = img_to_feat(in_ch=9, out_ch=ch)
        self.in_feat2 = img_to_feat(in_ch=9, out_ch=ch)
        self.in_feat3 = img_to_feat(in_ch=9, out_ch=ch)

        module_U_net1 = [
            U_shaped_Net_with_CA_dense(ch=ch, bias=use_bias, bn=use_bn, act=use_act, res_scale=res_scale,
                                       ch_reduction=ch_reduction)
            for _ in range(block_n)]
        self.U_net1 = nn.Sequential(*module_U_net1)

        module_U_net2 = [
            U_shaped_Net_with_CA_dense(ch=ch, bias=use_bias, bn=use_bn, act=use_act, res_scale=res_scale,
                                       ch_reduction=ch_reduction)
            for _ in range(block_n)]
        self.U_net2 = nn.Sequential(*module_U_net2)

        module_U_net3 = [
            U_shaped_Net_with_CA_dense(ch=ch, bias=use_bias, bn=use_bn, act=use_act, res_scale=res_scale,
                                       ch_reduction=ch_reduction)
            for _ in range(block_n)]
        self.U_net3 = nn.Sequential(*module_U_net3)

        self.scale_up1 = scale_up_x2(ch)
        self.scale_up2 = scale_up_x2(ch)
        self.scale_up3 = scale_up_x2(ch)

        self.in_feat4 = img_to_feat(in_ch=9, out_ch=ch)
        self.conv_1d = nn.Conv2d(in_channels=4 * ch, out_channels=2 * ch, kernel_size=1, stride=1, padding=0,
                                 bias=False)
        ch = 2 * ch
        ch_reduction = 32
        module_U_net4 = [
            U_shaped_Net_with_CA_dense(ch=ch, bias=use_bias, bn=use_bn, act=use_act, res_scale=res_scale,
                                       ch_reduction=ch_reduction)
            for _ in range(block_n)]
        self.U_net4 = nn.Sequential(*module_U_net4)

        self.scale_up4 = scale_up_x2(ch)

        self.conv_out = nn.Conv2d(in_channels=ch, out_channels=3, kernel_size=3, stride=1, padding=1,
                                  bias=False)

        self.initialize_weights()

        self.GMA_model = torch.nn.DataParallel(RAFTGMA(flow_args))
        self.GMA_model.load_state_dict(torch.load(flow_args.model))
        self.GMA_model = self.GMA_model.module
        self.GMA_model.to('cuda')
        self.GMA_model.eval()
        # print(f"Loaded checkpoint at {flow_args.model}")

    def forward(self, x):
        path = 'D:/REDS/ori.png'
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img.astype(np.float32) / 255.
        img = img[:, :, [2, 1, 0]]
        img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
        img = img[None].to('cuda') * 255
        # image1 = img[:, :, 73:73 + 64, 16:16 + 64]
        # image1 = x[0][0:8, :, :, :]
        image1 = x[0]

        path = 'D:/REDS/ori_3.png'
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img.astype(np.float32) / 255.
        img = img[:, :, [2, 1, 0]]
        img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
        img = img[None].to('cuda') * 255
        # image2 = img[:, :, 73:73 + 64, 16:16 + 64]
        # image2 = x[2][0:8, :, :, :]
        image2 = x[2]

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        flow_low, flow_up = self.GMA_model(image1, image2, iters=12, test_mode=True)

        ####
        W, H = image1.shape[-1], image1.shape[-2]
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))

        self.gridX = torch.tensor(gridX, requires_grad=False, device='cuda')
        self.gridY = torch.tensor(gridY, requires_grad=False, device='cuda')

        u = flow_up[:, 0, :, :]
        v = flow_up[:, 1, :, :]
        grid_x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        grid_y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        # grid_x = torch.clamp(grid_x, min=-W, max=W)
        # grid_y = torch.clamp(grid_y, min=-H, max=H)
        # range -1 to 1
        grid_x = 2 * (grid_x / W - 0.5)
        grid_y = 2 * (grid_y / H - 0.5)

        # stacking X and Y
        grid = torch.stack((grid_x, grid_y), dim=3)
        # Sample pixels using bilinear interpolation.
        warped_images = torch.nn.functional.grid_sample(image1, grid)

        show_PIL_image(warped_images[:, [2, 1, 0], :, :]).show()

        # _, flow_tmp = self.GMA_model(x[0], x[2], iters=12, test_mode=True)
        # flow.append(flow_tmp)
        # _, flow_tmp = self.GMA_model(x[1], x[2], iters=12, test_mode=True)
        # flow.append(flow_tmp)
        # _, flow_tmp = self.GMA_model(x[3], x[2], iters=12, test_mode=True)
        # flow.append(flow_tmp)
        # _, flow_tmp = self.GMA_model(x[4], x[2], iters=12, test_mode=True)
        # flow.append(flow_tmp)
        #
        # x1 = torch.cat((x[0], x[1], x[2]), 1)
        # x2 = torch.cat((x[1], x[2], x[3]), 1)
        # x3 = torch.cat((x[2], x[3], x[4]), 1)
        #
        # x1 = self.in_feat1(x1)
        # x2 = self.in_feat2(x2)
        # x3 = self.in_feat3(x3)
        #
        # U1 = self.U_net1([x1, x2])
        # U2 = self.U_net2([x2, x2])
        # U3 = self.U_net3([x3, x2])
        #
        # x1 = self.scale_up1(U1[0])
        # x2 = self.scale_up2(U2[0])
        # x3 = self.scale_up3(U3[0])
        #
        # scale_up_x1 = F.interpolate(x[1], scale_factor=2, mode='bilinear', align_corners=True)
        # scale_up_x2 = F.interpolate(x[2], scale_factor=2, mode='bilinear', align_corners=True)
        # scale_up_x3 = F.interpolate(x[3], scale_factor=2, mode='bilinear', align_corners=True)
        #
        # scale_up_x = self.in_feat4(torch.cat((scale_up_x1, scale_up_x2, scale_up_x3), 1))
        # out = self.conv_1d(torch.cat((scale_up_x, x1, x2, x3), 1))
        # out = self.U_net4([out, out])
        # # out = self.U_net4(torch.cat((scale_up_x, x1, x2, x3), 1))
        # out = self.scale_up4(out[0])
        # out = self.conv_out(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


class DRBNet_side_2nd(nn.Module):
    def __init__(self):
        super(DRBNet_side_2nd, self).__init__()
        use_act = nn.PReLU()
        use_bias = False
        use_bn = False

        res_scale = 0.1
        block_n = 2

        ch = 64
        ch_reduction = 16
        self.in_feat1 = img_to_feat(in_ch=3, out_ch=ch)
        self.in_feat2 = img_to_feat(in_ch=9, out_ch=ch)
        self.in_feat3 = img_to_feat(in_ch=3, out_ch=ch)

        module_U_net1 = [
            U_shaped_Net_with_CA_dense(ch=ch, bias=use_bias, bn=use_bn, act=use_act, res_scale=res_scale,
                                       ch_reduction=ch_reduction)
            for _ in range(block_n)]
        self.U_net1 = nn.Sequential(*module_U_net1)

        module_U_net2 = [
            U_shaped_Net_with_CA_dense(ch=ch, bias=use_bias, bn=use_bn, act=use_act, res_scale=res_scale,
                                       ch_reduction=ch_reduction)
            for _ in range(block_n)]
        self.U_net2 = nn.Sequential(*module_U_net2)

        module_U_net3 = [
            U_shaped_Net_with_CA_dense(ch=ch, bias=use_bias, bn=use_bn, act=use_act, res_scale=res_scale,
                                       ch_reduction=ch_reduction)
            for _ in range(block_n)]
        self.U_net3 = nn.Sequential(*module_U_net3)

        self.scale_up1 = scale_up_x2(ch)
        self.scale_up2 = scale_up_x2(ch)
        self.scale_up3 = scale_up_x2(ch)

        self.in_feat4 = img_to_feat(in_ch=9, out_ch=ch)
        self.conv_1d = nn.Conv2d(in_channels=4 * ch, out_channels=2 * ch, kernel_size=1, stride=1, padding=0,
                                 bias=False)

        ch = 2 * ch
        ch_reduction = 32
        module_U_net4 = [
            U_shaped_Net_with_CA_dense(ch=ch, bias=use_bias, bn=use_bn, act=use_act, res_scale=res_scale,
                                       ch_reduction=ch_reduction)
            for _ in range(block_n)]
        self.U_net4 = nn.Sequential(*module_U_net4)

        self.scale_up4 = scale_up_x2(ch)

        self.conv_out = nn.Conv2d(in_channels=ch, out_channels=3, kernel_size=3, stride=1, padding=1,
                                  bias=False)

        self.initialize_weights()

    def forward(self, x):
        x1 = x[0]
        x2 = torch.cat((x[0], x[1], x[2]), 1)
        x3 = x[2]

        x1 = self.in_feat1(x1)
        x2 = self.in_feat2(x2)
        x3 = self.in_feat3(x3)

        U1 = self.U_net1([x1, x2])
        U2 = self.U_net2([x2, x2])
        U3 = self.U_net3([x3, x2])

        x1 = self.scale_up1(U1[0])
        x2 = self.scale_up2(U2[0])
        x3 = self.scale_up3(U3[0])

        scale_up_x1 = F.interpolate(x[0], scale_factor=2, mode='bilinear', align_corners=True)
        scale_up_x2 = F.interpolate(x[1], scale_factor=2, mode='bilinear', align_corners=True)
        scale_up_x3 = F.interpolate(x[2], scale_factor=2, mode='bilinear', align_corners=True)

        scale_up_x = self.in_feat4(torch.cat((scale_up_x1, scale_up_x2, scale_up_x3), 1))
        out = self.conv_1d(torch.cat((scale_up_x, x1, x2, x3), 1))
        out = self.U_net4([out, out])
        # out = self.U_net4(torch.cat((scale_up_x, x1, x2, x3), 1))
        out = self.scale_up4(out[0])
        out = self.conv_out(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


class DRBNet_side_1st(nn.Module):
    def __init__(self):
        super(DRBNet_side_1st, self).__init__()
        use_act = nn.PReLU()
        use_bias = False
        use_bn = False

        res_scale = 0.1
        block_n = 2

        ch = 64
        ch_reduction = 16
        self.in_feat1 = img_to_feat(in_ch=9, out_ch=ch)
        self.in_feat2 = img_to_feat(in_ch=3, out_ch=ch)
        self.in_feat3 = img_to_feat(in_ch=3, out_ch=ch)

        module_U_net1 = [
            U_shaped_Net_with_CA_dense(ch=ch, bias=use_bias, bn=use_bn, act=use_act, res_scale=res_scale,
                                       ch_reduction=ch_reduction)
            for _ in range(block_n)]
        self.U_net1 = nn.Sequential(*module_U_net1)

        module_U_net2 = [
            U_shaped_Net_with_CA_dense(ch=ch, bias=use_bias, bn=use_bn, act=use_act, res_scale=res_scale,
                                       ch_reduction=ch_reduction)
            for _ in range(block_n)]
        self.U_net2 = nn.Sequential(*module_U_net2)

        module_U_net3 = [
            U_shaped_Net_with_CA_dense(ch=ch, bias=use_bias, bn=use_bn, act=use_act, res_scale=res_scale,
                                       ch_reduction=ch_reduction)
            for _ in range(block_n)]
        self.U_net3 = nn.Sequential(*module_U_net3)

        self.scale_up1 = scale_up_x2(ch)
        self.scale_up2 = scale_up_x2(ch)
        self.scale_up3 = scale_up_x2(ch)

        self.in_feat4 = img_to_feat(in_ch=9, out_ch=ch)
        self.conv_1d = nn.Conv2d(in_channels=4 * ch, out_channels=2 * ch, kernel_size=1, stride=1, padding=0,
                                 bias=False)
        ch = 2 * ch
        ch_reduction = 32
        module_U_net4 = [
            U_shaped_Net_with_CA_dense(ch=ch, bias=use_bias, bn=use_bn, act=use_act, res_scale=res_scale,
                                       ch_reduction=ch_reduction)
            for _ in range(block_n)]
        self.U_net4 = nn.Sequential(*module_U_net4)

        self.scale_up4 = scale_up_x2(ch)

        self.conv_out = nn.Conv2d(in_channels=ch, out_channels=3, kernel_size=3, stride=1, padding=1,
                                  bias=False)

        self.initialize_weights()

    def forward(self, x):
        x1 = torch.cat((x[0], x[1], x[2]), 1)
        x2 = x[1]
        x3 = x[2]

        x1 = self.in_feat1(x1)
        x2 = self.in_feat2(x2)
        x3 = self.in_feat3(x3)

        U1 = self.U_net1([x1, x1])
        U2 = self.U_net2([x2, x1])
        U3 = self.U_net3([x3, x1])

        x1 = self.scale_up1(U1[0])
        x2 = self.scale_up2(U2[0])
        x3 = self.scale_up3(U3[0])

        scale_up_x1 = F.interpolate(x[0], scale_factor=2, mode='bilinear', align_corners=True)
        scale_up_x2 = F.interpolate(x[1], scale_factor=2, mode='bilinear', align_corners=True)
        scale_up_x3 = F.interpolate(x[2], scale_factor=2, mode='bilinear', align_corners=True)

        scale_up_x = self.in_feat4(torch.cat((scale_up_x1, scale_up_x2, scale_up_x3), 1))
        out = self.conv_1d(torch.cat((scale_up_x, x1, x2, x3), 1))
        out = self.U_net4([out, out])
        # out = self.U_net4(torch.cat((scale_up_x, x1, x2, x3), 1))
        out = self.scale_up4(out[0])
        out = self.conv_out(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
