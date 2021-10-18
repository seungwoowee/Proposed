import os
import sys
import time
import math
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
import cv2
import torch
from torchvision.utils import make_grid
from shutil import get_terminal_size
import torch.nn as nn

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper




# from models.acceleration import *
# for ShowImage Class
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

####################
# functions for DRB
####################
def initialize_weights(net_list, scale=1):
    if not isinstance(net_list, list):
        net_list = [net_list]
    for net in net_list:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    # nn.init.constant_(m.bias, 0)
                    m.bias.data.zero_()

            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    # nn.init.constant_(m.bias, 0)
                    m.bias.data.zero_()

####################
# miscellaneous
####################


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


####################
# image convert
####################


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)


####################
# metric
####################


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()

class ShowImage:
    def __init__(self):
        self.f = plt.figure(figsize=(17, 7))                                     # figure axis setup
        self.f.subplots_adjust(bottom=0.15)
        self.data = []
        self.im = []


    def put_data(self, im, I, t=1):     # 칸plt  , 이미지
        with torch.no_grad():
            if type(I) is list and len(I)==3 and I[1] is not None: # 함수[0] + 2args[1][2]
                if I[1].shape[1]==3:  im.set_data(  I[0](I[1],I[2]*t     )[0][0,:,:,:].permute([1,2,0]).cpu().numpy()  );return   # 3채널이미지일땐 그냥 plot되지만
                if I[1].shape[1]==2:  im.set_data(  self.ycbcr2rgb(  I[0](I[1],I[2]*t     )[0][0,:,:,:]  )  )           ;return   # 2채널플로우일때 다시한번 3채널로만들어줘야함
            if type(I) is list and len(I)==4 and I[1] is not None: # 함수[0] + 3args[1][2][3]     (함수의 아웃풋이 2개 튜플임)
                if I[1].shape[1]==3:  im.set_data(  I[0](I[1],I[2]*t,I[3])[0][0,:,:,:].permute([1,2,0]).cpu().numpy()  );return   # 3채널이미지일땐 그냥 plot되지만
                if I[1].shape[1]==2:  im.set_data(  self.ycbcr2rgb(  I[0](I[1],I[2]*t,I[3])[0][0,:,:,:]  )  )           ;return   # 2채널플로우일때 다시한번 3채널로만들어줘야함
            if I is None:                        im.set_data(  np.full((128,128,3),0.1)  )                              ;return   # None
            if type(I[0]) is not torch.Tensor:   im.set_data(  np.full((128,128,3),0.3)  )                              ;return
            if I[0].shape[0] == 2: im.set_data(  self.ycbcr2rgb( I )  )                                                 ;return   # flow 맵
            if I[0].shape[0] == 3: im.set_data(  I[:, [2, 1, 0], :, :][0].permute([1,2,0]).cpu().numpy() )                            ;return   # 3cb rgb타입
            if I[0].shape[0] == 1: im.set_data(  I[0,:,:,:].repeat(3,1,1).permute([1,2,0]).cpu().numpy())               ;return  # mask

            im.set_data(np.full((256, 256, 3), 0.22))      # 모두 해당안할시

    def append(self, *args):    # ShowImage 디버그는 배치사이즈 1일때만 동작한다는걸 명심
        with torch.no_grad():
            for I in args:
                self.data.append(I)

    def update_depth(self, page):   # update the figure with a change on the slider
        with torch.no_grad():
            if self.slider_depth.val != round(self.slider_depth.val): self.slider_depth.set_val(float(round(self.slider_depth.val))) ; plt.pause(0.4)  # 정수로 고정
            idx = int(self.slider_depth.val)
            t = (self.slider_deptht.val)
            for nn in range(len(self.im)):      # 8개 화면에 대한 for
                if len(self.data) > idx*len(self.im)+nn:   self.put_data( self.im[nn], self.data[idx*len(self.im)+nn], t)
                else: self.put_data( self.im[nn], None)         # 아웃오브 인덱스는 None 이미지

    def show(self, n=1):
        with torch.no_grad():
            r = 1; c = n;
            if n > 4: r=n//4; c=4;
            self.slider = plt.axes([0.44, 0.005, 0.09, 0.02])            # setup a slider axis and the Slider
            self.slidert = plt.axes([0.24, 0.005, 0.09, 0.02])
            self.slider_depth = Slider(self.slider, 'cnt:', 0, (math.ceil(len(self.data)/n))-1, valinit=0)
            self.slider_deptht = Slider(self.slidert, 't:', 0, 1, valinit=1)
            self.slider_depth.on_changed(self.update_depth)
            self.slider_deptht.on_changed(self.update_depth)
            for nn in range(0, n): self.im.append( self.f.add_subplot(r, c, nn+1).imshow( np.full((128,128,3),0.7), interpolation='nearest') )
            self.update_depth(0)
            plt.tight_layout()
            plt.show()

    def ycbcr2rgb(self, I):
        rgb = torch.cat(  (torch.Tensor(I[0,:,:,:].squeeze()[1:,:,:].shape).fill_(0.72).cuda(),  I[0,:,:,:].squeeze())  ,0).permute([1,2,0]).cpu().numpy()
        xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
        rgb[:, :, [1, 2]] /= 32        # uv 범위는 +- 128이니까 128*8 범위만큼커버
        return rgb.dot(xform.T)

    def ycbcr2rgbt(self, I):
        I = torch.cat((torch.Tensor(I[:, 1:, :].shape).fill_(0.99).cuda(), I[:, :, :, :]), 1)
        y: torch.Tensor =  I[..., 0, :, :]
        cb: torch.Tensor = I[..., 1, :, :]
        cr: torch.Tensor = I[..., 2, :, :]
        delta: float = 0.5
        cb_shifted: torch.Tensor = cb - delta
        cr_shifted: torch.Tensor = cr - delta
        r: torch.Tensor = y + 1.403 * cr_shifted
        g: torch.Tensor = y - 0.714 * cr_shifted - 0.344 * cb_shifted
        b: torch.Tensor = y + 1.773 * cb_shifted
        return torch.stack([r, g, b], -3)
