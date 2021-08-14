import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
from data.data_loader import noiseDataset


class DRBDataset(data.Dataset):
    def __init__(self, opt):
        super(DRBDataset, self).__init__()
        self.opt = opt
        self.position = self.opt['position']
        self.data_type = self.opt['data_type']
        self.ref_frame_num = self.opt['ref_frame_num']
        self.paths_LR, self.paths_GT = None, None
        self.sizes_LR, self.sizes_GT = None, None
        self.LR_env, self.GT_env = None, None  # environment for lmdb
        if self.data_type == 'img':
            self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
            self.paths_LR, self.sizes_LR = util.get_image_paths(self.data_type, opt['dataroot_LR'])
        elif self.data_type == 'seq':
            self.paths_GT = util.get_sequence_paths(opt['dataroot_GT'])
            self.paths_LR = util.get_sequence_paths(opt['dataroot_LR'])
        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LR and self.paths_GT:
            assert len(self.paths_LR) == len(
                self.paths_GT
            ), 'GT and LR datasets have different number of images - {}, {}.'.format(
                len(self.paths_LR), len(self.paths_GT))
        self.random_scale_list = [1]
        # print(opt['aug'])
        if self.opt['phase'] == 'train':
            if opt['aug'] and 'noise' in opt['aug']:
                self.noises = noiseDataset(opt['noise_data'], opt['GT_size'] / opt['scale'])

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LR_env = lmdb.open(self.opt['dataroot_LR'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):
        if self.data_type == 'lmdb':
            if (self.GT_env is None) or (self.LR_env is None):
                self._init_lmdb()

        GT_path, LR_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        targetSequence = index // (len(self.paths_GT[0]) - self.ref_frame_num + 1)
        # randomly select target frame & get LR_paths   #: ref, o: target
        if self.position == 'mid':
            # targetFrame = random.randint(2, len(self.paths_GT[0]) - 3)  # when used 5 frames
            targetFrame = index % (len(self.paths_GT[0]) - self.ref_frame_num + 1) + (
                    self.ref_frame_num // 2)  # when used 5 frames
            LR_path = self.paths_LR[targetSequence][targetFrame - 2:targetFrame + 3]  # ##o##
        elif self.position == 'side_1st':
            targetFrame = index % len(self.paths_GT[0])

            if targetFrame < 2:  # using right two frames
                LR_path = self.paths_LR[targetSequence][targetFrame:targetFrame + 3]  # o##
            elif targetFrame > (len(self.paths_GT[0]) - 3):  # using left two frames
                LR_path = self.paths_LR[targetSequence][targetFrame - 2:targetFrame + 1]  # ##o
                LR_path.reverse()
            else:
                if random.randint(0, 1):  # using right two frames
                    LR_path = self.paths_LR[targetSequence][targetFrame:targetFrame + 3]  # o##
                else:  # using left two frames
                    LR_path = self.paths_LR[targetSequence][targetFrame - 2:targetFrame + 1]  # ##o
                    LR_path.reverse()
        elif self.position == 'side_2nd':
            targetFrame = (index % (len(self.paths_GT[0]) - 2)) + 1  # when used 3 frames
            LR_path = self.paths_LR[targetSequence][targetFrame - 1:targetFrame + 2]  # #o#
        else:
            assert None, 'Error: wrong position.'

        # get GT image
        GT_path = self.paths_GT[targetSequence][targetFrame]
        if self.data_type == 'lmdb':
            resolution = [int(s) for s in self.sizes_GT[index].split('_')]
        else:
            resolution = None
        img_GT = util.read_img(self.GT_env, GT_path, resolution)
        # modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            img_GT = util.modcrop(img_GT, scale)
        # change color space if necessary
        if self.opt['color']:
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]

        # get LR image
        img_LR = util.read_seq(LR_path)
        # if self.paths_LR:
        #     LR_path = self.paths_LR[index]
        #     if self.data_type == 'lmdb':
        #         resolution = [int(s) for s in self.sizes_LR[index].split('_')]
        #     else:
        #         resolution = None
        #     img_LR = util.read_img(self.LR_env, LR_path, resolution)
        #
        # else:  # down-sampling on-the-fly
        #     # randomly scale during training
        #     if self.opt['phase'] == 'train':
        #         random_scale = random.choice(self.random_scale_list)
        #         H_s, W_s, _ = img_GT.shape
        #
        #         def _mod(n, random_scale, scale, thres):
        #             rlt = int(n * random_scale)
        #             rlt = (rlt // scale) * scale
        #             return thres if rlt < thres else rlt
        #
        #         H_s = _mod(H_s, random_scale, scale, GT_size)
        #         W_s = _mod(W_s, random_scale, scale, GT_size)
        #         img_GT = cv2.resize(np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
        #         # force to 3 channels
        #         if img_GT.ndim == 2:
        #             img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)
        #
        #     H, W, _ = img_GT.shape
        #     # using matlab imresize
        #     img_LR = util.imresize_np(img_GT, 1 / scale, True)
        #     if img_LR.ndim == 2:
        #         img_LR = np.expand_dims(img_LR, axis=2)

        if self.opt['phase'] == 'train':

            # if the image size is too small
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
                                    interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LR = util.imresize_np(img_GT, 1 / scale, True)
                if img_LR.ndim == 2:
                    img_LR = np.expand_dims(img_LR, axis=2)

            H, W, C = img_LR[0].shape
            LR_size = GT_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))

            for idx, LR in enumerate(img_LR):
                img_LR[idx] = LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            img_LR, img_GT = util.augment([img_LR, img_GT], self.opt['use_flip'],
                                          self.opt['use_rot'])

        # change color space if necessary
        if self.opt['color']:
            img_LR = util.channel_convert(C, self.opt['color'],
                                          [img_LR])[0]  # TODO during val no definition

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
            for idx, LR in enumerate(img_LR):
                img_LR[idx] = LR[:, :, [2, 1, 0]]
                img_LR[idx] = torch.from_numpy(np.ascontiguousarray(np.transpose(LR, (2, 0, 1)))).float()
        if self.opt['phase'] == 'train':
            # add noise to LR during train
            if self.opt['aug'] and 'noise' in self.opt['aug']:
                noise = self.noises[np.random.randint(0, len(self.noises))]
                img_LR = torch.clamp(img_LR + noise, 0, 1)

        if LR_path is None:
            LR_path = GT_path
        return {'LR': img_LR, 'GT': img_GT, 'LR_path': LR_path, 'GT_path': GT_path}

    def __len__(self):
        if self.opt['phase'] == 'train' or self.opt['phase'] == 'val':
            return len(self.paths_GT) * (len(self.paths_GT[0]) - self.ref_frame_num + 1)
        elif self.opt['phase'] == 'test':
            return len(self.paths_LR)
