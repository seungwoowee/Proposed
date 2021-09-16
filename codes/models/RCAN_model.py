import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import CharbonnierLoss
import torch.optim.lr_scheduler as lrs

logger = logging.getLogger('base')


class RCANModel(BaseModel):
    def __init__(self, opt):
        super(RCANModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netRCAN = networks.define_RCAN(opt).to(self.device)
        if opt['dist']:
            self.netRCAN = DistributedDataParallel(self.netRCAN, device_ids=[torch.cuda.current_device()])
        else:
            self.netRCAN = DataParallel(self.netRCAN)

        if self.is_train:
            self.netRCAN.train()
            # pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                elif l_pix_type == 'cb':
                    self.cri_pix = CharbonnierLoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            # feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                elif l_fea_type == 'cb':
                    self.cri_fea = CharbonnierLoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None

            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)
                if opt['dist']:
                    self.netF = DistributedDataParallel(self.netF,
                                                        device_ids=[torch.cuda.current_device()])
                else:
                    self.netF = DataParallel(self.netF)

            # D_update_ratio and D_init_iters
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            # optimizers
            # RCAN
            wd_RCAN = train_opt['weight_decay_RCAN'] if train_opt['weight_decay_RCAN'] else 0
            optim_params = []
            for k, v in self.netRCAN.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_RCAN = torch.optim.Adam(optim_params, lr=train_opt['lr_RCAN'],
                                                   weight_decay=wd_RCAN,
                                                   betas=(train_opt['beta1_RCAN'], train_opt['beta2_RCAN']),
                                                   eps=train_opt['epsilon'])
            self.optimizers.append(self.optimizer_RCAN)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            elif train_opt['lr_scheme'] == 'step_RCAN':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lrs.StepLR(
                            optimizer,
                            step_size=train_opt['lr_decay'],
                            gamma=train_opt['gamma']
                        ))

            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        # print network
        self.print_network()
        self.load()

    def feed_data(self, data, need_GT=True):
        self.LQ_data, self.GT_data = [], []
        for idx, LQ in enumerate(data['LQ']):  # LQ
            self.LQ_data.append([])
            self.LQ_data[idx] = LQ.to(self.device)
        if need_GT:
            self.GT_data = data['GT'].to(self.device)  # GT

    def optimize_parameters(self, step):
        self.optimizer_RCAN.zero_grad()
        self.SR_data = self.netRCAN(self.LQ_data)

        l_total_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.SR_data, self.GT_data)
                l_total_total += l_g_pix
            if self.cri_fea:  # feature loss
                real_fea = self.netF(self.GT_data).detach()
                fake_fea = self.netF(self.SR_data)
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_total_total += l_g_fea
        l_total_total.backward()
        self.optimizer_RCAN.step()

        # set log
        self.log_dict['l_pix'] = l_total_total.item()

    def test(self):
        self.netRCAN.eval()
        with torch.no_grad():
            self.SR_data = self.netRCAN(self.LQ_data)
        self.netRCAN.train()

    def test_x8(self):
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.netRCAN.eval()

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [self.LQ_data]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        with torch.no_grad():
            sr_list = [self.netRCAN(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        self.SR_data = output_cat.mean(dim=0, keepdim=True)
        self.netRCAN.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.LQ_data[len(self.LQ_data) // 2].detach()[0].float().cpu()
        out_dict['SR'] = self.SR_data.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.GT_data.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netRCAN)
        if isinstance(self.netRCAN, nn.DataParallel) or isinstance(self.netRCAN, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netRCAN.__class__.__name__,
                                             self.netRCAN.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netRCAN.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network RCAN structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_RCAN = self.opt['path']['pretrain_model']
        if load_path_RCAN is not None:
            logger.info('Loading model for RCAN [{:s}] ...'.format(load_path_RCAN))
            self.load_network(load_path_RCAN, self.netRCAN, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netRCAN, 'G', iter_label)
