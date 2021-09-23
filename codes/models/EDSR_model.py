import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
from .base_model_EDSR import BaseModel
import torch.optim.lr_scheduler as lrs

logger = logging.getLogger('base')


class EDSRModel(BaseModel):
    def __init__(self, opt):
        super(EDSRModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.net = networks.define_EDSR(opt).to(self.device)
        if opt['dist']:
            self.net = DistributedDataParallel(self.net, device_ids=[torch.cuda.current_device()])
        else:
            self.net = DataParallel(self.net)

        if self.is_train:
            self.net.train()
            # pixel loss
            self.cri_pix = nn.L1Loss().to(self.device)
            self.l_pix_w = train_opt['pixel_weight']

            # optimizers
            # EDSR
            wd = train_opt['weight_decay'] if train_opt['weight_decay'] else 0
            optim_params = []
            for k, v in self.net.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer = torch.optim.Adam(optim_params, lr=train_opt['lr'],
                                              weight_decay=wd,
                                              betas=(train_opt['beta1'], train_opt['beta2']),
                                              eps=train_opt['epsilon'])
            self.optimizers.append(self.optimizer)

            # schedulers
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lrs.StepLR(
                        optimizer,
                        step_size=train_opt['lr_decay'],
                        gamma=train_opt['gamma']
                    ))

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
        self.optimizer.zero_grad()
        self.SR_data = self.net(self.LQ_data)

        l_g_pix = self.l_pix_w * self.cri_pix(self.SR_data, self.GT_data)   # GT * 255

        l_g_pix.backward()
        self.optimizer.step()

        # set log
        self.log_dict['l_pix'] = l_g_pix.item()

    def test(self):
        self.net.eval()
        with torch.no_grad():
            self.SR_data = self.net(self.LQ_data)
        self.net.train()

    def test_x8(self):
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.net.eval()

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
            sr_list = [self.net(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        self.SR_data = output_cat.mean(dim=0, keepdim=True)
        self.net.train()

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
        s, n = self.get_network_description(self.net)
        if isinstance(self.net, nn.DataParallel) or isinstance(self.net, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.net.__class__.__name__,
                                             self.net.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.net.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path = self.opt['path']['pretrain_model']
        if load_path is not None:
            logger.info('Loading model for [{:s}] ...'.format(load_path))
            self.load_network(load_path, self.net, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.net, 'G', iter_label)
