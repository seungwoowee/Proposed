import torch
import torch.nn as nn
import logging
import models.modules.SRResNet_arch as SRResNet_arch
import models.modules.discriminator_vgg_arch as SRGAN_arch
import models.modules.RRDBNet_arch as RRDBNet_arch
import models.modules.DRBNet_arch as DRBNet_arch
import models.modules.RCAN_arch as RCAN_arch

logger = logging.getLogger('base')


####################
# define network
####################
#### Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'MSRResNet':
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'RRDBNet':
        netG = RRDBNet_arch.RCAN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'])
    elif which_model == 'RCAN':
        netG = RCAN_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                 nf=opt_net['nf'], nb=opt_net['nb'])
    # elif which_model == 'sft_arch':  # SFT-GAN
    #     netG = sft_arch.SFT_Net()
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG


#### Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'NLayerDiscriminator':
        netD = SRGAN_arch.NLayerDiscriminator(input_nc=opt_net['in_nc'], ndf=opt_net['nf'], n_layers=opt_net['nlayer'])
    elif which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


#### Define Network used for Perceptual Loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF


#### My model
def define_DRB(opt):
    opt_net = opt['network_DRB']
    which_model = opt_net['which_model_DRB']
    if which_model == 'DRB_mid':
        net = DRBNet_arch.DRBNet_mid()
    elif which_model == 'DRB_side_1st':
        net = DRBNet_arch.DRBNet_side_1st()
    elif which_model == 'DRB_side_2nd':
        net = DRBNet_arch.DRBNet_side_2nd()
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return net


## RCAN
def define_RCAN(opt):
    opt_net = opt['network_RCAN']
    which_model = opt_net['which_model']
    if which_model == 'RCAN':
        net = RCAN_arch.RCAN(n_resgroups=opt_net['n_resgroups'], n_resblocks=opt_net['n_resblocks'],
                             n_feats=opt_net['n_feats'], reduction=opt_net['reduction'], scale=opt_net['scale'],
                             rgb_range=opt_net['rgb_range'], n_colors=opt_net['n_colors'],
                             res_scale=opt_net['res_scale'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format('RCAN'))
    return net
