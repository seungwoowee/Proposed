import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    if model == 'DRB':
        from .DRB_model import DRBModel as M
    elif model == 'RCAN':
        from .RCAN_model import RCANModel as M
    elif model == 'EDSR':
        from .EDSR_model import EDSRModel as M
    elif model == 'DBPN':
        from .DBPN_model import DBPNModel as M

    elif model == 'sr':
        from .SR_model import SRModel as M
    elif model == 'srgan':
        from .SRGAN_model import SRGANModel as M

    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
