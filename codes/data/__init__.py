'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    if mode == 'DRB':
        from data.DRB_dataset import DRBDataset as D
    elif mode == 'DRB_no_patch':
        from data.DRB_dataset_no_patch import DRBDataset_no_patch as D
    elif mode == 'LR':
        from data.LR_dataset import LRDataset as D
    elif mode == 'LQGT':
        from data.LQGT_dataset import LQGTDataset as D
    elif mode == 'RCAN':
        from data.LQGT_RCAN import LQGTDataset as D
    elif mode == 'EDSR':
        from data.LQGT_EDSR import LQGTDataset as D
    elif mode == 'DBPN':
        from data.LQGT_DBPN import LQGTDataset as D
    elif mode == 'vimeo':
        from data.LQGT_vimeo import LQGTDataset as D
    # elif mode == 'LQGTseg_bg':
    #     from data.LQGT_seg_bg_dataset import LQGTSeg_BG_Dataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
