# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

from .swin_transformer import build_swin
from .vision_transformer import build_vit
from .simmim import build_simmim
# from .pvit import build_pvit
from .pvit_plus import build_pvit
from .cvit import build_cvit


def build_model(config, logger=None, is_pretrain=True):
    model_type = config.MODEL.TYPE
    if is_pretrain:
        if model_type == 'pvit':
            model = build_pvit(config, logger)
        elif model_type == 'cvit':
            model = build_cvit(config, logger)
        else:
            model = build_simmim(config)
    else:
        if model_type == 'swin':
            model = build_swin(config)
        elif model_type in ['vit', 'pvit', 'cvit']:
            model = build_vit(config)
        else:
            raise NotImplementedError(f"Unknown fine-tune model: {model_type}")

    return model
